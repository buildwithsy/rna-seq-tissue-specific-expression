"""
analysis.py
ヒト組織別RNA-seq組織特異的発現解析

データ出典：
    Human Protein Atlas - RNA expression (consensus)
    https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip
    ライセンス：CC BY-SA 4.0（出典明記で自由に使用・改変・再配布可）
    取得日：2026-09-15
    HPA version 25.1 時点のデータ（ダウンロードページの表記に基づく）

内容：
    51組織 × 20,162遺伝子のnTPM（正規化発現量）。
    組織ごとに集計済みの値であり、個体・サンプル単位の生カウントデータではない。
    そのため、DESeq2等の古典的DEG検定（群間有意差検定）ではなく、
    PCA・階層的クラスタリング・組織特異性スコア(Tau index)による
    探索的解析を採用している。
    （方針の詳細な経緯は 02_意思決定ログ.md の2026-09-15の各エントリを参照）

実行方法：
    python analysis.py --input rna_tissue_consensus.tsv --outdir ./results

必要ライブラリ：
    pandas, numpy, scikit-learn, scipy, matplotlib
    （いずれもPython標準的なデータ分析環境に含まれる一般的なライブラリ。
    　SQLの実行にはPython標準ライブラリのsqlite3を使用しており、
    　別途データベースサーバーのインストールは不要）

設計メモ：
    データの前処理（低発現遺伝子・欠損遺伝子の除外）はSQL(SQLite)で行い、
    PCA・階層的クラスタリング・Tau index計算はPython(pandas/numpy/
    scikit-learn)で行う、という役割分担にしている。条件に合う行の
    抽出・集計はSQLの得意分野である一方、行列演算・線形代数を要する
    統計処理はSQL向きではないため。
"""

import argparse
import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist


# ---------------------------------------------------------------------------
# 設定値（暫定的な工学的判断のものは、その旨と根拠をコメントで明記）
# ---------------------------------------------------------------------------

# 全組織でほぼ発現していない遺伝子を除外する閾値。
# 暫定的な工学的判断（意思決定ログ 2026-09-15「PCA・クラスタリング前に、
# 全組織でほぼ発現していない遺伝子（最大nTPM<1）を除外する」参照）。
# 本プロジェクト固有の検証は未実施。
MIN_EXPRESSION_FOR_PCA = 1.0

# Tau index計算後、マーカー遺伝子として採用する最低発現量。
# 暫定的な工学的判断（意思決定ログ 2026-09-15「組織特異性スコア（Tau index）の
# ランキングに、最低発現量フィルタ（最大nTPM≥10）を追加する」参照）。
# フィルタなしではノイズ由来の見せかけの特異性（IFNA5/IFNA8等）が上位を
# 占めることを確認した上で導入。閾値10自体は経験的な値であり、
# 感度分析（5・20との比較）は未実施。
MIN_EXPRESSION_FOR_TAU = 10.0

# ヒートマップに表示するマーカー遺伝子数（＝対象組織数）
N_MARKER_GENES = 20


# ---------------------------------------------------------------------------
# データ読み込み・前処理
# ---------------------------------------------------------------------------

def load_and_filter_via_sql(tsv_path, min_expression=MIN_EXPRESSION_FOR_PCA):
    """
    HPAのlong形式tsv (Gene, Gene name, Tissue, nTPM) をSQLite（インメモリDB）
    に読み込み、以下2つのフィルタをSQLクエリで適用した上で、
    wide形式（行=遺伝子, 列=組織）に変換して返す。

    フィルタの内容と順序（意思決定ログ 2026-09-15の記録に準拠）：
      1. 低発現遺伝子の除外：全51組織における最大nTPMが閾値未満の遺伝子を除外
         （暫定的な工学的判断。本プロジェクト固有の検証は未実施）
      2. 欠損遺伝子の除外：51組織すべてにデータが揃っていない遺伝子を除外
         （脳の細かい部位10組織で特定119遺伝子が構造的に欠損していることを
         確認済み。平均値等での補完は行わず、単純に除外する方針とした）

    データの前処理（条件に合う行の抽出・集計）はSQLの得意分野であり、
    後段のPCA・階層的クラスタリング・Tau index計算（行列演算）は
    引き続きPython(pandas/numpy/scikit-learn)側で行う、という役割分担
    にしている。DBはファイルを介さないインメモリSQLiteを使うため、
    追加のミドルウェアのインストールは不要（Python標準ライブラリのsqlite3のみ）。

    注意：ダウンロード後にNumbers等の表計算ソフトで開くと、
    行数制限（100万行）により末尾データが欠落することを確認済み
    （意思決定ログ 2026-09-15参照）。このtsvはNumbers等を経由せず、
    ダウンロードしたファイルをそのまま使用すること。

    戻り値：(wide, stage_counts)
      wide：フィルタ後のwide形式DataFrame
      stage_counts：各段階の遺伝子数を記録した辞書（ログ表示用）
    """
    df = pd.read_csv(tsv_path, sep="\t")
    n_tissues = df["Tissue"].nunique()
    n_total = df["Gene"].nunique()

    conn = sqlite3.connect(":memory:")
    df.to_sql("expression", conn, index=False)

    # 段階1：低発現遺伝子の除外後の件数
    n_after_low_expr = pd.read_sql(
        f"""
        SELECT COUNT(*) AS n FROM (
            SELECT Gene FROM expression GROUP BY Gene HAVING MAX(nTPM) >= {min_expression}
        )
        """,
        conn,
    )["n"][0]

    # 段階2：欠損遺伝子も除外した最終結果（long形式）
    query = f"""
    WITH low_expr_filtered AS (
        SELECT Gene, "Gene name" AS gene_name, Tissue, nTPM
        FROM expression
        WHERE Gene IN (
            SELECT Gene FROM expression
            GROUP BY Gene
            HAVING MAX(nTPM) >= {min_expression}
        )
    ),
    complete_genes AS (
        SELECT Gene
        FROM low_expr_filtered
        GROUP BY Gene
        HAVING COUNT(DISTINCT Tissue) = {n_tissues}
    )
    SELECT lef.Gene, lef.gene_name, lef.Tissue, lef.nTPM
    FROM low_expr_filtered lef
    JOIN complete_genes cg ON lef.Gene = cg.Gene
    ORDER BY lef.Gene, lef.Tissue
    """
    filtered_long = pd.read_sql(query, conn)
    conn.close()

    wide = filtered_long.pivot_table(index=["Gene", "gene_name"], columns="Tissue", values="nTPM")
    wide.index = wide.index.set_names(["Gene", "Gene name"])

    stage_counts = {
        "total": n_total,
        "after_low_expression_filter": n_after_low_expr,
        "after_missing_value_filter": wide.shape[0],
    }
    return wide, stage_counts


# ---------------------------------------------------------------------------
# PCA・階層的クラスタリング
# ---------------------------------------------------------------------------

def run_pca(complete_log_data, n_components=5):
    """
    組織を観測単位（行）、遺伝子を変数（列）として標準化しPCAを行う。
    戻り値：PC1-3のDataFrame（index=組織）、寄与率の配列
    """
    X = complete_log_data.T  # 組織 x 遺伝子
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_scaled)
    pc_df = pd.DataFrame(pcs[:, :3], index=X.index, columns=["PC1", "PC2", "PC3"])
    return pc_df, pca.explained_variance_ratio_


def plot_pca(pc_df, explained_variance_ratio, outpath):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(pc_df["PC1"], pc_df["PC2"], s=60, alpha=0.7)
    for tissue, row in pc_df.iterrows():
        ax.annotate(tissue, (row["PC1"], row["PC2"]), fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}%)")
    ax.set_title("PCA of tissues based on transcriptome (Human Protein Atlas consensus)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_dendrogram(complete_log_data, outpath):
    """相関距離(1-相関係数)による階層的クラスタリング(average法)。"""
    X = complete_log_data.T
    dist = pdist(X.values, metric="correlation")
    Z = linkage(dist, method="average")

    fig, ax = plt.subplots(figsize=(10, 14))
    dendrogram(Z, labels=X.index.tolist(), orientation="left", ax=ax, leaf_font_size=9)
    ax.set_title("Hierarchical clustering of tissues\n(based on correlation of gene expression)")
    ax.set_xlabel("Distance (1 - correlation)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 組織特異性スコア（Tau index）
# ---------------------------------------------------------------------------

def tau_index(row):
    """
    組織特異性スコア Tau を計算する。
    定義：tau = sum(1 - x_i / x_max) / (n - 1)
        0に近い = どの組織でも均等に発現、1に近い = 1組織に特異的
    線形のnTPM値（対数変換前）で計算するのが標準的な扱い。
    """
    x = row.values
    xmax = x.max()
    if xmax == 0:
        return np.nan
    xi = x / xmax
    n = len(x)
    return np.sum(1 - xi) / (n - 1)


def compute_tau(complete_wide, min_expression=MIN_EXPRESSION_FOR_TAU):
    """
    Tau indexを計算し、ノイズ由来の見せかけの特異性を避けるため
    最低発現量フィルタを適用した上で降順に並べる。
    （経緯：意思決定ログ 2026-09-15「組織特異性スコア（Tau index）の
    ランキングに、最低発現量フィルタ（最大nTPM≥10）を追加する」）

    Tauが同点の場合、max_nTPM降順→Gene ID昇順で明示的にタイブレークする。
    これを指定しないと、入力データの行順（SQLクエリの結果順など、
    実行環境によって変わり得るもの）に依存して結果が変わってしまう
    ことが判明したため（意思決定ログ参照）。
    """
    tau = complete_wide.apply(tau_index, axis=1)
    max_expr = complete_wide.max(axis=1)
    result = pd.DataFrame({"tau": tau, "max_nTPM": max_expr})
    result = result.reset_index()  # Gene, Gene name を列に戻し、Gene ID順のタイブレークに使う
    robust = result[result["max_nTPM"] >= min_expression].sort_values(
        ["tau", "max_nTPM", "Gene"], ascending=[False, False, True]
    )
    robust = robust.set_index(["Gene", "Gene name"])
    return robust


def select_top_marker_per_tissue(complete_wide, robust_tau, n_tissues=N_MARKER_GENES):
    """Tau上位から、組織ごとに1つずつマーカー遺伝子を選ぶ。"""
    top_per_tissue = {}
    for gene_key in robust_tau.index:
        tissue = complete_wide.loc[gene_key].idxmax()
        if tissue not in top_per_tissue:
            top_per_tissue[tissue] = gene_key
        if len(top_per_tissue) >= n_tissues:
            break
    return top_per_tissue


def plot_marker_heatmap(complete_wide, top_per_tissue, outpath):
    """
    マーカー遺伝子 x 組織のヒートマップ。
    行(遺伝子)の順序に列(組織)の順序を合わせることで対角線状に整形する。
    """
    ordered_tissues = list(top_per_tissue.keys())
    ordered_genes = list(top_per_tissue.values())
    gene_names = [g[1] for g in ordered_genes]

    heat_data = complete_wide.loc[ordered_genes, ordered_tissues]
    heat_data.index = gene_names
    log_heat = np.log2(heat_data + 1)

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(log_heat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(log_heat.columns)))
    ax.set_xticklabels(log_heat.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(len(log_heat.index)))
    ax.set_yticklabels(log_heat.index, fontsize=9)
    ax.set_title("Top tissue-specific marker genes (rows and columns matched, log2(nTPM+1))")
    plt.colorbar(im, ax=ax, label="log2(nTPM+1)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def main(input_path, outdir):
    os.makedirs(outdir, exist_ok=True)

    # 1. 読み込み・前処理（低発現遺伝子の除外・欠損遺伝子の除外）をSQLで実行
    #    条件に合う行の抽出・集計はSQLの得意分野であるため、この段階のみ
    #    SQLite（インメモリDB）を使用する。以降のPCA・クラスタリング・
    #    Tau計算は行列演算が中心のためPython(pandas/numpy等)側で行う。
    complete_wide, stage_counts = load_and_filter_via_sql(input_path)
    log_data = np.log2(complete_wide + 1)
    print(f"[1/6] 読み込み完了: {stage_counts['total']}遺伝子 x {complete_wide.shape[1]}組織")
    print(f"[2/6] 低発現遺伝子除外後: {stage_counts['after_low_expression_filter']}遺伝子"
          f"（閾値: 最大nTPM >= {MIN_EXPRESSION_FOR_PCA}、SQLのHAVING句で抽出）")
    print(f"[3/6] 欠損値除外後: {stage_counts['after_missing_value_filter']}遺伝子"
          f"（除外: {stage_counts['after_low_expression_filter'] - stage_counts['after_missing_value_filter']}遺伝子、"
          f"SQLのCOUNT(DISTINCT Tissue)で全組織揃っている遺伝子のみ抽出）")

    # 4. PCA・階層的クラスタリング
    pc_df, explained_variance_ratio = run_pca(log_data)
    plot_pca(pc_df, explained_variance_ratio, os.path.join(outdir, "pca_plot.png"))
    plot_dendrogram(log_data, os.path.join(outdir, "dendrogram.png"))
    pc_df.to_csv(os.path.join(outdir, "pca_result.csv"))
    print(f"[4/6] PCA完了: PC1={explained_variance_ratio[0]*100:.1f}%, "
          f"PC2={explained_variance_ratio[1]*100:.1f}%")

    # 5. Tau index計算・マーカー遺伝子選定
    #    Tau自体は complete_wide（線形値、低発現フィルタ+欠損値除外済み）で計算する
    robust_tau = compute_tau(complete_wide)
    robust_tau.to_csv(os.path.join(outdir, "tau_index_robust.csv"))
    top_per_tissue = select_top_marker_per_tissue(complete_wide, robust_tau)
    print(f"[5/6] Tau index計算完了: フィルタ後候補 {len(robust_tau)}遺伝子、"
          f"マーカー選定 {len(top_per_tissue)}組織分")

    # 6. ヒートマップ
    plot_marker_heatmap(complete_wide, top_per_tissue, os.path.join(outdir, "marker_heatmap_diagonal.png"))
    print(f"[6/6] 可視化完了。出力先: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA-seq組織特異的発現解析")
    parser.add_argument("--input", required=True, help="rna_tissue_consensus.tsv のパス")
    parser.add_argument("--outdir", default="./results", help="出力先ディレクトリ")
    args = parser.parse_args()
    main(args.input, args.outdir)
