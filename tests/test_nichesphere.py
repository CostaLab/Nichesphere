"""
NicheSphere test suite
======================
Tests cover core functions in nichesphere.coloc, nichesphere.tl,
nichesphere.comm, and nichesphere.niche_stats using small synthetic datasets
that require no external files.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pytest

import nichesphere
import nichesphere.coloc as coloc
import nichesphere.tl as tl
import nichesphere.comm as comm
import nichesphere.niche_stats as niche_stats


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

CELL_TYPES = ["CM", "Fib", "Mac", "EC"]
N_CTS = len(CELL_TYPES)
SAMPLES = ["ctrl_1", "ctrl_2", "ctrl_3", "exp_1", "exp_2", "exp_3"]
N_SPOTS = 20


@pytest.fixture
def ct_props():
    """Synthetic cell type proportion matrix (spots × cell types)."""
    rng = np.random.default_rng(42)
    props = rng.dirichlet(alpha=np.ones(N_CTS), size=N_SPOTS)
    spot_ids = [f"spot_{i}" for i in range(N_SPOTS)]
    return pd.DataFrame(props, index=spot_ids, columns=CELL_TYPES)


@pytest.fixture
def spot_samples(ct_props):
    """Sample label per spot — first half ctrl_1, second half exp_1."""
    labels = ["ctrl_1"] * (N_SPOTS // 2) + ["exp_1"] * (N_SPOTS // 2)
    return pd.Series(labels, index=ct_props.index)


@pytest.fixture
def coloc_probs(ct_props, spot_samples):
    """CTcolocalizationP as returned by getColocProbs."""
    return coloc.getColocProbs(CTprobs=ct_props, spotSamples=spot_samples)


@pytest.fixture
def coloc_per_sample(coloc_probs):
    """colocPerSample as returned by reshapeColoc."""
    return coloc.reshapeColoc(CTcoloc=coloc_probs, complete=1)


@pytest.fixture
def sample_types_df():
    """Sample–condition mapping DataFrame."""
    return pd.DataFrame({
        "sample":     SAMPLES,
        "sampleType": ["ctrl", "ctrl", "ctrl", "exp", "exp", "exp"]
    })


@pytest.fixture
def multi_sample_coloc():
    """Larger colocPerSample with all 6 samples for diffColoc_test."""
    rng = np.random.default_rng(0)
    pairs = [f"{a}-{b}" for a in CELL_TYPES for b in CELL_TYPES]
    data = rng.dirichlet(alpha=np.ones(len(pairs)), size=len(SAMPLES))
    return pd.DataFrame(data, index=SAMPLES, columns=pairs)


@pytest.fixture
def niches_dict():
    return {
        "NicheA": ["CM", "Fib"],
        "NicheB": ["Mac", "EC"],
    }


@pytest.fixture
def niches_df(niches_dict):
    niche_colors = pd.Series(["#ff0000", "#0000ff"], index=["NicheA", "NicheB"])
    return tl.cells_niche_colors(
        CTs=CELL_TYPES,
        niche_colors=niche_colors,
        niche_dict=niches_dict
    )


@pytest.fixture
def adj():
    """Small signed adjacency matrix."""
    data = np.array([
        [ 0.0,  0.5, -0.3,  0.0],
        [ 0.5,  0.0,  0.2, -0.1],
        [-0.3,  0.2,  0.0,  0.4],
        [ 0.0, -0.1,  0.4,  0.0],
    ])
    return pd.DataFrame(data, index=CELL_TYPES, columns=CELL_TYPES)


@pytest.fixture
def signed_graph(adj):
    """NetworkX graph built from the signed adjacency matrix."""
    G = nx.from_pandas_adjacency(adj)
    return G


@pytest.fixture
def partition():
    return {"CM": "NicheA", "Fib": "NicheA", "Mac": "NicheB", "EC": "NicheB"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Import
# ─────────────────────────────────────────────────────────────────────────────

class TestImport:
    def test_package_imports(self):
        import nichesphere
        assert hasattr(nichesphere, "coloc")
        assert hasattr(nichesphere, "tl")
        assert hasattr(nichesphere, "comm")
        assert hasattr(nichesphere, "niche_stats")

    def test_submodule_functions_exist(self):
        assert callable(coloc.getColocProbs)
        assert callable(coloc.reshapeColoc)
        assert callable(coloc.diffColoc_test)
        assert callable(tl.cells_niche_colors)
        assert callable(tl.get_pairCatDFdir)
        assert callable(tl.compute_network_stats)
        assert callable(comm.equalizeScoresTables)
        assert callable(comm.diffCcommStats)
        assert callable(niche_stats.community_edge_stats_df)


# ─────────────────────────────────────────────────────────────────────────────
# 2. nichesphere.coloc
# ─────────────────────────────────────────────────────────────────────────────

class TestGetColocProbs:
    def test_returns_dataframe(self, coloc_probs):
        assert isinstance(coloc_probs, pd.DataFrame)

    def test_has_sample_column(self, coloc_probs):
        assert "sample" in coloc_probs.columns

    def test_cell_type_columns_present(self, coloc_probs):
        for ct in CELL_TYPES:
            assert ct in coloc_probs.columns

    def test_values_non_negative(self, coloc_probs):
        numeric = coloc_probs[CELL_TYPES]
        assert (numeric >= 0).all().all()

    def test_one_row_per_ct_per_sample(self, coloc_probs, spot_samples):
        n_samples = spot_samples.nunique()
        assert len(coloc_probs) == N_CTS * n_samples


class TestReshapeColoc:
    def test_returns_dataframe(self, coloc_per_sample):
        assert isinstance(coloc_per_sample, pd.DataFrame)

    def test_index_is_samples(self, coloc_per_sample, spot_samples):
        for s in spot_samples.unique():
            assert s in coloc_per_sample.index

    def test_columns_are_ct_pairs(self, coloc_per_sample):
        expected_pairs = [f"{a}-{b}" for a in CELL_TYPES for b in CELL_TYPES]
        for pair in expected_pairs:
            assert pair in coloc_per_sample.columns

    def test_rows_sum_to_one(self, coloc_per_sample):
        row_sums = coloc_per_sample.sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-6


class TestDiffColocTest:
    def test_returns_dataframe(self, multi_sample_coloc, sample_types_df):
        result = coloc.diffColoc_test(
            coloc_pair_sample=multi_sample_coloc,
            sampleTypes=sample_types_df,
            exp_condition="exp",
            ctrl_condition="ctrl"
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, multi_sample_coloc, sample_types_df):
        result = coloc.diffColoc_test(
            coloc_pair_sample=multi_sample_coloc,
            sampleTypes=sample_types_df,
            exp_condition="exp",
            ctrl_condition="ctrl"
        )
        assert "statistic" in result.columns
        assert "p-value" in result.columns

    def test_one_row_per_pair(self, multi_sample_coloc, sample_types_df):
        result = coloc.diffColoc_test(
            coloc_pair_sample=multi_sample_coloc,
            sampleTypes=sample_types_df,
            exp_condition="exp",
            ctrl_condition="ctrl"
        )
        assert len(result) == len(multi_sample_coloc.columns)

    def test_pvalues_in_range(self, multi_sample_coloc, sample_types_df):
        result = coloc.diffColoc_test(
            coloc_pair_sample=multi_sample_coloc,
            sampleTypes=sample_types_df,
            exp_condition="exp",
            ctrl_condition="ctrl"
        )
        pvals = result["p-value"].astype(float)
        assert (pvals >= 0).all() and (pvals <= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. nichesphere.tl
# ─────────────────────────────────────────────────────────────────────────────

class TestCellsNicheColors:
    def test_returns_dataframe(self, niches_df):
        assert isinstance(niches_df, pd.DataFrame)

    def test_has_required_columns(self, niches_df):
        for col in ["cell", "niche", "color"]:
            assert col in niches_df.columns

    def test_all_cell_types_present(self, niches_df):
        assert set(niches_df.cell) == set(CELL_TYPES)

    def test_niche_assignment(self, niches_df, niches_dict):
        for niche, members in niches_dict.items():
            for ct in members:
                assert niches_df.loc[ct, "niche"] == niche

    def test_niche_is_categorical(self, niches_df):
        assert hasattr(niches_df["niche"], "cat")


class TestPvalFilteredHMdf:
    def test_returns_square_dataframe(self, multi_sample_coloc, sample_types_df):
        test_df = coloc.diffColoc_test(
            coloc_pair_sample=multi_sample_coloc,
            sampleTypes=sample_types_df,
            exp_condition="exp",
            ctrl_condition="ctrl"
        )
        one_ct_ints = pd.Index([f"{ct}-{ct}" for ct in CELL_TYPES])
        hm = tl.pval_filtered_HMdf(
            testDF=test_df, oneCTinteractions=one_ct_ints,
            p=1.0, cell_types=CELL_TYPES
        )
        assert hm.shape == (N_CTS, N_CTS)

    def test_same_ct_pairs_are_zero(self, multi_sample_coloc, sample_types_df):
        test_df = coloc.diffColoc_test(
            coloc_pair_sample=multi_sample_coloc,
            sampleTypes=sample_types_df,
            exp_condition="exp",
            ctrl_condition="ctrl"
        )
        one_ct_ints = pd.Index([f"{ct}-{ct}" for ct in CELL_TYPES])
        hm = tl.pval_filtered_HMdf(
            testDF=test_df, oneCTinteractions=one_ct_ints,
            p=1.0, cell_types=CELL_TYPES
        )
        for ct in CELL_TYPES:
            assert hm.loc[ct, ct] == 0.0

    def test_strict_pval_zeroes_all(self, multi_sample_coloc, sample_types_df):
        """With p=0 everything should be zeroed out."""
        test_df = coloc.diffColoc_test(
            coloc_pair_sample=multi_sample_coloc,
            sampleTypes=sample_types_df,
            exp_condition="exp",
            ctrl_condition="ctrl"
        )
        one_ct_ints = pd.Index([f"{ct}-{ct}" for ct in CELL_TYPES])
        hm = tl.pval_filtered_HMdf(
            testDF=test_df, oneCTinteractions=one_ct_ints,
            p=0.0, cell_types=CELL_TYPES
        )
        assert (hm == 0).all().all()


class TestGetPairCatDFdir:
    def test_returns_dataframe(self, niches_df):
        result = tl.get_pairCatDFdir(niches_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, niches_df):
        result = tl.get_pairCatDFdir(niches_df)
        assert "cell_pairs" in result.columns
        assert "niche_pairs" in result.columns

    def test_n_rows(self, niches_df):
        result = tl.get_pairCatDFdir(niches_df)
        assert len(result) == N_CTS ** 2

    def test_intra_niche_pairs_correct(self, niches_df, niches_dict):
        result = tl.get_pairCatDFdir(niches_df)
        # CM->CM should map to NicheA->NicheA
        row = result[result.cell_pairs == "CM->CM"]
        assert row["niche_pairs"].values[0] == "NicheA->NicheA"


class TestGetColocFilter:
    def test_returns_dataframe(self, niches_df, adj):
        pair_cat = tl.get_pairCatDFdir(niches_df)
        # Subset adj to only CT pairs in pairCatDF
        one_ct_ints = pd.Index([f"{ct}->{ct}" for ct in CELL_TYPES])
        result = tl.getColocFilter(pairCatDF=pair_cat, adj=adj, oneCTints=one_ct_ints)
        assert isinstance(result, pd.DataFrame)

    def test_filter_column_is_binary(self, niches_df, adj):
        pair_cat = tl.get_pairCatDFdir(niches_df)
        one_ct_ints = pd.Index([f"{ct}->{ct}" for ct in CELL_TYPES])
        result = tl.getColocFilter(pairCatDF=pair_cat, adj=adj, oneCTints=one_ct_ints)
        assert set(result["filter"].astype(int).unique()).issubset({0, 1})


class TestProcessCTKRoutput:
    def test_removes_suffixes(self):
        df = pd.DataFrame({
            "gene_A":  ["FN1|L",  "COL1A1|L", "TGFB1|L"],
            "gene_B":  ["ITGA5|R", "DDR1|R",   "TGFBR1|R"],
            "allpair": ["FN1|L/ITGA5|R/CellA/CellB",
                        "COL1A1|L/DDR1|R/CellA/CellB",
                        "TGFB1|L/TGFBR1|R/CellA/CellB"],
            "source":  ["CellA", "CellA", "CellB"],
            "target":  ["CellB", "CellB", "CellA"],
            "MeanLR":  [1.0, 0.5, 0.8],
        })
        result = tl.processCTKRoutput(df)
        assert not result["gene_A"].str.contains(r"\|L").any()
        assert not result["gene_B"].str.contains(r"\|R").any()
        assert not result["allpair"].str.contains(r"\|L|\|R|\|TF").any()


class TestComputeNetworkStats:
    def test_returns_dataframe(self, signed_graph):
        result = tl.compute_network_stats(signed_graph)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, signed_graph):
        result = tl.compute_network_stats(signed_graph)
        for col in ["betweenness", "pagerank", "degree_positive", "degree_negative"]:
            assert col in result.columns

    def test_index_matches_nodes(self, signed_graph):
        result = tl.compute_network_stats(signed_graph)
        assert set(result.index) == set(signed_graph.nodes)

    def test_degree_non_negative(self, signed_graph):
        result = tl.compute_network_stats(signed_graph)
        assert (result["degree_positive"] >= 0).all()
        assert (result["degree_negative"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. nichesphere.comm
# ─────────────────────────────────────────────────────────────────────────────

def _make_scores_table(condition, cell_pairs, categories, seed=0):
    """Helper: build a minimal lr_ctPairScores table."""
    rng = np.random.default_rng(seed)
    rows = []
    for pair in cell_pairs:
        for cat in categories:
            rows.append({
                "cell_pairs":  pair,
                "niche_pairs": "NicheA->NicheB",
                "LRcat":       cat,
                "LRscores":    rng.uniform(-1, 1),
                "condition":   condition,
            })
    df = pd.DataFrame(rows)
    df.index = [f"{r['cell_pairs']}/{r['LRcat']}" for _, r in df.iterrows()]
    return df


CELL_PAIRS  = ["CM->Fib", "Fib->Mac", "Mac->EC"]
LR_CATS     = ["ECM", "Cytokine"]


@pytest.fixture
def scores_ctrl():
    return _make_scores_table("ctrl", CELL_PAIRS, LR_CATS, seed=0)


@pytest.fixture
def scores_exp():
    return _make_scores_table("exp", CELL_PAIRS, LR_CATS, seed=1)


class TestEqualizeScoresTables:
    def test_returns_two_dataframes(self, scores_ctrl, scores_exp):
        ctrl_out, exp_out = comm.equalizeScoresTables(
            ctrlTbl=scores_ctrl, expTbl=scores_exp,
            ctrlCondition="ctrl", expCondition="exp"
        )
        assert isinstance(ctrl_out, pd.DataFrame)
        assert isinstance(exp_out, pd.DataFrame)

    def test_same_index_after_equalization(self, scores_ctrl, scores_exp):
        ctrl_out, exp_out = comm.equalizeScoresTables(
            ctrlTbl=scores_ctrl, expTbl=scores_exp,
            ctrlCondition="ctrl", expCondition="exp"
        )
        assert set(ctrl_out.index) == set(exp_out.index)


class TestDiffCcommStats:
    def test_returns_dataframe(self, scores_ctrl, scores_exp):
        ctrl_eq, exp_eq = comm.equalizeScoresTables(
            ctrlTbl=scores_ctrl, expTbl=scores_exp,
            ctrlCondition="ctrl", expCondition="exp"
        )
        result = comm.diffCcommStats(
            c1CTpairScores_byCat=exp_eq,
            c2CTpairScores_byCat=ctrl_eq,
            cellCatCol="cell_pairs"
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, scores_ctrl, scores_exp):
        ctrl_eq, exp_eq = comm.equalizeScoresTables(
            ctrlTbl=scores_ctrl, expTbl=scores_exp,
            ctrlCondition="ctrl", expCondition="exp"
        )
        result = comm.diffCcommStats(
            c1CTpairScores_byCat=exp_eq,
            c2CTpairScores_byCat=ctrl_eq,
            cellCatCol="cell_pairs"
        )
        for col in ["wilcoxStat", "wilcoxPval", "cellCat", "LRcat"]:
            assert col in result.columns

    def test_one_row_per_pair_per_category(self, scores_ctrl, scores_exp):
        ctrl_eq, exp_eq = comm.equalizeScoresTables(
            ctrlTbl=scores_ctrl, expTbl=scores_exp,
            ctrlCondition="ctrl", expCondition="exp"
        )
        result = comm.diffCcommStats(
            c1CTpairScores_byCat=exp_eq,
            c2CTpairScores_byCat=ctrl_eq,
            cellCatCol="cell_pairs"
        )
        assert len(result) == len(CELL_PAIRS) * len(LR_CATS)


# ─────────────────────────────────────────────────────────────────────────────
# 5. nichesphere.niche_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestCommunityEdgeStatsDf:
    def test_returns_dataframe(self, signed_graph, partition):
        result = niche_stats.community_edge_stats_df(signed_graph, partition)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_communities(self, signed_graph, partition):
        result = niche_stats.community_edge_stats_df(signed_graph, partition)
        assert set(result.index) == {"NicheA", "NicheB"}

    def test_has_count_and_weight_columns(self, signed_graph, partition):
        result = niche_stats.community_edge_stats_df(signed_graph, partition)
        for col in ["internal_pos_count", "internal_neg_count",
                    "external_pos_count", "external_neg_count"]:
            assert col in result.columns

    def test_counts_non_negative(self, signed_graph, partition):
        result = niche_stats.community_edge_stats_df(signed_graph, partition)
        count_cols = [c for c in result.columns if "count" in c]
        assert (result[count_cols] >= 0).all().all()


class TestPairwiseRanksumsIntExt:
    def test_returns_dataframe(self):
        data_int = [[0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4]]
        data_ext = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        communities = ["NicheA", "NicheB"]
        result = niche_stats.pairwise_ranksums_int_ext(data_int, data_ext, communities)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        data_int = [[0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4]]
        data_ext = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        communities = ["NicheA", "NicheB"]
        result = niche_stats.pairwise_ranksums_int_ext(data_int, data_ext, communities)
        for col in ["niche", "statistic", "p_value", "p_value_corrected"]:
            assert col in result.columns

    def test_pvalues_in_range(self):
        data_int = [[0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4]]
        data_ext = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        communities = ["NicheA", "NicheB"]
        result = niche_stats.pairwise_ranksums_int_ext(data_int, data_ext, communities)
        assert (result["p_value"] >= 0).all() and (result["p_value"] <= 1).all()
        assert (result["p_value_corrected"] >= 0).all() and (result["p_value_corrected"] <= 1).all()

    def test_skips_communities_with_too_few_edges(self):
        """Communities with < 2 edges in either distribution should be skipped."""
        data_int = [[0.5], [0.1, 0.2, 0.3, 0.4]]   # NicheA has only 1 value
        data_ext = [[0.1], [0.5, 0.6, 0.7, 0.8]]
        communities = ["NicheA", "NicheB"]
        result = niche_stats.pairwise_ranksums_int_ext(data_int, data_ext, communities)
        assert "NicheA" not in result["niche"].values
        assert "NicheB" in result["niche"].values


# ─────────────────────────────────────────────────────────────────────────────
# 6. nichesphere.database
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadDB:
    def test_load_db_returns_dataframe(self):
        db = nichesphere.load_DB()
        assert isinstance(db, pd.DataFrame)

    def test_load_db_not_empty(self):
        db = nichesphere.load_DB()
        assert len(db) > 0

    def test_load_db_has_expected_columns(self):
        db = nichesphere.load_DB()
        assert "Ligand" in db.columns
        assert "category" in db.columns
