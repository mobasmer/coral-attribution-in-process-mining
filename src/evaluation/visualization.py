
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, ConstantInputWarning

RESULTS_DIR = "../../data/results"

# Set font size
SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ============================================================================
# Lineage/value-computation timing (data/results/*/lineage_timing_*.csv +
# data/results/lineage_bpifull/runtime_summary.csv)
# ============================================================================

GRANULARITY_ORDER = ["event", "context", "relation"]
STEP_LABELS = {
    "build_provenance": "Build provenance",
    "compute_lineage": "Compute lineage",
    "compute_values": "Compute values",
}
LINEAGE_TYPE_ORDER = ["alternate_response_viol", "simple_handover", "freq_handover"]
# lineage_type -> analysis label. "*" flags the frequency-based variant of a metric that also
# has a "simple" variant (see convert_lineage_to_input.py's `selection` dict: "efg"/"simple_efg"
# and "simple_handover" are the plain queries, "freq_efg"/"freq_handover" the frequency ones).
# A lineage_type not listed here (e.g. "alternate_precedence_viol") is left as its raw name
# rather than guessed at - add it here if it should be folded into one of these labels instead.
LINEAGE_TYPE_TO_ANALYSIS = {
    "alternate_response_viol": "Conformance",
    "simple_handover": "Handover",
    "freq_handover": "Handover*",
    "efg": "EFG",
    "simple_efg": "EFG",
    "freq_efg": "EFG*",
}
SCHEMA_ORDER = ["workflow", "application", "offer"]
SCHEMA_CODES = {"workflow": "W", "application": "A", "offer": "O"}
ALGORITHM_ORDER = ["banzhaf", "shapley"]
ALGORITHM_LABELS = {"banzhaf": "Banzhaf", "shapley": "Shapley"}

COMBINED_ANALYSIS_ORDER = ["Conformance", "Handover", "EFG"]
# combined_timing_table/_latex show exactly one canonical lineage_type per analysis (unlike
# LINEAGE_TYPE_TO_ANALYSIS, which shows both the simple and frequency variant) - candidates are
# tried in order and the first with any data for a given granularity is used. "efg"/"simple_efg"
# are the same underlying query under different naming across convert_lineage_to_input.py
# versions.
COMBINED_ANALYSIS_LINEAGE_TYPES = {
    "Conformance": ["alternate_response_viol"],
    "Handover": ["freq_handover"],
    "EFG": ["simple_efg", "efg"],
}

def load_value_computation_timing(results_dir: str = RESULTS_DIR, pattern: str = "time_eval_*.csv") -> pd.DataFrame:
    """Load and concatenate every value-computation-timing CSV matching `pattern` in
    `results_dir` (LExaBan = banzhaf, LExaShap = shapley runtimes)."""
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in '{results_dir}'")
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    df["lineage_type"] = df["lineage_type"].str.removesuffix("_lineage")
    df["algorithm"] = df["algorithm"].map({"LExaBan": "banzhaf", "LExaShap": "shapley"})
    return df


def load_lineage_timing(results_dir: str = RESULTS_DIR, pattern: str = "lineage_timing_*.csv") -> pd.DataFrame:
    """Load and concatenate all lineage timing CSVs matching `pattern` in `results_dir`."""
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in '{results_dir}'")
    return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)


def _pipeline_sort_key(index_tuple):
    granularity, step, lineage_type = index_tuple
    return (
        GRANULARITY_ORDER.index(granularity) if granularity in GRANULARITY_ORDER else len(GRANULARITY_ORDER),
        list(STEP_LABELS).index(step) if step in STEP_LABELS else len(STEP_LABELS),
        -1 if lineage_type == "" else
        LINEAGE_TYPE_ORDER.index(lineage_type) if lineage_type in LINEAGE_TYPE_ORDER else len(LINEAGE_TYPE_ORDER),
    )


def _value_computation_sort_key(index_tuple):
    granularity, lineage_type, algorithm = index_tuple
    return (
        GRANULARITY_ORDER.index(granularity) if granularity in GRANULARITY_ORDER else len(GRANULARITY_ORDER),
        LINEAGE_TYPE_ORDER.index(lineage_type) if lineage_type in LINEAGE_TYPE_ORDER else len(LINEAGE_TYPE_ORDER),
        ALGORITHM_ORDER.index(algorithm) if algorithm in ALGORITHM_ORDER else len(ALGORITHM_ORDER),
    )


def _sorted_schema_columns(pivot: pd.DataFrame) -> pd.DataFrame:
    pivot = pivot.reindex(columns=sorted(
        pivot.columns, key=lambda c: SCHEMA_ORDER.index(c) if c in SCHEMA_ORDER else len(SCHEMA_ORDER)
    ))
    pivot.columns = pivot.columns.map(str.capitalize)
    pivot.columns.name = "Schema"
    return pivot


def _pipeline_timing_stats(results_dir: str, pattern: str):
    """
    Load every lineage_timing CSV matching `pattern` in `results_dir` and group by (
    granularity, step, lineage_type, schema) - the same combination is commonly measured by
    more than one file (e.g. a rerun to fill in a cell that previously timed out). Returns
    three Series sharing that index: the mean duration_seconds, whether every contributing run
    timed out, and whether any did (files predating the "timed_out" column are treated as not
    timed out).
    """
    df = load_lineage_timing(results_dir, pattern)
    df = df.copy()
    df["lineage_type"] = df["lineage_type"].fillna("")
    if "timed_out" not in df.columns:
        df["timed_out"] = False
    df["timed_out"] = df["timed_out"].fillna(False).astype(bool)

    grouped = df.groupby(["granularity", "step", "lineage_type", "schema"])
    return grouped["duration_seconds"].mean(), grouped["timed_out"].all(), grouped["timed_out"].any()


def _finalize_pipeline_pivot(series: pd.Series) -> pd.DataFrame:
    """Reshape a (granularity, step, lineage_type, schema)-indexed Series (see
    _pipeline_timing_stats) into the standard pipeline-timing shape: schema as columns (sorted,
    capitalized), rows sorted by _pipeline_sort_key and relabeled (step -> STEP_LABELS,
    lineage_type -> analysis label via LINEAGE_TYPE_TO_ANALYSIS)."""
    pivot = series.unstack("schema")
    pivot = pivot.reindex(sorted(pivot.index, key=_pipeline_sort_key))
    pivot = _sorted_schema_columns(pivot)
    pivot = pivot.rename(index=STEP_LABELS, level="step")
    pivot = pivot.rename(index=LINEAGE_TYPE_TO_ANALYSIS, level="lineage_type")
    pivot.index = pivot.index.set_names(["Granularity", "Step", "Analysis"])
    return pivot


def pipeline_timing_table(
    results_dir: str = RESULTS_DIR,
    pattern: str = "lineage_timing_*.csv",
) -> pd.DataFrame:
    """
    Runtime (seconds) of the two lineage-*building* steps (build_provenance, compute_lineage)
    per granularity, step and analysis, one column per schema, averaged across every file
    matching `pattern` in `results_dir` (see _pipeline_timing_stats). A cell where every
    contributing run timed out has no real duration to average - only the timeout cutoff - so
    it's reported as missing (NaN here, rendered "--" by pipeline_timing_latex) rather than as
    a misleadingly capped mean; see pipeline_timing_latex for cells where only *some* runs
    timed out. Kept separate from value_computation_timing_table (rather than one combined
    table indexed by [Granularity, Step, Analysis, Algorithm]) because Algorithm only applies
    to compute_values - folding it in here would leave that index level blank on every row.
    """
    mean_duration, all_timed_out, _ = _pipeline_timing_stats(results_dir, pattern)
    return _finalize_pipeline_pivot(mean_duration.where(~all_timed_out))


def value_computation_timing_table(
    results_dir: str = RESULTS_DIR,
    pattern: str = "time_eval_*.csv",
) -> pd.DataFrame:
    """
    Runtime (seconds) of the compute_values step (Shapley/Banzhaf value computation) per
    granularity, lineage type and algorithm, one column per schema, averaged across every file
    matching `pattern` in `results_dir` (repeated trials of the same combination are common -
    see pipeline_timing_table for the analogous build_provenance/compute_lineage timing).
    """
    df = load_value_computation_timing(results_dir, pattern)

    pivot = df.pivot_table(
        index=["granularity", "lineage_type", "algorithm"],
        columns="schema",
        values="duration_seconds",
        aggfunc="mean",
    )
    pivot = pivot.reindex(sorted(pivot.index, key=_value_computation_sort_key))
    pivot = _sorted_schema_columns(pivot)

    pivot = pivot.rename(index=ALGORITHM_LABELS, level="algorithm")
    pivot.index = pivot.index.set_names(["Granularity", "Lineage type", "Algorithm"])

    return pivot


def _timing_table_to_latex(table: pd.DataFrame, output_path: str | None, caption: str, label: str) -> str:
    latex = table.to_latex(
        float_format="%.2f",
        na_rep="--",
        escape=True,
        multirow=True,
        multicolumn=True,
        multicolumn_format="c",
        column_format="l" * table.index.nlevels + "r" * len(table.columns),
        caption=caption,
        label=label,
        position="ht",
    )

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)

    return latex


def pipeline_timing_latex(
    results_dir: str = RESULTS_DIR,
    pattern: str = "lineage_timing_*.csv",
    output_path: str | None = None,
    caption: str = "Runtime (in seconds) of the provenance-building and lineage-computation "
                    "steps, per schema, granularity and analysis, averaged over every matching "
                    "run. \"--\" marks a cell where every run timed out (no real duration was "
                    "recorded); \"*\" marks one where only some runs did (the average likely "
                    "understates the true runtime, since a timed-out run is capped at the "
                    "timeout length).",
    label: str = "tab:pipeline-timing",
) -> str:
    """
    Render pipeline_timing_table (see there for the averaging/all-timed-out-> "--" behavior) as
    a LaTeX table, additionally suffixing a cell with "*" if some but not all of the runs
    contributing to its average timed out - such a cell is still averaged over every run
    (including the timeout-capped ones), not just the completed ones, so the "*" is a caution
    that the shown value likely understates the true runtime rather than a note that data was
    dropped. Optionally writes the result to `output_path`.
    """
    mean_duration, all_timed_out, any_timed_out = _pipeline_timing_stats(results_dir, pattern)
    duration = _finalize_pipeline_pivot(mean_duration.where(~all_timed_out))
    partial_timeout = _finalize_pipeline_pivot(any_timed_out & ~all_timed_out).fillna(False).astype(bool)

    value_strs = duration.apply(lambda col: col.map(lambda v: "--" if pd.isna(v) else f"{v:.2f}"))
    marked_strs = value_strs.apply(lambda col: col + "*")
    table = marked_strs.where(partial_timeout, value_strs)

    latex = table.to_latex(
        na_rep="--",
        escape=True,
        multirow=True,
        multicolumn=True,
        multicolumn_format="c",
        column_format="l" * table.index.nlevels + "r" * len(table.columns),
        caption=caption,
        label=label,
        position="ht",
    )

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)

    return latex


def value_computation_timing_latex(
    results_dir: str = RESULTS_DIR,
    pattern: str = "time_eval_*.csv",
    output_path: str | None = None,
    caption: str = "Runtime (in seconds) of Shapley/Banzhaf value computation, per schema, "
                    "granularity, lineage type and algorithm, averaged over every matching run.",
    label: str = "tab:value-computation-timing",
) -> str:
    """Render value_computation_timing_table as a LaTeX table, optionally writing it to `output_path`."""
    table = value_computation_timing_table(results_dir, pattern)
    return _timing_table_to_latex(table, output_path, caption, label)


def _combined_timing_rows(
    pipeline_results_dir: str, pipeline_pattern: str,
    value_results_dir: str, value_pattern: str,
) -> list[dict]:
    """
    Shared data-gathering for combined_timing_table/combined_timing_latex: one row per
    granularity's "Initialize" (build_provenance) step, plus one row per analysis in
    COMBINED_ANALYSIS_ORDER that has any compute_lineage data for that granularity (via its
    canonical lineage_type - see COMBINED_ANALYSIS_LINEAGE_TYPES). Each row carries, per group
    ("Init / Provenance", "Banzhaf", "Shapley"), either None - the group doesn't apply to this
    row at all (Banzhaf/Shapley for "Initialize", or an algorithm never computed for this row's
    lineage_type in any schema) - or a list of 3 values in SCHEMA_ORDER (NaN where that specific
    schema has no completed run).
    """
    mean_duration, all_timed_out, _ = _pipeline_timing_stats(pipeline_results_dir, pipeline_pattern)
    pipeline = mean_duration.where(~all_timed_out)

    value_df = load_value_computation_timing(value_results_dir, value_pattern)
    value_mean = value_df.groupby(["granularity", "lineage_type", "algorithm", "schema"])["duration_seconds"].mean()

    def schema_values(series: pd.Series, index_prefix: tuple):
        keys = [index_prefix + (schema,) for schema in SCHEMA_ORDER]
        if not any(k in series.index for k in keys):
            return None
        return [series.get(k, float("nan")) for k in keys]

    rows = []
    for granularity in GRANULARITY_ORDER:
        init_values = schema_values(pipeline, (granularity, "build_provenance", ""))
        if init_values is not None:
            rows.append({
                "Granularity": granularity, "Row": "Initialize",
                "Init / Provenance": init_values, "Banzhaf": None, "Shapley": None,
            })

        for analysis in COMBINED_ANALYSIS_ORDER:
            lineage_type = next(
                (lt for lt in COMBINED_ANALYSIS_LINEAGE_TYPES[analysis]
                 if schema_values(pipeline, (granularity, "compute_lineage", lt)) is not None),
                None,
            )
            if lineage_type is None:
                continue

            rows.append({
                "Granularity": granularity, "Row": analysis,
                "Init / Provenance": schema_values(pipeline, (granularity, "compute_lineage", lineage_type)),
                "Banzhaf": schema_values(value_mean, (granularity, lineage_type, "banzhaf")),
                "Shapley": schema_values(value_mean, (granularity, lineage_type, "shapley")),
            })

    return rows


def combined_timing_table(
    pipeline_results_dir: str = RESULTS_DIR,
    pipeline_pattern: str = "lineage_timing_*.csv",
    value_results_dir: str = RESULTS_DIR,
    value_pattern: str = "time_eval_*.csv",
) -> pd.DataFrame:
    """
    Combine pipeline_timing_table's provenance/lineage-building timings and
    value_computation_timing_table's Shapley/Banzhaf timings into the paper's combined runtime
    table shape: one row per granularity's "Initialize" (build_provenance) step and one row per
    analysis (Conformance/Handover/EFG - each shown via a single canonical lineage_type, see
    COMBINED_ANALYSIS_LINEAGE_TYPES, unlike pipeline_timing_table which shows every variant);
    columns are (Init / Provenance, Banzhaf, Shapley) x (W, A, O) schema. NaN either means that
    schema had no completed run, or that the whole algorithm was never computed for this row's
    lineage_type - combined_timing_latex renders the former as "--" and the latter as a blank
    cell; this numeric table doesn't distinguish the two (see combined_timing_latex if that
    distinction matters for your use).
    """
    rows = _combined_timing_rows(pipeline_results_dir, pipeline_pattern, value_results_dir, value_pattern)
    if not rows:
        raise ValueError("no data found for the combined timing table")

    codes = list(SCHEMA_CODES.values())
    columns = pd.MultiIndex.from_product([["Init / Provenance", "Banzhaf", "Shapley"], codes])
    index = pd.MultiIndex.from_tuples([(r["Granularity"], r["Row"]) for r in rows], names=["Granularity", "Row"])

    data = [
        (r["Init / Provenance"] or [float("nan")] * 3) +
        (r["Banzhaf"] or [float("nan")] * 3) +
        (r["Shapley"] or [float("nan")] * 3)
        for r in rows
    ]
    return pd.DataFrame(data, index=index, columns=columns)


def combined_timing_latex(
    pipeline_results_dir: str = RESULTS_DIR,
    pipeline_pattern: str = "lineage_timing_*.csv",
    value_results_dir: str = RESULTS_DIR,
    value_pattern: str = "time_eval_*.csv",
    output_path: str | None = None,
    caption: str = "Runtime (seconds) of provenance/lineage-computation steps and "
                    "Shapley/Banzhaf value computation, per object type (W[orkflow], "
                    "A[pplication], O[ffer]), granularity and analysis. The sign -- indicates "
                    "a timeout.",
    label: str = "tab:combined-timing",
) -> str:
    """
    Render combined_timing_table (see there) as the hand-built LaTeX table used in the paper:
    \\multirow granularity labels, a \\cline separating each granularity's "Initialize" row from
    its analysis rows, \\multicolumn/\\cmidrule group headers for Init/Provenance/Banzhaf/
    Shapley, and a blank cell (rather than "--") wherever a whole algorithm was never computed
    for a row's lineage_type at all - "--" is reserved for a schema that specifically has no
    completed run (a timeout, in the Init/Provenance columns - see pipeline_timing_table).
    to_latex() can't produce this shape (irregular per-row blanks + a custom multi-level
    header), so it's built directly here rather than going through a DataFrame. Optionally
    writes the result to `output_path`.
    """
    rows = _combined_timing_rows(pipeline_results_dir, pipeline_pattern, value_results_dir, value_pattern)
    if not rows:
        raise ValueError("no data found for the combined timing table")

    def fmt_group(values):
        if values is None:
            return ["", "", ""]
        return ["--" if pd.isna(v) else f"{v:.2f}" for v in values]

    lines = [
        r"\begin{table}[htb]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        r"    \begin{tabular}{ll rrr rrr rrr}",
        r"    \toprule",
        r"    & & \multicolumn{3}{c}{Init / Provenance} & \multicolumn{3}{c}{Banzhaf} & \multicolumn{3}{c}{Shapley} \\",
        r"    \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}",
        r"    Gran. & Analysis & \texttt{W} & \texttt{A} & \texttt{O}"
        r" & \texttt{W} & \texttt{A} & \texttt{O} & \texttt{W} & \texttt{A} & \texttt{O} \\",
        r"    \midrule",
    ]

    for g_idx, granularity in enumerate(GRANULARITY_ORDER):
        granularity_rows = [r for r in rows if r["Granularity"] == granularity]
        if not granularity_rows:
            continue
        if g_idx > 0:
            lines.append(r"    \midrule")

        lines.append(rf"    \multirow[t]{{{len(granularity_rows)}}}{{*}}{{{granularity}}}")
        for row in granularity_rows:
            label_text = "Initialize" if row["Row"] == "Initialize" else rf"\textsc{{{row['Row']}}}"
            cells = fmt_group(row["Init / Provenance"]) + fmt_group(row["Banzhaf"]) + fmt_group(row["Shapley"])
            lines.append(f"     & {label_text} & " + " & ".join(cells) + r" \\")
            if row["Row"] == "Initialize":
                lines.append(r"     \cline{2-5}")

    lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"    \end{table}",
    ]

    latex = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)

    return latex


# ============================================================================
# eval_naive_and_sampling.py results (data/results/<timestamp>/{runs,values}.csv)
# ============================================================================

EVAL_ANALYSIS_ORDER = ["handover", "efg", "conformance"]
ANALYSIS_LABELS = {"handover": "Handover", "efg": "EFG", "conformance": "Conformance"}

VALUE_LABELS = {"shapley": "Shapley", "banzhaf": "Banzhaf", "owen": "Owen", "banzhaf-owen": "Banzhaf-Owen"}
# fixed color + linestyle per value method: color alone doesn't clear the CVD
# floor for 4 series shown side by side (see dataviz skill, palette.md), so
# linestyle carries identity too
EVAL_VALUE_STYLE = {
    "shapley": {"color": "#2a78d6", "linestyle": "-"},
    "banzhaf": {"color": "#eb6834", "linestyle": "--"},
    "owen": {"color": "#1baf7a", "linestyle": "-."},
    "banzhaf-owen": {"color": "#eda100", "linestyle": ":"},
}

SAMPLING_SCENARIO_COLUMNS = ["log", "case", "log_size", "value", "analysis"]
SAMPLING_METRIC_COLUMNS = {"spearman": "Spearman's rho", "kendall": "Kendall's tau"}


def load_eval_results(results_dir: str = RESULTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate every runs.csv/values.csv pair written by eval_naive_and_sampling.py under `results_dir/<timestamp>/`."""
    run_paths = sorted(glob.glob(os.path.join(results_dir, "*", "runs.csv")))
    value_paths = sorted(glob.glob(os.path.join(results_dir, "*", "values.csv")))
    if not run_paths or not value_paths:
        raise FileNotFoundError(f"No */runs.csv or */values.csv found under '{results_dir}/'")
    runs = pd.concat((pd.read_csv(p) for p in run_paths), ignore_index=True)
    values = pd.concat((pd.read_csv(p) for p in value_paths), ignore_index=True)
    return runs, values


def load_run_dir(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the single runs.csv/values.csv pair written by one eval_naive_and_sampling.py invocation, e.g. `run_dir='data/results/20260918_015755'`."""
    runs_path = os.path.join(run_dir, "runs.csv")
    values_path = os.path.join(run_dir, "values.csv")
    if not os.path.exists(runs_path) or not os.path.exists(values_path):
        raise FileNotFoundError(f"runs.csv/values.csv not found in '{run_dir}/'")
    return pd.read_csv(runs_path), pd.read_csv(values_path)


# ---- small helpers shared by the tables/plots below --------------------------------------

def _ok(runs: pd.DataFrame) -> pd.DataFrame:
    """Rows of a runs.csv/table where the computation completed successfully."""
    return runs[runs["status"] == "ok"]


def _ordered(order, present) -> list:
    """`order` filtered down to the elements also found in `present`, keeping `order`'s sequence."""
    present = set(present)
    return [x for x in order if x in present]


def _pick_one(available, chosen, what: str, context: str = "", required: bool = True):
    """
    Resolve a dimension (case/analysis/...) to a single value: if `chosen` is given, check it's
    actually available; if not given and exactly one value is available, use that; otherwise
    (ambiguous, or `required=False` and nothing was chosen) either raise a clear error or, for
    an optional dimension, return None so the caller can fall back to "show everything".
    """
    available = sorted(available)
    if chosen is None:
        if not required:
            return None
        if len(available) != 1:
            raise ValueError(f"ambiguous `{what}`{context}: pass one of {available}")
        return available[0]
    if chosen not in available:
        raise ValueError(f"no completed run for {what}={chosen!r}{context}; available: {available}")
    return chosen


# ---- plot-building helpers shared by sampling_accuracy_plot/sampling_runtime_plot --------

def _facet_grid(n_rows: int, n_cols: int, cell_w: float = 5, cell_h: float = 2.8):
    """A grid of subplots sized to the facet count, sharing both axes (for line plots compared
    across panels - value_distribution_plot draws its own grid, since its panels don't share
    a y-axis)."""
    return plt.subplots(n_rows, n_cols, figsize=(cell_w * n_cols, cell_h * n_rows),
                         sharex=True, sharey=True, squeeze=False)


def _plot_algorithm_line(ax, x, y, algo: str, handles: dict):
    """Plot one algorithm's line styled per EVAL_VALUE_STYLE, recording its handle (keyed by
    display label) into `handles` so _algorithm_legend can build one shared legend from it."""
    style = EVAL_VALUE_STYLE[algo]
    label = VALUE_LABELS[algo]
    handle, = ax.plot(x, y, label=label, color=style["color"], linestyle=style["linestyle"],
                       marker="o", markersize=4, linewidth=2)
    handles[label] = handle


def _algorithm_legend(fig, handles: dict, y: float):
    """Add one figure-level legend covering every algorithm plotted via _plot_algorithm_line,
    in EVAL_VALUE_STYLE's fixed order, positioned at height `y` in figure coordinates."""
    labels = [VALUE_LABELS[a] for a in EVAL_VALUE_STYLE if VALUE_LABELS[a] in handles]
    fig.legend([handles[l] for l in labels], labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, y), frameon=False)


def _label_facet_edges(ax, row: int, n_rows: int, col: int, n_cols: int | None = None,
                        title=None, ylabel=None, xlabel=None, right_label=None):
    """Apply title/ylabel/xlabel only at the facet grid's outer edges (title on row 0, ylabel
    on column 0, xlabel on the last row) so inner subplots stay uncluttered. `right_label`, if
    given (together with `n_cols`), is drawn as a row-strip label on the outside of the last
    column, rotated to read bottom-to-top - the conventional spot for a row facet's label."""
    if row == 0 and title is not None:
        ax.set_title(title)
    if col == 0 and ylabel is not None:
        ax.set_ylabel(ylabel)
    if row == n_rows - 1 and xlabel is not None:
        ax.set_xlabel(xlabel)
    if n_cols is not None and col == n_cols - 1 and right_label is not None:
        ax.text(1.05, 0.5, right_label, transform=ax.transAxes, rotation=-90, va="center", ha="left")


# ---- sampling accuracy: exact vs. sampled contribution values ----------------------------

def sampling_accuracy_table(runs: pd.DataFrame, values: pd.DataFrame, log_size: int | None = None) -> pd.DataFrame:
    """
    For every (log, case, log_size, value, analysis) scenario where both an exact run
    (status "ok", sample_size NaN) and a sampled run (status "ok", sample_size not NaN)
    completed, match their per-event values by event id and compute Kendall's tau and
    Spearman's rank correlation between the exact and sampled value vectors. One row per
    matched (scenario, sample_size) pair; scenarios without a completed exact/sampled
    pair are simply absent, since no correlation can be computed for them. `runs`/`values`
    can come from a single eval_naive_and_sampling.py invocation (load_run_dir) or span
    several concatenated ones (load_eval_results); either way, if a (scenario, sample_size)
    combination has more than one completed run (e.g. reruns across several `results_dir/
    <timestamp>/` directories - sampling there is seeded deterministically, so reruns
    reproduce identical values), only the most recent one is kept to avoid duplicate rows.

    If `log_size` is given, only runs with that log_size are considered. The "log" column is
    always omitted from the result, since the eval currently only ever runs against one log
    (see eval_naive_and_sampling.LOGS) - if that changes, reintroduce it as an index level.
    """
    if log_size is not None:
        runs = runs[runs["log_size"] == log_size]

    ok = _ok(runs)
    exact = ok[ok["sample_size"].isna()].drop_duplicates(subset=SAMPLING_SCENARIO_COLUMNS, keep="last")
    sampled = ok[ok["sample_size"].notna()].drop_duplicates(subset=SAMPLING_SCENARIO_COLUMNS + ["sample_size"], keep="last")

    pairs = sampled.merge(
        exact[SAMPLING_SCENARIO_COLUMNS + ["run_id"]],
        on=SAMPLING_SCENARIO_COLUMNS,
        suffixes=("_sampled", "_exact"),
    )

    rows = []
    for _, pair in pairs.iterrows():
        exact_values = values.loc[values["run_id"] == pair["run_id_exact"], ["event", "value"]]
        sampled_values = values.loc[values["run_id"] == pair["run_id_sampled"], ["event", "value"]]
        joined = exact_values.merge(sampled_values, on="event", suffixes=("_exact", "_sampled"))
        if len(joined) < 2:
            continue  # correlation is undefined for fewer than 2 matched events

        # a constant exact or sampled vector (e.g. conformance scenarios where every
        # event gets the same score) makes the correlation undefined - scipy warns and
        # returns NaN in that case, which is the correct, expected result here (rendered
        # as "--" in the LaTeX table), so the warning is suppressed rather than surfaced
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            tau, tau_p = kendalltau(joined["value_exact"], joined["value_sampled"])
            rho, rho_p = spearmanr(joined["value_exact"], joined["value_sampled"])

        row = {col: pair[col] for col in SAMPLING_SCENARIO_COLUMNS}
        row.update({
            "sample_size": int(pair["sample_size"]),
            "n_events": len(joined),
            "kendall_tau": tau,
            "kendall_p": tau_p,
            "spearman_rho": rho,
            "spearman_p": rho_p,
        })
        rows.append(row)

    if not rows:
        raise ValueError("no scenario has both an exact and a sampled run with status 'ok'")

    table = pd.DataFrame(rows)
    table["value"] = pd.Categorical(table["value"], categories=list(VALUE_LABELS), ordered=True)
    table["analysis"] = pd.Categorical(table["analysis"], categories=EVAL_ANALYSIS_ORDER, ordered=True)
    table = table.sort_values(by=SAMPLING_SCENARIO_COLUMNS + ["sample_size"])

    table["value"] = table["value"].map(VALUE_LABELS)
    table["analysis"] = table["analysis"].map(ANALYSIS_LABELS)
    table = table.drop(columns=["log"]).rename(columns={
        "case": "Case", "log_size": "Log size", "value": "Algorithm",
        "analysis": "Analysis", "sample_size": "Sample size", "n_events": "N",
        "kendall_tau": "Kendall's tau", "kendall_p": "Kendall's p",
        "spearman_rho": "Spearman's rho", "spearman_p": "Spearman's p",
    })
    return table.set_index(["Case", "Log size", "Algorithm", "Analysis", "Sample size"])


def sampling_accuracy_summary_table(runs: pd.DataFrame, values: pd.DataFrame, log_size: int | None = None) -> pd.DataFrame:
    """
    Reduce sampling_accuracy_table down to the extremes: for each of Kendall's tau and
    Spearman's rho, the minimum and maximum value reached across every matched (scenario,
    sample_size) pair, each paired with the p-value of that specific correlation test.
    Scenarios where the correlation is undefined (a constant exact-value vector - see
    sampling_accuracy_table) are excluded from the min/max, same as they'd be blank ("--")
    in the detailed table.
    """
    detail = sampling_accuracy_table(runs, values, log_size=log_size)

    rows = []
    for metric, value_col, p_col in [
        ("Kendall's tau", "Kendall's tau", "Kendall's p"),
        ("Spearman's rho", "Spearman's rho", "Spearman's p"),
    ]:
        defined = detail[detail[value_col].notna()]
        if defined.empty:
            rows.append({"Metric": metric, "Min": float("nan"), "Min p": float("nan"),
                         "Max": float("nan"), "Max p": float("nan")})
            continue
        min_row = defined.loc[defined[value_col].idxmin()]
        max_row = defined.loc[defined[value_col].idxmax()]
        rows.append({
            "Metric": metric,
            "Min": min_row[value_col], "Min p": min_row[p_col],
            "Max": max_row[value_col], "Max p": max_row[p_col],
        })

    return pd.DataFrame(rows).set_index("Metric")


def sampling_accuracy_latex(
    run_dir: str,
    log_size: int | None = 2,
    output_path: str | None = None,
    caption: str = "Minimum and maximum Kendall's tau / Spearman's rank correlation reached "
                    "between exact and sampled contribution values, with the p-value of that "
                    "correlation test, across every matched scenario and sample size.",
    label: str = "tab:sampling-accuracy",
) -> str:
    """
    Render the sampling-accuracy summary (see sampling_accuracy_summary_table) as a LaTeX
    table for a single eval_naive_and_sampling.py run, e.g. `run_dir='data/results/
    20260918_015755'`, optionally writing it to `output_path`. Restricted to `log_size`
    (default 2) since larger log sizes rarely have a completed exact run to compare against.
    """
    runs, values = load_run_dir(run_dir)
    table = sampling_accuracy_summary_table(runs, values, log_size=log_size).copy()

    # a plain "%.3f" would silently round a genuinely tiny p-value (e.g. 6e-5) down to
    # "0.000", misrepresenting it as exactly zero - report those as "<0.001" instead, the
    # conventional way to note a small p-value without implying more precision than warranted
    for p_col in ["Min p", "Max p"]:
        table[p_col] = table[p_col].map(lambda p: "--" if pd.isna(p) else "<0.001" if p < 0.001 else f"{p:.3f}")

    latex = table.to_latex(
        float_format="%.3f",
        na_rep="--",
        escape=True,
        column_format="l" + "r" * len(table.columns),
        caption=caption,
        label=label,
        position="ht",
    )

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)

    return latex

# ---- sampling runtime: how long each sampled approach takes -------------------------------

def sampling_runtime_plot(
    run_dir: str,
    case: str | None = None,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Plot runtime (seconds, log scale) of each sampling-based value-computation approach
    against log size, for a single eval_naive_and_sampling.py run. One subplot column per
    analysis, one subplot row per sample size; within each subplot, one line per algorithm,
    colored/styled per EVAL_VALUE_STYLE. Only successfully completed sampled runs (status
    "ok", sample_size not null) are included - a timed-out/errored run has no runtime to plot.

    `case` selects one case (e.g. "Application") when the run covers more than one; if the
    run covers exactly one case, it's picked automatically.
    """
    runs, _ = load_run_dir(run_dir)
    sampled_ok = _ok(runs)
    sampled_ok = sampled_ok[sampled_ok["sample_size"].notna()]

    case = _pick_one(sampled_ok["case"].unique(), case, "case", context=f" in {run_dir!r}")
    sampled_ok = sampled_ok[sampled_ok["case"] == case]

    #sample_sizes = sorted(sampled_ok["sample_size"].unique())
    sample_sizes = [1000,5000]
    analyses = _ordered(EVAL_ANALYSIS_ORDER, sampled_ok["analysis"].unique())
    log_sizes = sorted(sampled_ok["log_size"].unique())

    fig, axes = _facet_grid(len(sample_sizes), len(analyses))

    handles = {}
    for row, sample_size in enumerate(sample_sizes):
        for col, analysis in enumerate(analyses):
            ax = axes[row][col]
            subset = sampled_ok[(sampled_ok["sample_size"] == sample_size) & (sampled_ok["analysis"] == analysis)]
            for algo in EVAL_VALUE_STYLE:
                # mean() collapses any reruns of the same log_size into one point
                line = subset[subset["value"] == algo].groupby("log_size")["runtime"].mean().sort_index()
                if not line.empty:
                    _plot_algorithm_line(ax, line.index, line.values, algo, handles)

            _label_facet_edges(ax, row, len(sample_sizes), col, n_cols=len(analyses),
                                title=ANALYSIS_LABELS[analysis], ylabel="Runtime (s)", xlabel="Log size",
                                right_label=f"s: {int(sample_size)}")
            ax.set_yscale("log")
            ax.set_xticks(log_sizes)
            ax.grid(False)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    #fig.suptitle(f"Case: {case}", y=0.99)
    _algorithm_legend(fig, handles, y=0.94)

    if output_path:
        fig.savefig(os.path.join(output_path, f"runtime_{case}.pdf"),
                    format="pdf", bbox_inches="tight")

    return fig


# ---- value distribution: spread of per-event values for one scenario ----------------------

def _draw_boxes(ax, box_data, positions, colors):
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                     showfliers=False, medianprops={"color": "black"})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)


def _draw_violins(ax, box_data, positions, colors):
    for pos, event_values, color in zip(positions, box_data, colors):
        if event_values.nunique() < 2:
            # gaussian_kde needs variance to estimate a density from - a constant vector (or
            # a single event) has none, so just mark the one value instead of raising
            ax.plot([pos - 0.2, pos + 0.2], [event_values.iloc[0]] * 2, color="black", linewidth=1.5, zorder=2)
            continue
        vp = ax.violinplot([event_values], positions=[pos], widths=0.5, showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.35)
            body.set_edgecolor(color)
        vp["cmedians"].set_color("black")


def _overlay_points(ax, box_data, positions, colors, rng):
    for pos, event_values, color in zip(positions, box_data, colors):
        jitter = rng.uniform(-0.12, 0.12, size=len(event_values))
        ax.scatter(pos + jitter, event_values, color=color, s=14, zorder=3,
                   edgecolor="white", linewidth=0.4)


def value_distribution_plot(
    run_dir: str,
    log_size: int,
    case: str | None = None,
    sample_size: int | None = None,
    analysis: str | None = None,
    kind: str = "box",
    output_path: str | None = None,
) -> plt.Figure:
    """
    Show the distribution of per-event contribution values for one (case, log_size,
    sample_size) scenario of a single eval_naive_and_sampling.py run: one subplot per
    analysis, with one box/violin (raw per-event values overlaid as jittered points, since a
    scenario usually has only a handful of events) per algorithm that completed successfully
    (status "ok") for that scenario. `sample_size=None` (default) selects the exact ("naive")
    run; pass a sample size (e.g. 1000) to look at a sampled run instead. `case` is auto-picked
    if the run has exactly one for the given `log_size`/`sample_size`.

    `analysis` restricts the plot to a single analysis (one of EVAL_ANALYSIS_ORDER, e.g.
    "handover") instead of a subplot per analysis - handy since handover/efg/conformance
    values live on different scales and are usually inspected one at a time.

    `kind` is "box" (default) or "violin" - a vertical, mirrored density estimate that shows
    the distribution's shape rather than just its quartiles, at the cost of looking more
    confident than a handful of events really supports (hence the point overlay either way).
    A constant per-event vector has no density to estimate, so it falls back to a flat marker.
    """
    if kind not in ("box", "violin"):
        raise ValueError(f"kind must be 'box' or 'violin', got {kind!r}")

    runs, values = load_run_dir(run_dir)
    ok = _ok(runs[runs["log_size"] == log_size])
    ok = ok[ok["sample_size"].isna()] if sample_size is None else ok[ok["sample_size"] == sample_size]
    if ok.empty:
        raise ValueError(f"no completed run for log_size={log_size}, sample_size={sample_size}")

    case = _pick_one(ok["case"].unique(), case, "case", context=f" for log_size={log_size}, sample_size={sample_size}")
    ok = ok[ok["case"] == case]

    analyses = _ordered(EVAL_ANALYSIS_ORDER, ok["analysis"].unique())
    analysis = _pick_one(analyses, analysis, "analysis", required=False,
                          context=f" for case={case!r}, log_size={log_size}, sample_size={sample_size}")
    if analysis is not None:
        analyses = [analysis]
    algorithms = _ordered(EVAL_VALUE_STYLE, ok["value"].unique())

    fig, axes = plt.subplots(1, len(analyses), figsize=(3.0 * len(analyses), 3), squeeze=False)
    rng = np.random.default_rng(0)  # fixed seed: jitter should look the same across redraws

    for col, analysis_name in enumerate(analyses):
        ax = axes[0][col]
        box_data, positions, colors = [], [], []
        for i, algo in enumerate(algorithms):
            run = ok[(ok["analysis"] == analysis_name) & (ok["value"] == algo)]
            if run.empty:
                continue
            event_values = values.loc[values["run_id"] == run.iloc[0]["run_id"], "value"]
            box_data.append(event_values)
            positions.append(i)
            colors.append(EVAL_VALUE_STYLE[algo]["color"])

        (_draw_boxes if kind == "box" else _draw_violins)(ax, box_data, positions, colors)
        _overlay_points(ax, box_data, positions, colors, rng)

        #ax.set_title(ANALYSIS_LABELS[analysis_name])
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels([VALUE_LABELS[a] for a in algorithms], rotation=20, ha="right")
        if col == 0:
            ax.set_ylabel("Contribution value")
        ax.grid(True, axis="y", alpha=0.3)

    sample_label = "exact" if sample_size is None else f"sample size {int(sample_size)}"
    #fig.suptitle(f"Case: {case}, log size: {log_size} ({sample_label})")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if output_path:
        fig.savefig(os.path.join(output_path,f"distribution_{case}_{log_size}_{sample_label}_{analysis}.pdf"), format="pdf", bbox_inches="tight")

    return fig


if __name__ == "__main__":
    print(pipeline_timing_latex(results_dir=os.path.join(RESULTS_DIR, "new_lineage_results")))
    print(value_computation_timing_latex(results_dir=os.path.join(RESULTS_DIR, "new_lineage_results")))
    print(combined_timing_latex(pipeline_results_dir=os.path.join(RESULTS_DIR, "new_lineage_results"),value_results_dir=os.path.join(RESULTS_DIR, "new_lineage_results")))
    #print(sampling_accuracy_latex(os.path.join(RESULTS_DIR, "20260917_011630"), log_size=2))

    #sampling_runtime_plot(os.path.join(RESULTS_DIR, "20260918_015755"), "Application", output_path=os.path.join(RESULTS_DIR,"plots"))
    #value_distribution_plot(os.path.join(RESULTS_DIR, "20260918_015755"), 2, "Application",
    #                         kind="violin", analysis="handover", output_path=os.path.join(RESULTS_DIR,"plots"))
    #plt.show()
