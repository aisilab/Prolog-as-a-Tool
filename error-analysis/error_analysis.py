"""
Error analysis for Prolog-as-a-tool multi-try evaluation logs.

Parses log files, classifies per-attempt errors and per-sample outcomes,
then prints summary tables and dumps a CSV for further inspection.

Usage:
    python error_analysis.py
"""

import re
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AttemptResult:
    attempt_num: int
    status: str          # "success" | "no_output" | "no_code" | "timeout" | "silent_fail"
    raw_output: str      # the value returned by Prolog, or empty string
    error_type: str      # Level-1 category (empty for success)


@dataclass
class SampleResult:
    log_name: str
    sample_num: int
    problem: str
    gold_value: str
    final_prolog_output: str          # "None" if all attempts failed
    total_attempts: int
    success_attempt: Optional[int]    # None if N/A
    strict: bool
    arithmetic: bool
    structure: bool
    full: bool
    semantic_score: float
    model_output: str
    attempts: list[AttemptResult] = field(default_factory=list)

    # Level-2 / Level-3 outcome (filled by classify_sample_outcome)
    outcome: str = ""
    wrong_answer_subtype: str = ""


# ---------------------------------------------------------------------------
# Level-1: per-attempt error classification
# ---------------------------------------------------------------------------

# Patterns for the raw Prolog output (content inside single quotes)
_RE_UNINSTANTIATED = re.compile(r"^_\d+$")
_RE_RATIONAL       = re.compile(r"^-?\d+r\d+$")
_RE_ARITH_EXPR     = re.compile(r"[+\-*/]")          # contains arithmetic op
_RE_VAR_MIX        = re.compile(r"_\d+")             # contains uninstantiated var
_RE_ALPHA_ATOM     = re.compile(r"^[a-z_][a-z_A-Z0-9]*$")  # lowercase atom
_RE_COMPOUND       = re.compile(r"\(.*\)")            # Prolog compound term


def classify_raw_output(raw: str) -> str:
    """Classify the raw Prolog output string into a Level-1 error type."""
    r = raw.strip()
    if _RE_UNINSTANTIATED.match(r):
        return "uninstantiated_var"
    if _RE_RATIONAL.match(r):
        return "rational_number"
    # Mix of number and unbound var: e.g. "20+_26612+14"
    if _RE_VAR_MIX.search(r):
        return "unevaluated_expr"
    if _RE_ARITH_EXPR.search(r):
        return "unevaluated_expr"
    if _RE_COMPOUND.search(r):
        return "compound_term"
    if _RE_ALPHA_ATOM.match(r):
        return "atom_instead_of_number"
    return "other_no_output"


# Regexes for attempt lines
_RE_ATTEMPT_SUCCESS  = re.compile(r"Attempt (\d+): Successful numeric output: (.+)")
_RE_ATTEMPT_NO_OUT   = re.compile(r"Attempt (\d+): Prolog code did not yield a numeric result \('(.*)'\)")
_RE_ATTEMPT_NO_CODE  = re.compile(r"Attempt (\d+): No Prolog code extracted\.")
_RE_TIMEOUT          = re.compile(r"Error executing Prolog code:.*timed out")
_RE_PROCESSED        = re.compile(r"Processed prompts:")


def parse_attempts(pre_summary_text: str) -> list[AttemptResult]:
    """
    Extract all attempt results from the text that precedes a sample summary block.

    The tricky cases:
    - Timeouts: appear as "Error executing Prolog code: ... timed out" with no Attempt N: prefix.
      We count these as attempts by tracking how many Processed prompts lines appear vs
      how many explicit Attempt lines there are.
    - Silent failures in struct log: "Processed prompts:" lines with no Attempt or Error line.
      These are also "no code extracted" failures.
    """
    results = []
    lines = pre_summary_text.splitlines()

    # We'll do two passes:
    # Pass 1: collect all explicit attempt lines (success, no_output, no_code, timeout)
    explicit_attempts: dict[int, AttemptResult] = {}
    timeout_count = 0
    processed_count = 0

    for line in lines:
        if _RE_PROCESSED.search(line):
            processed_count += 1
            continue

        m = _RE_ATTEMPT_SUCCESS.search(line)
        if m:
            n = int(m.group(1))
            explicit_attempts[n] = AttemptResult(
                attempt_num=n, status="success",
                raw_output=m.group(2).strip(), error_type=""
            )
            continue

        m = _RE_ATTEMPT_NO_OUT.search(line)
        if m:
            n = int(m.group(1))
            raw = m.group(2).strip()
            explicit_attempts[n] = AttemptResult(
                attempt_num=n, status="no_output",
                raw_output=raw,
                error_type=classify_raw_output(raw)
            )
            continue

        m = _RE_ATTEMPT_NO_CODE.search(line)
        if m:
            n = int(m.group(1))
            explicit_attempts[n] = AttemptResult(
                attempt_num=n, status="no_code",
                raw_output="", error_type="no_code_extracted"
            )
            continue

        if _RE_TIMEOUT.search(line):
            timeout_count += 1
            continue

    # Build ordered attempt list. Explicit attempts go in by number.
    # Timeouts: they consumed a "Processed prompts" slot; insert them as synthetic entries.
    # Silent failures (struct log): processed_count > explicit_attempts = silent no-code fails.

    # Determine implicit attempts: those not covered by explicit Attempt N: lines
    # Each "Processed prompts" corresponds to one generation → one attempt.
    n_explicit = len(explicit_attempts)
    n_implicit = processed_count - n_explicit  # attempts without any explicit line

    # First, add all explicit attempts in order
    results = [explicit_attempts[k] for k in sorted(explicit_attempts)]

    # Add timeouts (we know how many there were; assign pseudo-attempt numbers)
    # We assign them after explicit ones for simplicity, since we can't always
    # determine the exact slot from the log text.
    for _ in range(timeout_count):
        results.append(AttemptResult(
            attempt_num=-1, status="timeout",
            raw_output="", error_type="timeout"
        ))

    # Add remaining silent failures (no attempt line, no timeout, no explicit result)
    silent_count = n_implicit - timeout_count
    if silent_count > 0:
        for _ in range(silent_count):
            results.append(AttemptResult(
                attempt_num=-1, status="silent_fail",
                raw_output="", error_type="no_code_extracted"
            ))

    return results


# ---------------------------------------------------------------------------
# Level-2 / Level-3: per-sample outcome classification
# ---------------------------------------------------------------------------

def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def classify_wrong_answer(final_output: str, gold: str) -> str:
    """Sub-classify wrong-answer samples (Level 3)."""
    got = _safe_float(final_output)
    exp = _safe_float(gold)
    if got is None or exp is None or exp == 0:
        return "other"

    ratio = got / exp
    # Off-by-factor (within 1% tolerance)
    for factor in [2, 3, 4, 5, 10, 0.5, 1/3, 0.25, 0.1]:
        if abs(ratio - factor) / max(abs(factor), 1e-9) < 0.01:
            if factor > 1:
                return f"off_by_factor_{factor:.0f}x"
            else:
                return f"off_by_factor_1/{round(1/factor)}x"

    # Partial computation heuristic: answer is a divisor or factor of gold
    if exp != 0 and abs(got) < abs(exp) and abs(exp % got) < 0.01 if got != 0 else False:
        return "partial_computation"

    return "other"


def classify_sample_outcome(sample: SampleResult) -> None:
    """Fill sample.outcome and sample.wrong_answer_subtype in place."""
    if sample.final_prolog_output == "None":
        sample.outcome = "never_executed"
        return

    # Correct: strict match suffices
    if sample.strict or sample.full:
        sample.outcome = "correct"
        return

    # Got a number but it was wrong
    sample.outcome = "wrong_answer"
    sample.wrong_answer_subtype = classify_wrong_answer(
        sample.final_prolog_output, sample.gold_value
    )


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

_RE_SAMPLE_HEADER  = re.compile(r"--- Sample (\d+) Summary ---")
_RE_RESULT_LINE    = re.compile(r"Result achieved in attempt: (.+?) / (\d+)")
_RE_FINAL_OUTPUT   = re.compile(r"Final Prolog Output: (.+)")
_RE_GOLD           = re.compile(r"Gold Value: (.+)")
_RE_FLAGS          = re.compile(
    r"Strict: (True|False) \| Arithmetic: (True|False) \| Structure: (True|False) \| Full: (True|False)"
)
_RE_SEMANTIC       = re.compile(r"Semantic Score: ([\d.]+)%")
_RE_MODEL_OUTPUT   = re.compile(r"Successful Model Output:")
_RE_SEPARATOR      = re.compile(r"^-{40,}$", re.MULTILINE)


def parse_log(filepath: Path) -> list[SampleResult]:
    log_name = filepath.stem
    text = filepath.read_text(encoding="utf-8", errors="replace")

    # Split into per-sample chunks. Each chunk = everything from after the
    # previous summary block up to and including the next summary block.
    # Strategy: find all "--- Sample N Summary ---" positions and split there.

    sample_headers = list(_RE_SAMPLE_HEADER.finditer(text))
    if not sample_headers:
        print(f"WARNING: no sample headers found in {filepath.name}", file=sys.stderr)
        return []

    results = []
    for i, hdr in enumerate(sample_headers):
        sample_num = int(hdr.group(1))

        # Text before this summary header = the attempt lines for this sample
        block_start = sample_headers[i - 1].end() if i > 0 else 0
        pre_summary = text[block_start: hdr.start()]

        # Text from the header to either the next header start or end of file
        block_end = sample_headers[i + 1].start() if i + 1 < len(sample_headers) else len(text)
        summary_block = text[hdr.start(): block_end]

        # --- Parse summary block ---
        m_result = _RE_RESULT_LINE.search(summary_block)
        if not m_result:
            continue

        success_str = m_result.group(1).strip()
        total_attempts = int(m_result.group(2))
        success_attempt = None if success_str == "N/A" else int(success_str)

        m_final = _RE_FINAL_OUTPUT.search(summary_block)
        final_output = m_final.group(1).strip() if m_final else "None"

        m_gold = _RE_GOLD.search(summary_block)
        gold_value = m_gold.group(1).strip() if m_gold else ""

        m_flags = _RE_FLAGS.search(summary_block)
        strict = arithmetic = structure = full = False
        if m_flags:
            strict     = m_flags.group(1) == "True"
            arithmetic = m_flags.group(2) == "True"
            structure  = m_flags.group(3) == "True"
            full       = m_flags.group(4) == "True"

        m_sem = _RE_SEMANTIC.search(summary_block)
        semantic_score = float(m_sem.group(1)) if m_sem else 0.0

        # Model output: between "Successful Model Output:" and the next "---..." separator
        model_output = ""
        m_mo = _RE_MODEL_OUTPUT.search(summary_block)
        if m_mo:
            rest = summary_block[m_mo.end():]
            m_sep = _RE_SEPARATOR.search(rest)
            model_output = rest[: m_sep.start()].strip() if m_sep else rest.strip()

        # Problem text: last (USER) line in the pre-summary text
        problem = ""
        user_lines = re.findall(
            r"\(USER\) Please generate a piece of Prolog code to solve the given math problem\.\s*\n(.+?)(?=\nProcessed|\Z)",
            pre_summary, re.DOTALL
        )
        if user_lines:
            problem = user_lines[-1].strip()

        # --- Parse attempts from pre-summary text ---
        attempts = parse_attempts(pre_summary)

        sample = SampleResult(
            log_name=log_name,
            sample_num=sample_num,
            problem=problem,
            gold_value=gold_value,
            final_prolog_output=final_output,
            total_attempts=total_attempts,
            success_attempt=success_attempt,
            strict=strict,
            arithmetic=arithmetic,
            structure=structure,
            full=full,
            semantic_score=semantic_score,
            model_output=model_output,
            attempts=attempts,
        )
        classify_sample_outcome(sample)
        results.append(sample)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  0.0%"
    return f"{100 * n / total:5.1f}%"


def report_log(results: list[SampleResult], log_name: str) -> None:
    n = len(results)
    print(f"\n{'='*70}")
    print(f"  {log_name}  ({n} samples)")
    print(f"{'='*70}")

    # --- Sample-level outcome ---
    outcome_counts = Counter(r.outcome for r in results)
    print("\n[Sample outcomes]")
    for cat in ["correct", "wrong_answer", "never_executed"]:
        c = outcome_counts.get(cat, 0)
        print(f"  {cat:<20}  {c:4d}  ({_pct(c, n)})")

    # Wrong answer sub-types
    wrong = [r for r in results if r.outcome == "wrong_answer"]
    if wrong:
        sub = Counter(r.wrong_answer_subtype for r in wrong)
        print(f"\n  Wrong-answer breakdown ({len(wrong)} samples):")
        for cat, c in sub.most_common():
            print(f"    {cat:<30}  {c:4d}  ({_pct(c, len(wrong))})")

    # --- Attempt-level error distribution ---
    all_failed_attempts = [
        a for r in results for a in r.attempts
        if a.status != "success"
    ]
    n_failed_attempts = len(all_failed_attempts)
    n_total_attempts  = sum(r.total_attempts for r in results)
    n_success_attempts = n_total_attempts - n_failed_attempts

    print(f"\n[Attempt-level stats]")
    print(f"  Total attempts:          {n_total_attempts:5d}")
    print(f"  Successful attempts:     {n_success_attempts:5d}  ({_pct(n_success_attempts, n_total_attempts)})")
    print(f"  Failed attempts:         {n_failed_attempts:5d}  ({_pct(n_failed_attempts, n_total_attempts)})")

    if all_failed_attempts:
        err_counts = Counter(a.error_type for a in all_failed_attempts)
        print(f"\n  Failed-attempt error types:")
        for cat, c in err_counts.most_common():
            print(f"    {cat:<25}  {c:4d}  ({_pct(c, n_failed_attempts)})")

    # --- Retry statistics (among samples that eventually succeeded) ---
    succeeded = [r for r in results if r.success_attempt is not None]
    if succeeded:
        # After the None-check above, success_attempt is int for all items in succeeded
        attempt_nums: list[int] = [r.success_attempt for r in succeeded]  # type: ignore[misc]
        retry_needed = [n for n in attempt_nums if n > 1]
        print(f"\n[Retry statistics]")
        print(f"  Samples that eventually succeeded:  {len(succeeded):4d}")
        print(f"  Needed >1 attempt:                  {len(retry_needed):4d}  ({_pct(len(retry_needed), len(succeeded))})")
        print(f"  Avg attempts to success:            {sum(attempt_nums)/len(attempt_nums):.2f}")
        print(f"  Max attempts to success:            {max(attempt_nums)}")
        dist: Counter[int] = Counter(attempt_nums)
        print(f"  Distribution of success attempt:")
        for k in sorted(dist):
            print(f"    attempt {k:2d}: {dist[k]:4d}  ({_pct(dist[k], len(succeeded))})")


def comparative_table(all_results: dict[str, list[SampleResult]]) -> None:
    print(f"\n{'='*70}")
    print("  CROSS-RUN COMPARISON")
    print(f"{'='*70}")

    header = f"{'Metric':<35}" + "".join(f"{name[:18]:>20}" for name in all_results)
    print(header)
    print("-" * len(header))

    def row(label: str, values: list[str]) -> None:
        print(f"{label:<35}" + "".join(f"{v:>20}" for v in values))

    logs = list(all_results.keys())
    results_list = [all_results[k] for k in logs]

    def pct_outcome(rs, cat):
        n = len(rs)
        c = sum(1 for r in rs if r.outcome == cat)
        return f"{100*c/n:.1f}%" if n else "-"

    def avg_attempts(rs):
        succeeded = [r for r in rs if r.success_attempt is not None]
        if not succeeded:
            return "-"
        return f"{sum(r.success_attempt for r in succeeded)/len(succeeded):.2f}"

    def pct_error(rs, etype):
        failed = [a for r in rs for a in r.attempts if a.status != "success"]
        n = len(failed)
        c = sum(1 for a in failed if a.error_type == etype)
        return f"{100*c/n:.1f}%" if n else "-"

    row("Correct (%)",               [pct_outcome(r, "correct")       for r in results_list])
    row("Wrong answer (%)",          [pct_outcome(r, "wrong_answer")   for r in results_list])
    row("Never executed (%)",        [pct_outcome(r, "never_executed") for r in results_list])
    row("Avg attempts to success",   [avg_attempts(r)                  for r in results_list])
    print()
    row("Failed-attempt errors:",    [""] * len(logs))
    for etype in ["uninstantiated_var", "rational_number", "unevaluated_expr",
                  "atom_instead_of_number", "compound_term", "timeout",
                  "no_code_extracted", "other_no_output"]:
        row(f"  {etype}",            [pct_error(r, etype)              for r in results_list])


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(all_results: dict[str, list[SampleResult]], out_path: Path) -> None:
    fieldnames = [
        "log_name", "sample_num", "problem", "gold_value", "final_prolog_output",
        "total_attempts", "success_attempt", "strict", "arithmetic", "structure", "full",
        "semantic_score", "outcome", "wrong_answer_subtype",
        "n_uninstantiated_var", "n_rational_number", "n_unevaluated_expr",
        "n_atom_instead_of_number", "n_compound_term", "n_timeout",
        "n_no_code_extracted", "n_other_no_output",
        "model_output",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for results in all_results.values():
            for r in results:
                err_counts = Counter(a.error_type for a in r.attempts if a.status != "success")
                writer.writerow({
                    "log_name": r.log_name,
                    "sample_num": r.sample_num,
                    "problem": r.problem,
                    "gold_value": r.gold_value,
                    "final_prolog_output": r.final_prolog_output,
                    "total_attempts": r.total_attempts,
                    "success_attempt": r.success_attempt if r.success_attempt is not None else "",
                    "strict": r.strict,
                    "arithmetic": r.arithmetic,
                    "structure": r.structure,
                    "full": r.full,
                    "semantic_score": r.semantic_score,
                    "outcome": r.outcome,
                    "wrong_answer_subtype": r.wrong_answer_subtype,
                    "n_uninstantiated_var":     err_counts.get("uninstantiated_var", 0),
                    "n_rational_number":        err_counts.get("rational_number", 0),
                    "n_unevaluated_expr":       err_counts.get("unevaluated_expr", 0),
                    "n_atom_instead_of_number": err_counts.get("atom_instead_of_number", 0),
                    "n_compound_term":          err_counts.get("compound_term", 0),
                    "n_timeout":                err_counts.get("timeout", 0),
                    "n_no_code_extracted":      err_counts.get("no_code_extracted", 0),
                    "n_other_no_output":        err_counts.get("other_no_output", 0),
                    "model_output": r.model_output,
                })
    print(f"\nCSV written to: {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Human-readable labels for each run
RUN_LABELS = {
    "sp-struct-rwd1-multipletry": "Struct rwd1",
    "sp-declare-rwd1-multitry":   "Declare rwd1",
    "sp-declare-rwd3-multipletry": "Declare rwd3",
}

# Error-type display names (order = legend order, bottom to top in stacked bar)
ERROR_TYPES_DISPLAY = [
    ("no_code_extracted",      "No code extracted"),
    ("uninstantiated_var",     "Uninstantiated variable"),
    ("rational_number",        "Rational number"),
    ("unevaluated_expr",       "Unevaluated expression"),
    ("atom_instead_of_number", "Atom instead of number"),
    ("timeout",                "Timeout"),
    ("compound_term",          "Compound term"),
    ("other_no_output",        "Other"),
]


def plot_figure(all_results: dict[str, list[SampleResult]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # -- Collect data --
    run_keys = list(all_results.keys())
    run_labels = [RUN_LABELS.get(k, k) for k in run_keys]
    n_runs = len(run_keys)

    # Panel (a): sample-level outcomes (percentages)
    outcome_cats = ["correct", "wrong_answer", "never_executed"]
    outcome_labels = ["Correct", "Wrong answer", "Never executed"]
    outcome_colors = ["#4daf4a", "#ff7f00", "#e41a1c"]  # green, orange, red

    outcome_pcts: dict[str, list[float]] = {cat: [] for cat in outcome_cats}
    for key in run_keys:
        rs = all_results[key]
        n = len(rs)
        for cat in outcome_cats:
            outcome_pcts[cat].append(100 * sum(1 for r in rs if r.outcome == cat) / n)

    # Panel (b): per-attempt error types (absolute counts)
    error_counts: dict[str, list[int]] = {et: [] for et, _ in ERROR_TYPES_DISPLAY}
    for key in run_keys:
        rs = all_results[key]
        failed = [a for r in rs for a in r.attempts if a.status != "success"]
        c = Counter(a.error_type for a in failed)
        for et, _ in ERROR_TYPES_DISPLAY:
            error_counts[et].append(c.get(et, 0))

    # -- Layout --
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.5, 5.2),
                                      gridspec_kw={"height_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.18, top=0.95, hspace=0.55)

    # ---- Panel (a): stacked horizontal bars ----
    y_pos = np.arange(n_runs)
    bar_h = 0.55
    left = np.zeros(n_runs)
    for cat, label, color in zip(outcome_cats, outcome_labels, outcome_colors):
        vals = np.array(outcome_pcts[cat])
        ax_a.barh(y_pos, vals, height=bar_h, left=left, label=label, color=color,
                  edgecolor="white", linewidth=0.5)
        # Percentage labels right-aligned inside each bar segment (if wide enough)
        for j, v in enumerate(vals):
            if v >= 5:
                ax_a.text(left[j] + v - 1.5, y_pos[j], f"{v:.0f}%",
                          ha="right", va="center", fontsize=7,
                          color="white" if cat != "wrong_answer" else "black",
                          fontweight="bold")
        left += vals

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(run_labels, fontsize=8)
    ax_a.set_xlabel("Samples (%)", fontsize=8)
    ax_a.set_xlim(0, 100)
    ax_a.invert_yaxis()
    ax_a.legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    ax_a.set_title("(a) Sample outcomes", fontsize=9, fontweight="bold", pad=8)
    ax_a.tick_params(axis="x", labelsize=7)

    # ---- Panel (b): stacked vertical bars for error types ----
    x_pos = np.arange(n_runs)
    bar_w = 0.5
    bottom = np.zeros(n_runs)

    # Use a qualitative colormap
    cmap_colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
                   "#fb9a99", "#e31a1c", "#fdbf6f", "#cab2d6"]

    for idx, (et, elabel) in enumerate(ERROR_TYPES_DISPLAY):
        vals = np.array(error_counts[et])
        if vals.sum() == 0:
            continue
        ax_b.bar(x_pos, vals, width=bar_w, bottom=bottom, label=elabel,
                 color=cmap_colors[idx % len(cmap_colors)],
                 edgecolor="white", linewidth=0.4)
        bottom += vals

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(run_labels, fontsize=8, rotation=15, ha="right")
    ax_b.set_ylabel("Failed attempts (count)", fontsize=8)
    ax_b.legend(loc="upper left", fontsize=5.5, framealpha=0.9, ncol=1,
                borderpad=0.3, labelspacing=0.3, handlelength=1.2, handletextpad=0.4)
    ax_b.set_title("(b) Attempt-level error types", fontsize=9, fontweight="bold", pad=8)
    ax_b.tick_params(axis="both", labelsize=7)

    # -- Save --
    for fmt in ["pdf", "png"]:
        out_path = out_dir / f"error_analysis.{fmt}"
        fig.savefig(out_path, dpi=300)
        print(f"Figure saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent

LOG_FILES = [
    "sp-struct-rwd1-multipletry.log",
    "sp-declare-rwd1-multitry.log",
    "sp-declare-rwd3-multipletry.log",
]


def main() -> None:
    all_results: dict[str, list[SampleResult]] = {}

    for fname in LOG_FILES:
        fpath = LOG_DIR / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping.", file=sys.stderr)
            continue
        print(f"Parsing {fname}...", end=" ", flush=True)
        results = parse_log(fpath)
        print(f"{len(results)} samples parsed.")
        all_results[fpath.stem] = results

    # Sanity check
    for name, results in all_results.items():
        n = len(results)
        if n != 375:
            print(f"  WARNING: {name} has {n} samples, expected 375", file=sys.stderr)
        n_correct = sum(1 for r in results if r.strict)
        pct = 100 * n_correct / n if n else 0
        print(f"  {name}: Prolog accuracy (strict) = {pct:.2f}%  (from log summary = check above)")

    # Per-log reports
    for name, results in all_results.items():
        report_log(results, name)

    # Cross-run comparison
    if len(all_results) > 1:
        comparative_table(all_results)

    # Plot
    plot_figure(all_results, LOG_DIR)

    # CSV export
    csv_path = LOG_DIR / "error_analysis_results.csv"
    export_csv(all_results, csv_path)


if __name__ == "__main__":
    main()
