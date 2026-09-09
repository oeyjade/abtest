"""
analysis.py — full A/B test protocol on the Cookie Cats mobile game experiment.

EXPERIMENT
  Cookie Cats gates player progression behind a forced wait. The experiment
  moved that gate from level 30 (control) to level 40 (treatment) to test
  whether letting players go deeper before the first wall improves retention.

  control   = gate_30   treatment = gate_40
  Effects are reported as treatment minus control throughout.

PRE-REGISTERED DECISIONS (fixed before looking at outcomes)
  Primary metric      day-7 retention
  Guardrail metrics   day-1 retention, total game rounds played
  Hypotheses          H0: retention is equal in both arms
                      H1: retention differs (two-sided)
  Alpha               0.05
  Practical threshold 1.0 percentage point on day-7 retention.
                      Below this, moving the gate is not worth the design and
                      QA cost of changing the progression curve.

  Day-7 is primary rather than day-1 because a progression gate at level 30 vs
  40 cannot plausibly bite within the first 24 hours for most players; day-7 is
  where the mechanism should show up. Choosing this in advance is what stops
  the analysis from becoming a search for whichever metric happens to move.

DATA
  Download 'cookie_cats.csv' from Kaggle (Mobile Games A/B Testing) into
  ./data/. Without it the script runs on a clearly labelled synthetic
  stand-in so the pipeline is testable end to end. Synthetic output is
  watermarked in every artefact and must never be presented as a finding.

Run:  python analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import abtest

# --------------------------------------------------------------------------
# Configuration — everything pre-registered lives here, at the top, in one place
# --------------------------------------------------------------------------

DATA_PATH = Path("data/cookie_cats.csv")
OUT_DIR = Path("output")

CONTROL, TREATMENT = "gate_30", "gate_40"
PRIMARY_METRIC = "retention_7"
GUARDRAILS = ["retention_1"]
ALPHA = 0.05
PRACTICAL_THRESHOLD = 0.010  # 1.0 percentage point, absolute
EXPECTED_SPLIT = 0.50

# Synthetic fallback only. A small negative effect is injected so the fallback
# exercises the full decision path. Note the day-1/day-7 coupling below shifts
# the realised day-7 rate away from the nominal baseline, so the recovered
# effect is diluted rather than an exact match — the clean recovery test lives
# in validate.py, which is where correctness is actually established.
SYNTH_N = 90_000
SYNTH_BASELINE_D1, SYNTH_BASELINE_D7 = 0.448, 0.190
SYNTH_TRUE_EFFECT_D1, SYNTH_TRUE_EFFECT_D7 = -0.006, -0.008


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, bool]:
    """Return (dataframe, is_real). Falls back to labelled synthetic data."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded real data: {DATA_PATH}  ({len(df):,} rows)")
        return df, True

    print("!" * 72)
    print("cookie_cats.csv not found — generating SYNTHETIC stand-in data.")
    print("Results below are NOT findings. Drop the real CSV in ./data/ and rerun.")
    print("!" * 72)

    rng = np.random.default_rng(2024)
    version = rng.choice([CONTROL, TREATMENT], size=SYNTH_N)
    is_treat = version == TREATMENT

    d1 = rng.random(SYNTH_N) < (SYNTH_BASELINE_D1 + is_treat * SYNTH_TRUE_EFFECT_D1)
    d7 = rng.random(SYNTH_N) < (SYNTH_BASELINE_D7 + is_treat * SYNTH_TRUE_EFFECT_D7)
    d7 &= d1 | (rng.random(SYNTH_N) < 0.3)  # most day-7 returners also came day-1

    rounds = rng.lognormal(mean=2.6, sigma=1.6, size=SYNTH_N).astype(int)
    rounds = np.where(d1, rounds * 3, rounds)

    df = pd.DataFrame({
        "userid": np.arange(SYNTH_N),
        "version": version,
        "sum_gamerounds": rounds,
        "retention_1": d1,
        "retention_7": d7,
    })
    return df, False


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Documented cleaning. Every removal gets a stated reason and a count."""
    log, n0 = [], len(df)

    dupes = df["userid"].duplicated().sum()
    if dupes:
        df = df.drop_duplicates("userid", keep="first")
        log.append(f"Dropped {dupes:,} duplicate userids (kept first occurrence).")

    # Cookie Cats contains one player with ~50k rounds in a week. That is a bot
    # or an instrumentation error, not a person. It cannot shift a retention
    # proportion, but it would badly distort the mean-rounds guardrail.
    if "sum_gamerounds" in df:
        cutoff = df["sum_gamerounds"].quantile(0.99999)
        outliers = (df["sum_gamerounds"] > cutoff).sum()
        if outliers:
            removed = df.loc[df["sum_gamerounds"] > cutoff, "sum_gamerounds"].tolist()
            df = df[df["sum_gamerounds"] <= cutoff]
            log.append(
                f"Dropped {outliers} extreme outlier(s) above the 99.999th "
                f"percentile ({cutoff:,.0f} rounds): {removed}. Implausible for "
                f"human play; distorts the mean-rounds guardrail."
            )

    for col in ["retention_1", "retention_7"]:
        if col in df:
            df[col] = df[col].astype(bool)

    log.append(f"Final sample: {len(df):,} users ({n0 - len(df):,} removed).")
    return df, log


# --------------------------------------------------------------------------
# Analysis steps
# --------------------------------------------------------------------------


def step_sanity(df: pd.DataFrame) -> dict:
    """Randomisation checks. Run before any outcome is examined."""
    n_c = int((df["version"] == CONTROL).sum())
    n_t = int((df["version"] == TREATMENT).sum())
    srm = abtest.srm_check(n_c, n_t, EXPECTED_SPLIT)

    print("\n[1] RANDOMISATION CHECKS")
    print(f"  control {n_c:,} / treatment {n_t:,}  "
          f"(observed split {srm['observed_ratio_a']:.4f})")
    print(f"  SRM chi2 = {srm['chi2']:.3f}, p = {srm['p_value']:.4f} -> "
          f"{'PASS' if srm['passed'] else 'FAIL — STOP, ASSIGNMENT IS BROKEN'}")

    if not srm["passed"]:
        print("  A failed SRM invalidates everything downstream. Investigate "
              "assignment and logging before interpreting any result.")

    return {"srm": srm}


def step_power(df: pd.DataFrame) -> dict:
    """What could this sample ever have detected?"""
    baseline = float(df.loc[df["version"] == CONTROL, PRIMARY_METRIC].mean())
    n_per_arm = int(min((df["version"] == CONTROL).sum(),
                        (df["version"] == TREATMENT).sum()))
    mde = abtest.minimum_detectable_effect(baseline, n_per_arm, ALPHA, 0.80)

    print("\n[2] POWER (solved backwards on a fixed sample)")
    print(f"  baseline {PRIMARY_METRIC} = {baseline:.4f}, n/arm = {n_per_arm:,}")
    print(f"  MDE at 80% power = {mde['mde_absolute']:+.4f} absolute "
          f"({mde['mde_relative']:+.2%} relative)")
    print(f"  Effects smaller than {abs(mde['mde_absolute']):.4f} would likely "
          f"be missed. A null result means 'no large effect', not 'no effect'.")

    return {"mde": mde}


def step_primary(df: pd.DataFrame) -> dict:
    """The one test that was pre-registered as primary."""
    c = df[df["version"] == CONTROL]
    t = df[df["version"] == TREATMENT]
    res = abtest.two_proportion_test(
        int(c[PRIMARY_METRIC].sum()), len(c),
        int(t[PRIMARY_METRIC].sum()), len(t), ALPHA,
    )
    decision = abtest.ship_decision(res, PRACTICAL_THRESHOLD)

    print(f"\n[3] PRIMARY METRIC — {PRIMARY_METRIC}")
    print(f"  control   {res['rate_a']:.4f}   ({int(c[PRIMARY_METRIC].sum()):,}/{len(c):,})")
    print(f"  treatment {res['rate_b']:.4f}   ({int(t[PRIMARY_METRIC].sum()):,}/{len(t):,})")
    print(f"  effect    {res['abs_diff']:+.4f} absolute ({res['rel_lift']:+.2%} relative)")
    print(f"  95% CI    [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]")
    print(f"  z = {res['z_stat']:.3f}, p = {res['p_value']:.5f}")
    print(f"  -> {decision['verdict']}: {decision['rationale']}")

    return {"result": res, "decision": decision}


def step_guardrails(df: pd.DataFrame) -> dict:
    """Secondary metrics that must not be quietly harmed."""
    print("\n[4] GUARDRAIL METRICS")
    out = {}

    for metric in GUARDRAILS:
        c, t = df[df["version"] == CONTROL], df[df["version"] == TREATMENT]
        r = abtest.two_proportion_test(
            int(c[metric].sum()), len(c), int(t[metric].sum()), len(t), ALPHA)
        out[metric] = r
        print(f"  {metric}: {r['rate_a']:.4f} -> {r['rate_b']:.4f}  "
              f"({r['abs_diff']:+.4f}, 95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
              f"p = {r['p_value']:.4f})")

    # Continuous guardrail: parametric and non-parametric, because rounds
    # played is severely right-skewed and the t-test leans on a normality
    # that is not present.
    c_r = df.loc[df["version"] == CONTROL, "sum_gamerounds"].values
    t_r = df.loc[df["version"] == TREATMENT, "sum_gamerounds"].values

    welch = abtest.welch_ttest(c_r, t_r, ALPHA)
    boot = abtest.bootstrap_diff(c_r, t_r, n_boot=5000, alpha=ALPHA)
    boot_med = abtest.bootstrap_diff(c_r, t_r, n_boot=5000, alpha=ALPHA, stat=np.median)

    out["sum_gamerounds"] = {"welch": welch, "bootstrap": boot, "bootstrap_median": boot_med}

    print(f"  sum_gamerounds (mean): {welch['mean_a']:.1f} -> {welch['mean_b']:.1f}  "
          f"({welch['abs_diff']:+.2f}, p = {welch['p_value']:.4f})")
    print(f"    Welch CI     [{welch['ci_low']:+.2f}, {welch['ci_high']:+.2f}]")
    print(f"    bootstrap CI [{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]  "
          f"(skew {float(pd.Series(c_r).skew()):.1f} — trust this one)")
    print(f"    median diff  {boot_med['abs_diff']:+.1f} "
          f"[{boot_med['ci_low']:+.1f}, {boot_med['ci_high']:+.1f}]")

    return out


def step_segments(df: pd.DataFrame) -> dict:
    """Exploratory segmentation, corrected and correctly caveated."""
    print("\n[5] SEGMENTS (exploratory — hypothesis-generating only)")

    # Engagement tier is post-treatment: the gate itself can change how many
    # rounds a player gets through, so slicing on it can manufacture a
    # difference where none exists. Segmenting on it anyway, and saying so, is
    # the honest version of a common mistake.
    df = df.copy()
    df["segment"] = pd.qcut(df["sum_gamerounds"], 4,
                            labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
                            duplicates="drop")

    rows = []
    for seg in df["segment"].cat.categories:
        s = df[df["segment"] == seg]
        c, t = s[s["version"] == CONTROL], s[s["version"] == TREATMENT]
        if len(c) < 30 or len(t) < 30:
            continue
        r = abtest.two_proportion_test(
            int(c[PRIMARY_METRIC].sum()), len(c),
            int(t[PRIMARY_METRIC].sum()), len(t), ALPHA)
        rows.append({"segment": str(seg), **r})

    bh = abtest.benjamini_hochberg([r["p_value"] for r in rows], fdr=0.05)
    for r, adj, rej in zip(rows, bh["p_adjusted"], bh["rejected"]):
        r["p_adjusted"], r["significant_after_bh"] = adj, bool(rej)
        print(f"  {r['segment']:<12} {r['rate_a']:.4f} -> {r['rate_b']:.4f}  "
              f"({r['abs_diff']:+.4f})  p={r['p_value']:.4f} -> "
              f"BH p={adj:.4f} {'*' if rej else ''}")

    print(f"  {bh['n_rejected']}/{bh['n_tests']} segments survive FDR correction.")
    print("  Caveat: engagement tier is measured AFTER treatment, so these splits")
    print("  are descriptive, not causal. Treat as leads for a follow-up test.")

    return {"segments": rows, "bh": bh}


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def make_charts(df: pd.DataFrame, primary: dict, is_real: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    tag = "" if is_real else "  [SYNTHETIC DATA — NOT A FINDING]"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    res = primary["result"]

    # (a) Primary metric with error bars — the headline chart.
    ax = axes[0]
    rates = [res["rate_a"], res["rate_b"]]
    errs = [1.96 * np.sqrt(r * (1 - r) / n)
            for r, n in zip(rates, [res["n_a"], res["n_b"]])]
    ax.bar(["gate_30\n(control)", "gate_40\n(treatment)"], rates,
           yerr=errs, capsize=8, color=["#4C72B0", "#DD8452"], width=0.55)
    ax.set_ylabel("day-7 retention")
    ax.set_title("Primary metric with 95% CI")
    ax.set_ylim(0, max(rates) * 1.35)
    for i, r in enumerate(rates):
        ax.text(i, r + errs[i] + 0.006, f"{r:.3f}", ha="center", fontweight="bold")

    # (b) Effect size vs the decision thresholds — the chart that drives the call.
    # Plotted in percentage points, which is how the decision is actually
    # discussed. Fractions with five decimal places are unreadable on an axis.
    ax = axes[1]
    eff, lo, hi = (res["abs_diff"] * 100, res["ci_low"] * 100, res["ci_high"] * 100)
    thr = PRACTICAL_THRESHOLD * 100

    ax.errorbar([eff], [0], xerr=[[eff - lo], [hi - eff]],
                fmt="o", color="#C44E52", capsize=8, markersize=11, linewidth=2.2,
                label="observed effect (95% CI)")
    ax.axvline(0, color="black", lw=1.2)
    ax.axvspan(-thr, thr, alpha=0.18, color="grey",
               label=f"±{thr:.1f}pp: too small to act on")

    span = max(abs(lo), abs(hi), thr) * 1.6
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("day-7 retention: treatment − control (percentage points)")
    ax.set_title("Effect size vs decision threshold")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.text(eff, -0.35, f"{eff:+.2f}pp\n[{lo:+.2f}, {hi:+.2f}]",
            ha="center", va="top", fontsize=9, color="#C44E52", fontweight="bold")

    # (c) Skew of the continuous guardrail — justifies the bootstrap.
    ax = axes[2]
    for v, colr, lbl in [(CONTROL, "#4C72B0", "gate_30"), (TREATMENT, "#DD8452", "gate_40")]:
        vals = df.loc[df["version"] == v, "sum_gamerounds"]
        ax.hist(np.log1p(vals), bins=50, alpha=0.55, color=colr, label=lbl)
    ax.set_xlabel("log(1 + game rounds)")
    ax.set_ylabel("players")
    ax.set_title("Rounds played is severely skewed\n(hence the bootstrap)")
    ax.legend()

    fig.suptitle(f"Cookie Cats: gate at level 30 vs 40{tag}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_DIR / 'results.png'}")


# --------------------------------------------------------------------------
# Memo generation — numbers come from the analysis, never typed by hand
# --------------------------------------------------------------------------


def write_memo(results: dict, is_real: bool) -> None:
    res = results["primary"]["result"]
    dec = results["primary"]["decision"]
    mde = results["power"]["mde"]
    srm = results["sanity"]["srm"]
    d1 = results["guardrails"]["retention_1"]
    rounds = results["guardrails"]["sum_gamerounds"]

    banner = "" if is_real else (
        "> **THIS MEMO WAS GENERATED FROM SYNTHETIC DATA.** The real "
        "`cookie_cats.csv` was not present. Numbers below are illustrative "
        "only and must not be cited. Add the CSV to `data/` and rerun.\n\n"
    )

    # Guardrail prose has to adapt to the result. A fixed template that says
    # "rounds played was flat, so the change does not affect how much people
    # play" will happily contradict itself the moment the number moves.
    w = rounds["welch"]
    if w["significant"]:
        rounds_sentence = (
            f"Rounds played also moved, from a mean of {w['mean_a']:.0f} to "
            f"{w['mean_b']:.0f} ({w['abs_diff']:+.1f}, p = {w['p_value']:.4f}), so the "
            f"gate appears to change how much people play as well as whether they "
            f"return. Note this metric is heavily right-skewed, and the median "
            f"barely moves ({rounds['bootstrap_median']['abs_diff']:+.0f} rounds) — "
            f"the mean shift is driven by the tail, not the typical player."
        )
    else:
        rounds_sentence = (
            f"Rounds played was statistically flat (mean {w['mean_a']:.0f} to "
            f"{w['mean_b']:.0f}, p = {w['p_value']:.4f}), so the change does not "
            f"appear to alter how much people play, only whether they return."
        )

    d1_sentence = (
        "pointing the same way as the day-7 result"
        if np.sign(d1["abs_diff"]) == np.sign(res["abs_diff"])
        else "pointing the opposite way from the day-7 result, which is worth a closer look"
    )

    headline = {
        "SHIP": "Move the gate to level 40.",
        "SHIP_NO": "Keep the gate at level 30. The difference is real but too small to be worth acting on.",
        "ROLLBACK": "Keep the gate at level 30. Moving it measurably hurts retention.",
        "NO_EFFECT": "Keep the gate at level 30. We can rule out any change large enough to matter.",
        "INCONCLUSIVE": "Do not decide yet. This test cannot separate 'no effect' from 'meaningful effect'.",
    }[dec["verdict"]]

    memo = f"""# Should we move the progression gate from level 30 to level 40?

{banner}**Recommendation: {headline}**

## What we tested

Cookie Cats makes players wait once they hit a progression gate. We randomly
assigned {res['n_a'] + res['n_b']:,} new players to hit that gate either at
level 30 (current behaviour) or level 40, and measured whether they came back on
day 7. Day-7 retention was chosen as the deciding metric before the test was
analysed, along with a rule that we would only act on a change of at least
{PRACTICAL_THRESHOLD * 100:.1f} percentage points — below that, reworking the
progression curve costs more than it returns.

## What we found

Day-7 retention was **{res['rate_a']:.1%}** with the gate at level 30 and
**{res['rate_b']:.1%}** with it at level 40, a difference of
**{res['abs_diff'] * 100:+.2f} percentage points** ({res['rel_lift']:+.1%} relative).
The plausible range for that difference runs from {res['ci_low'] * 100:+.2f} to
{res['ci_high'] * 100:+.2f} percentage points (p = {res['p_value']:.4f}).

{dec['rationale']}

Day-1 retention moved {d1['abs_diff'] * 100:+.2f} percentage points
({d1['rate_a']:.1%} to {d1['rate_b']:.1%}, p = {d1['p_value']:.4f}), {d1_sentence}.
{rounds_sentence}

## What we are not certain about

This sample could reliably detect a change of about
**{abs(mde['mde_absolute']) * 100:.2f} percentage points** or larger. Anything
smaller than that would likely have gone unnoticed, so this test tells us
there is no *large* effect, which is not the same as no effect at all.

Assignment looks sound: the split came out
{srm['observed_ratio_a']:.1%}/{1 - srm['observed_ratio_a']:.1%} against an
intended 50/50 (p = {srm['p_value']:.3f}), so there is no sign of a broken
randomisation corrupting the comparison.

The measurement window is short. A progression gate is a long-run retention
mechanism, and seven days may be too early to see where players actually
churn. We also cannot see any per-player context — country, device, acquisition
channel — so we cannot tell whether the answer differs by audience.

Segment-level differences did appear, but engagement tier is measured after
the treatment was applied, so those splits describe who played rather than
what the gate caused. They are leads, not conclusions.

## What we would do next

1. Extend measurement to day 14 and day 30. Day 7 is likely too early for a
   progression-pacing change to fully surface.
2. If a longer window still shows nothing, stop testing gate position and move
   the effort to a mechanism with more room to move the metric.
3. Log country and acquisition channel on assignment so the next test can be
   analysed by audience without post-treatment slicing.
4. Before running it, decide the sample size from the effect worth detecting,
   rather than analysing whatever accumulates.

---
*Analysis: `analysis.py`. Statistical methods: `abtest.py`, validated against
simulated data with known ground truth in `validate.py`.*
"""

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "memo.md").write_text(memo)
    print(f"Saved {OUT_DIR / 'memo.md'}")


# --------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("COOKIE CATS A/B TEST — gate at level 30 vs level 40")
    print("=" * 72)

    df, is_real = load_data()
    df, log = clean(df)
    print("\n[0] CLEANING")
    for line in log:
        print(f"  {line}")

    results = {
        "sanity": step_sanity(df),
        "power": step_power(df),
        "primary": step_primary(df),
    }
    results["guardrails"] = step_guardrails(df)
    results["segments"] = step_segments(df)

    make_charts(df, results["primary"], is_real)
    write_memo(results, is_real)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"Saved {OUT_DIR / 'results.json'}")

    print("\n" + "=" * 72)
    print(f"DECISION: {results['primary']['decision']['verdict']}")
    print(results["primary"]["decision"]["rationale"])
    if not is_real:
        print("\n(Synthetic data — replace with the real CSV before citing anything.)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
