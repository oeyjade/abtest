"""
validate.py — prove the analysis pipeline is correct before trusting it on data.

The premise: if you generate data with a KNOWN true effect and your pipeline
reports the wrong thing, the pipeline is broken. Almost nobody checks this,
and it is cheap to do.

Five properties get checked, each thousands of times:

  1. False positive rate    — with a true effect of zero, a test at alpha=0.05
                              should wrongly declare significance ~5% of the time.
                              Not 0%, not 20%. Exactly the rate advertised.
  2. Statistical power      — at an effect equal to the computed MDE, the test
                              should detect it ~80% of the time.
  3. CI coverage            — a 95% interval should contain the true effect
                              ~95% of the time.
  4. SRM sensitivity        — a deliberately broken 52/48 split should be caught.
  5. FDR control            — under all-null segment testing, Benjamini-Hochberg
                              should keep false discoveries near the stated rate.

Run:  python validate.py
"""

from __future__ import annotations

import numpy as np

import abtest

RNG = np.random.default_rng(7)
N_SIMS = 4000
N_PER_ARM = 5000
BASELINE = 0.20


def _simulate_arm(n: int, rate: float) -> int:
    """Return number of conversions in one arm."""
    return int(RNG.binomial(n, rate))


def check_false_positive_rate(alpha: float = 0.05) -> dict:
    """True effect is ZERO. Rejections should occur at exactly alpha."""
    rejections = 0
    for _ in range(N_SIMS):
        ca = _simulate_arm(N_PER_ARM, BASELINE)
        cb = _simulate_arm(N_PER_ARM, BASELINE)  # identical rate
        r = abtest.two_proportion_test(ca, N_PER_ARM, cb, N_PER_ARM, alpha=alpha)
        rejections += r["significant"]

    observed = rejections / N_SIMS
    # Monte Carlo standard error on a proportion, times 3 for the tolerance band.
    tol = 3 * np.sqrt(alpha * (1 - alpha) / N_SIMS)
    return {
        "check": "false positive rate",
        "expected": alpha,
        "observed": observed,
        "tolerance": tol,
        "pass": abs(observed - alpha) < tol,
        "note": "true effect = 0; rejections here are all false alarms",
    }


def check_power(target_power: float = 0.80) -> dict:
    """Inject an effect exactly equal to the MDE; should detect ~80% of the time."""
    mde = abtest.minimum_detectable_effect(BASELINE, N_PER_ARM, power=target_power)
    true_effect = mde["mde_absolute"]
    treatment_rate = BASELINE + true_effect

    detected = 0
    for _ in range(N_SIMS):
        ca = _simulate_arm(N_PER_ARM, BASELINE)
        cb = _simulate_arm(N_PER_ARM, treatment_rate)
        r = abtest.two_proportion_test(ca, N_PER_ARM, cb, N_PER_ARM)
        detected += r["significant"]

    observed = detected / N_SIMS
    tol = 3 * np.sqrt(target_power * (1 - target_power) / N_SIMS) + 0.01
    return {
        "check": "statistical power at MDE",
        "expected": target_power,
        "observed": observed,
        "tolerance": tol,
        "pass": abs(observed - target_power) < tol,
        "note": f"injected true lift of {true_effect:+.4f} absolute",
    }


def check_ci_coverage(alpha: float = 0.05) -> dict:
    """A 95% CI must contain the true effect 95% of the time. Tests the
    unpooled-SE choice — using the pooled SE here would under-cover."""
    true_effect = 0.03
    treatment_rate = BASELINE + true_effect

    contained = 0
    for _ in range(N_SIMS):
        ca = _simulate_arm(N_PER_ARM, BASELINE)
        cb = _simulate_arm(N_PER_ARM, treatment_rate)
        r = abtest.two_proportion_test(ca, N_PER_ARM, cb, N_PER_ARM, alpha=alpha)
        contained += r["ci_low"] <= true_effect <= r["ci_high"]

    target = 1 - alpha
    observed = contained / N_SIMS
    tol = 3 * np.sqrt(target * (1 - target) / N_SIMS)
    return {
        "check": "95% CI coverage",
        "expected": target,
        "observed": observed,
        "tolerance": tol,
        "pass": abs(observed - target) < tol,
        "note": f"true effect {true_effect:+.3f} should fall inside the interval",
    }


def check_srm_detection() -> dict:
    """A healthy 50/50 split must pass; a broken 52/48 split must be caught."""
    n = 100_000

    healthy = abtest.srm_check(*np.random.default_rng(1).multinomial(n, [0.5, 0.5]))
    broken = abtest.srm_check(int(n * 0.52), int(n * 0.48))

    return {
        "check": "SRM detection",
        "expected": "healthy passes, 52/48 caught",
        "observed": (
            f"healthy p={healthy['p_value']:.3f} ({'pass' if healthy['passed'] else 'FLAG'}), "
            f"broken p={broken['p_value']:.2e} ({'pass' if broken['passed'] else 'FLAG'})"
        ),
        "tolerance": None,
        "pass": healthy["passed"] and not broken["passed"],
        "note": "a 2pp skew on 100k users is not bad luck, it is a bug",
    }


def check_fdr_control(fdr: float = 0.05) -> dict:
    """Test 10 segments where NO segment has a real effect.

    Uncorrected, ~40% of runs produce at least one 'finding'. BH should pull
    the false discovery rate down to roughly the stated level.
    """
    n_segments, n_runs, n_seg = 10, 800, 2000

    raw_any, bh_any = 0, 0
    for _ in range(n_runs):
        pvals = []
        for _ in range(n_segments):
            ca = _simulate_arm(n_seg, BASELINE)
            cb = _simulate_arm(n_seg, BASELINE)  # no effect anywhere
            pvals.append(abtest.two_proportion_test(ca, n_seg, cb, n_seg)["p_value"])

        raw_any += any(p < 0.05 for p in pvals)
        bh_any += abtest.benjamini_hochberg(pvals, fdr=fdr)["n_rejected"] > 0

    return {
        "check": "FDR control on 10 null segments",
        "expected": f"<= {fdr:.0%} of runs with a false finding",
        "observed": f"uncorrected {raw_any / n_runs:.1%} -> BH {bh_any / n_runs:.1%}",
        "tolerance": None,
        "pass": (bh_any / n_runs) <= fdr * 1.6,
        "note": "shows why unadjusted segment slicing manufactures findings",
    }


def main() -> int:
    print("=" * 72)
    print("PIPELINE VALIDATION — testing the analysis code against known truth")
    print("=" * 72)
    print(f"{N_SIMS:,} simulations per check, {N_PER_ARM:,} users per arm\n")

    checks = [
        check_false_positive_rate(),
        check_power(),
        check_ci_coverage(),
        check_srm_detection(),
        check_fdr_control(),
    ]

    for c in checks:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"[{status}] {c['check']}")
        if c["tolerance"] is not None:
            print(f"        expected {c['expected']:.3f}  observed {c['observed']:.3f}"
                  f"  (tolerance +/-{c['tolerance']:.3f})")
        else:
            print(f"        expected {c['expected']}")
            print(f"        observed {c['observed']}")
        print(f"        {c['note']}\n")

    n_pass = sum(c["pass"] for c in checks)
    print("-" * 72)
    print(f"{n_pass}/{len(checks)} checks passed")
    if n_pass == len(checks):
        print("Pipeline behaves correctly on data with known ground truth.")
    else:
        print("FIX THE PIPELINE BEFORE RUNNING IT ON REAL DATA.")
    print("-" * 72)

    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
