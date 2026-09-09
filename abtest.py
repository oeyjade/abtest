"""
abtest.py — reusable statistics for two-arm experiments.

Every function returns a plain dict so results can be dropped straight into a
DataFrame or a report template. Nothing here prints or plots; that is the
caller's job.

Design notes worth defending in an interview:
  - Two-proportion z-test uses the POOLED proportion for the test statistic
    (correct under H0: p_a == p_b) but the UNPOOLED standard error for the
    confidence interval (we are no longer assuming H0 when we estimate the
    size of the difference). Mixing these up is the single most common bug
    in homegrown A/B test code.
  - Continuous metrics get Welch's t-test, never Student's. Equal variance
    between arms is an assumption you have no reason to make.
  - Revenue-style metrics are heavily skewed, so a bootstrap CI on the
    difference in means is reported alongside the parametric one. If they
    disagree, trust the bootstrap.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# --------------------------------------------------------------------------
# 1. Randomisation sanity checks — run these BEFORE looking at any outcome
# --------------------------------------------------------------------------


def srm_check(n_a: int, n_b: int, expected_ratio: float = 0.5, alarm_p: float = 0.001) -> dict:
    """Sample Ratio Mismatch check.

    If users were assigned 50/50 but the data comes back 51/49 on a large
    sample, the assignment or the logging is broken and every downstream
    number is suspect. This is a chi-square goodness-of-fit test against the
    intended split.

    The alarm threshold is 0.001, not 0.05, because with a few hundred
    experiments a year a 5% threshold would cry wolf constantly.

    expected_ratio is the intended share going to arm A.
    """
    n = n_a + n_b
    expected = np.array([n * expected_ratio, n * (1 - expected_ratio)])
    observed = np.array([n_a, n_b])

    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=1))

    return {
        "test": "srm_chi2",
        "n_a": int(n_a),
        "n_b": int(n_b),
        "observed_ratio_a": n_a / n,
        "expected_ratio_a": expected_ratio,
        "chi2": chi2,
        "p_value": p,
        "passed": p >= alarm_p,
        "alarm_p": alarm_p,
    }


def covariate_balance(a_values, b_values, name: str = "covariate") -> dict:
    """Check a pre-treatment variable is balanced across arms.

    A covariate is only valid here if it could not possibly have been
    affected by the treatment (signup date, device type, country). Testing a
    post-treatment variable for 'balance' is a category error.

    A significant imbalance on a pre-treatment variable means randomisation
    failed, and is a reason to stop and investigate rather than to adjust.
    """
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    t, p = stats.ttest_ind(a, b, equal_var=False)

    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    smd = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0

    return {
        "test": "covariate_balance",
        "covariate": name,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "std_mean_diff": float(smd),  # |SMD| > 0.1 is the usual concern line
        "p_value": float(p),
        "balanced": bool(abs(smd) < 0.1),
    }


# --------------------------------------------------------------------------
# 2. Power — what were we ever able to detect?
# --------------------------------------------------------------------------


def minimum_detectable_effect(
    baseline_rate: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80
) -> dict:
    """Solve the power equation backwards for an already-collected sample.

    On a historical dataset you cannot choose n, so the useful question is
    not 'was it significant' but 'how small an effect could this sample have
    caught at all'. Reporting the MDE is what lets you say a null result
    means something rather than nothing.
    """
    analysis = NormalIndPower()
    h = analysis.solve_power(
        effect_size=None, nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0,
        alternative="two-sided",
    )

    # h is Cohen's h; invert arcsine transform to get the detectable rate.
    # h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1))
    p2 = np.sin(np.arcsin(np.sqrt(baseline_rate)) + h / 2) ** 2
    abs_mde = float(p2 - baseline_rate)

    return {
        "baseline_rate": baseline_rate,
        "n_per_arm": int(n_per_arm),
        "alpha": alpha,
        "power": power,
        "cohens_h": float(h),
        "mde_absolute": abs_mde,
        "mde_relative": abs_mde / baseline_rate,
    }


def required_sample_size(
    baseline_rate: float, target_lift_relative: float, alpha: float = 0.05, power: float = 0.80
) -> dict:
    """Forward power calculation — use when you get to design the experiment."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + target_lift_relative)
    h = proportion_effectsize(p2, p1)
    n = NormalIndPower().solve_power(
        effect_size=h, nobs1=None, alpha=alpha, power=power, ratio=1.0,
        alternative="two-sided",
    )
    return {
        "baseline_rate": p1,
        "target_rate": p2,
        "n_per_arm": int(np.ceil(n)),
        "n_total": int(np.ceil(n) * 2),
    }


# --------------------------------------------------------------------------
# 3. Hypothesis tests
# --------------------------------------------------------------------------


def two_proportion_test(
    conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05
) -> dict:
    """Two-sided z-test for a difference in conversion rates.

    Reports absolute difference, relative lift, and a CI on the absolute
    difference. The CI is the number a stakeholder actually needs: it answers
    'how bad could this plausibly be', which a p-value never does.

    Convention throughout: effect = B - A, so a positive effect means the
    treatment arm B is higher.
    """
    p_a, p_b = conv_a / n_a, conv_b / n_b
    diff = p_b - p_a

    # Pooled SE — correct under the null, used for the test statistic.
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))

    # Unpooled SE — we are estimating the effect, not assuming it is zero.
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - crit * se_unpooled, diff + crit * se_unpooled)

    return {
        "test": "two_proportion_z",
        "rate_a": p_a,
        "rate_b": p_b,
        "n_a": int(n_a),
        "n_b": int(n_b),
        "abs_diff": float(diff),
        "rel_lift": float(diff / p_a) if p_a > 0 else np.nan,
        "z_stat": float(z),
        "p_value": p_value,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "alpha": alpha,
        "significant": p_value < alpha,
    }


def welch_ttest(a_values, b_values, alpha: float = 0.05) -> dict:
    """Welch's t-test on a continuous metric. Effect = mean(B) - mean(A)."""
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = b.mean() - a.mean()

    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )
    crit = stats.t.ppf(1 - alpha / 2, df)

    return {
        "test": "welch_t",
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "n_a": len(a),
        "n_b": len(b),
        "abs_diff": float(diff),
        "rel_lift": float(diff / a.mean()) if a.mean() != 0 else np.nan,
        "t_stat": float(t),
        "df": float(df),
        "p_value": float(p),
        "ci_low": float(diff - crit * se),
        "ci_high": float(diff + crit * se),
        "significant": float(p) < alpha,
    }


def bootstrap_diff(
    a_values, b_values, n_boot: int = 10_000, alpha: float = 0.05,
    stat=np.mean, seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for a difference in a summary statistic.

    Makes no distributional assumption, which matters for revenue and session
    length where the parametric CI leans on a normality that is not there.
    Swap `stat` for np.median or a trimmed mean when the tail is the problem
    rather than the centre.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)

    # Resample in batches. The naive version allocates an (n_boot x n) index
    # matrix, which is 1.8 GB at 5k resamples of 45k users and will kill the
    # process. Batching caps peak memory at roughly batch_size x n floats.
    max_cells = 20_000_000
    batch = max(1, min(n_boot, max_cells // max(len(a), len(b), 1)))

    diffs = np.empty(n_boot)
    done = 0
    while done < n_boot:
        k = min(batch, n_boot - done)
        idx_a = rng.integers(0, len(a), size=(k, len(a)))
        idx_b = rng.integers(0, len(b), size=(k, len(b)))
        diffs[done:done + k] = stat(b[idx_b], axis=1) - stat(a[idx_a], axis=1)
        done += k

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    observed = float(stat(b) - stat(a))

    return {
        "test": "bootstrap",
        "statistic": getattr(stat, "__name__", str(stat)),
        "n_boot": n_boot,
        "abs_diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        # A CI excluding zero is the bootstrap analogue of significance.
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# --------------------------------------------------------------------------
# 4. Multiple comparisons
# --------------------------------------------------------------------------


def benjamini_hochberg(p_values, fdr: float = 0.05) -> dict:
    """Control the false discovery rate across a family of segment tests.

    Run ten segment tests at alpha=0.05 on a true null and you expect roughly
    one 'finding' by luck alone. BH is the right correction for exploratory
    segmentation: less brutal than Bonferroni, and it controls the quantity
    you actually care about (share of claimed findings that are false).

    Segment results remain hypothesis-generating regardless of correction.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    thresholds = fdr * np.arange(1, m + 1) / m
    passing = ranked <= thresholds
    k = np.max(np.nonzero(passing)[0]) + 1 if passing.any() else 0

    rejected = np.zeros(m, dtype=bool)
    if k > 0:
        rejected[order[:k]] = True

    # Step-up adjusted p-values, monotonised.
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adj, 1.0)

    return {
        "method": "benjamini_hochberg",
        "fdr": fdr,
        "n_tests": m,
        "n_rejected": int(rejected.sum()),
        "rejected": rejected.tolist(),
        "p_adjusted": adjusted.tolist(),
    }


# --------------------------------------------------------------------------
# 5. Decision layer — the part that separates analysis from advice
# --------------------------------------------------------------------------


def ship_decision(result: dict, practical_threshold: float) -> dict:
    """Combine statistical evidence with a pre-registered business threshold.

    `practical_threshold` is the smallest absolute effect worth the cost of
    shipping, and it must be chosen BEFORE seeing results or it is just a
    rationalisation.

    Four outcomes, and the interesting one is SHIP_NO: statistically real,
    practically too small. Recommending against a significant result is the
    most credible thing an analyst can do.
    """
    lo, hi = result["ci_low"], result["ci_high"]
    sig = result["significant"]
    effect = result["abs_diff"]
    t = practical_threshold

    if sig and effect >= t:
        verdict, rationale = "SHIP", (
            "The effect is statistically distinguishable from zero and clears the "
            "pre-registered threshold for practical value."
        )
    elif sig and effect <= -t:
        verdict, rationale = "ROLLBACK", (
            "The treatment is materially worse than control: the harm is both "
            "statistically reliable and larger than the threshold we said would "
            "matter."
        )
    elif sig:
        # Reliably non-zero, but too small to be worth acting on. Direction
        # still matters for how it reads: a small harm reinforces staying put,
        # a small gain does not justify the build cost. Neither is a ship.
        # No units are formatted here: this function does not know whether the
        # metric is a proportion, a currency amount or a duration. The caller
        # owns presentation.
        direction = "better" if effect > 0 else "worse"
        verdict, rationale = "SHIP_NO", (
            f"The effect is real but smaller than the pre-registered threshold. "
            f"The treatment is reliably {direction} than control by an amount too "
            f"small to justify acting on."
        )
    elif lo > -t and hi < t:
        verdict, rationale = "NO_EFFECT", (
            "The confidence interval rules out any effect large enough to matter. "
            "This is an informative null, not an inconclusive one."
        )
    else:
        verdict, rationale = "INCONCLUSIVE", (
            "The confidence interval still contains effects large enough to matter. "
            "The sample cannot separate 'no effect' from 'meaningful effect'."
        )

    return {
        "verdict": verdict,
        "rationale": rationale,
        "practical_threshold": t,
        "observed_effect": effect,
        "ci": (lo, hi),
    }
