# Should we move the progression gate from level 30 to level 40?

> **THIS MEMO WAS GENERATED FROM SYNTHETIC DATA.** The real `cookie_cats.csv` was not present. Numbers below are illustrative only and must not be cited. Add the CSV to `data/` and rerun.

**Recommendation: Keep the gate at level 30. The difference is real but too small to be worth acting on.**

## What we tested

Cookie Cats makes players wait once they hit a progression gate. We randomly
assigned 89,999 new players to hit that gate either at
level 30 (current behaviour) or level 40, and measured whether they came back on
day 7. Day-7 retention was chosen as the deciding metric before the test was
analysed, along with a rule that we would only act on a change of at least
1.0 percentage points — below that, reworking the
progression curve costs more than it returns.

## What we found

Day-7 retention was **11.7%** with the gate at level 30 and
**11.2%** with it at level 40, a difference of
**-0.52 percentage points** (-4.4% relative).
The plausible range for that difference runs from -0.93 to
-0.10 percentage points (p = 0.0149).

The effect is real but smaller than the pre-registered threshold. The treatment is reliably worse than control by an amount too small to justify acting on.

Day-1 retention moved -0.33 percentage points
(44.9% to 44.5%, p = 0.3131), pointing the same way as the day-7 result.
Rounds played also moved, from a mean of 95 to 90 (-5.0, p = 0.0213), so the gate appears to change how much people play as well as whether they return. Note this metric is heavily right-skewed, and the median barely moves (+0 rounds) — the mean shift is driven by the tail, not the typical player.

## What we are not certain about

This sample could reliably detect a change of about
**0.61 percentage points** or larger. Anything
smaller than that would likely have gone unnoticed, so this test tells us
there is no *large* effect, which is not the same as no effect at all.

Assignment looks sound: the split came out
49.9%/50.1% against an
intended 50/50 (p = 0.377), so there is no sign of a broken
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
