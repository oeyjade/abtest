# Should a mobile game move its progression gate from level 30 to level 40?

An end-to-end analysis of a randomised experiment on ~90,000 Cookie Cats
players, from randomisation checks through to a written recommendation.

**The short version:** the gate move produced a small, statistically reliable
*drop* in day-7 retention — but one smaller than the threshold set in advance
for acting on it. The recommendation is to leave the gate where it is, and the
interesting part is *why a significant result is not automatically a reason to
ship*.

![Results](output/results.png)

**Read the [one-page memo](output/memo.md) first.** It is written for a product
manager, not a statistician, and it is the actual output of this project. The
code exists to make the memo trustworthy.

---

## What makes this more than a t-test

Most A/B test portfolio projects call `scipy.stats.ttest_ind`, check whether
p < 0.05, and stop. Five things here that do not usually appear:

**The pipeline is validated against known ground truth.** `validate.py`
generates thousands of experiments where the true effect is something I chose,
then checks the analysis code recovers it. If the false positive rate is not
5%, or a 95% interval does not contain the truth 95% of the time, the code is
wrong and I would rather find that out on fake data than in a recommendation.

```
[PASS] false positive rate          expected 0.050  observed 0.049
[PASS] statistical power at MDE     expected 0.800  observed 0.789
[PASS] 95% CI coverage              expected 0.950  observed 0.953
[PASS] SRM detection                healthy p=0.552 pass, broken p=1.1e-36 FLAG
[PASS] FDR control on 10 segments   uncorrected 39.8% -> BH 5.6%
```

That last line is the argument for correcting segment tests, made empirically:
slice a *completely null* experiment ten ways and 40% of the time you find
something. It is not a finding, it is arithmetic.

**Randomisation is checked before any outcome is looked at.** A Sample Ratio
Mismatch test catches broken assignment or logging. If the split is off, no
downstream number means anything, and the correct output of the analysis is
"this experiment is invalid" rather than a p-value.

**Power is solved backwards.** You cannot choose the sample size on data that
already exists, so the useful question is what this sample could ever have
detected. Reporting the minimum detectable effect turns "not significant" from
an empty result into an informative one: no *large* effect, which is not the
same as no effect.

**Statistical and practical significance are kept apart.** A decision threshold
is fixed before the analysis, and the code distinguishes five outcomes — ship,
don't ship, roll back, informative null, and genuinely inconclusive. The
distinction between "we can rule out anything that matters" and "we cannot
tell" is the one stakeholders most often need and least often get.

**The skew is taken seriously.** Rounds played has a skew of ~16, so the
parametric interval is reported next to a bootstrap one, and next to the median.
The mean moves by 5 rounds; the median does not move at all. That gap is the
whole story of the metric, and a t-test alone hides it.

---

## Running it

```bash
pip install -r requirements.txt
python validate.py     # prove the statistics are correct    (~1 min)
python analysis.py     # run the analysis, write the memo    (~30 s)
```

Download `cookie_cats.csv` from the Kaggle *Mobile Games A/B Testing* dataset
into `data/`. Without it the pipeline runs on synthetic stand-in data so it is
testable end to end; every synthetic artefact is watermarked and the memo
carries a warning banner.

## Layout

| File | What it does |
|---|---|
| `abtest.py` | Reusable statistics: SRM, power, two-proportion z, Welch, bootstrap, Benjamini–Hochberg, decision rule |
| `validate.py` | Simulation harness — checks the above against known ground truth |
| `analysis.py` | The experiment protocol, start to finish |
| `output/memo.md` | The recommendation, generated from computed numbers |
| `output/results.png` | Primary metric, effect vs threshold, and the skew |
| `output/results.json` | Every number, for reuse |

No figure in the memo is typed by hand. Every number is interpolated from the
analysis output, so the write-up cannot drift out of sync with the code.

## Choices worth defending

- **Pooled SE for the test statistic, unpooled for the interval.** Under the
  null the proportions are equal, so pooling is right for the test. Once
  estimating the size of a difference, that assumption is gone. The CI coverage
  check in `validate.py` fails if these are swapped.
- **Welch, never Student.** Equal variance across arms is an assumption with
  nothing behind it.
- **Day-7 retention as primary, chosen in advance.** A gate at level 30 versus
  40 cannot plausibly bite within 24 hours. Fixing this before analysis is what
  stops the project becoming a hunt for whichever metric happened to move.
- **Segments are labelled as exploratory and corrected.** Engagement tier is
  measured *after* treatment, so slicing on it is descriptive, not causal.
  Saying so in the memo matters more than the segmentation itself.

## Honest limitations

Seven days is short for a retention mechanism, the dataset carries no country,
device or acquisition channel, and there are no timestamps, so novelty effects
and day-over-day drift cannot be checked. A single experiment on one game does
not generalise to progression design in general.
