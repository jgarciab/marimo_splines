# From polynomials to smoothing splines

Standalone Marimo app for the ML4P nonlinear lecture (session 3). It uses the
lecture's example: height against age for 300 people from NHANES 2009–2012.
No quizzes or answer gates.

## Run

From this folder, with [uv](https://docs.astral.sh/uv/) installed:

```bash
bash run.sh
```

All code and data are in `app.py`. To use just that file elsewhere:

```bash
uvx marimo run --sandbox app.py
```

## What is on the page

**Six fits, the same people.** A 2 × 3 grid (stacked on a phone), shared axes.

1. **Polynomial:** one global polynomial, degree 0–3.
2. **Piecewise, per region:** an independent polynomial per region; jumps at knots.
3. **Piecewise, adjustments:** the same fit, written as a baseline plus adjustments
   after each knot.
4. **Regression spline:** degree d with matching value and first d − 1 derivatives
   at the knots (`bs()`). Beyond ages 2 and 80 the last piece continues.
5. **Natural cubic spline:** straight beyond the boundary knots 2 and 80, zero
   curvature there (`ns()`). Shown for d = 3 only; see below.
6. **Smoothing spline:** cubic, a knot at every distinct age, penalty λ∫f''².

**Choosing K and λ.** Two plots side by side: training RMSE (the 300 people) and
test RMSE (the lecture's 3,607 test people) against the number of coefficients
(effective df for the smoothing spline, log scale). One line per model: polynomial
degree 0–12, piecewise and splines with 0–20 knots at the current degree, smoothing
spline over a grid of λ. The natural spline line appears only when d = 3. Open circles mark the current slider settings, stars the
lowest test RMSE; a table lists those settings. The page says that the test set is
used here only to show the pattern and that K or λ should be chosen by
cross-validation on the training data.

Controls (sidebar): degree d, 0–3 (plots 1–4; plot 5 needs d = 3), number of interior knots K, 1–6, default 2 (plots 2–5),
λ (plot 6), a focus dropdown, and a checkbox for the extrapolation zones (ages 0–2
and 80–100, shown by default).

## Data

NHANES 2009–2012 from the R package `NHANES` (2.1.4), data frame `NHANESraw`: people
aged 2 or older with measured height (n = 18,035, unique IDs). Split and subsample
exactly as in the lecture:

```r
set.seed(2026); train_index <- sample.int(nrow(people), floor(.8 * nrow(people)))
set.seed(16);   small <- training[sample.int(nrow(training), 300L), ]
```

`app.py` embeds the ages and heights of the 300 people (`small`) and of the 3,607
test people. Age is in whole years and top-coded at 80; 77 distinct ages occur in
the 300. Unweighted; a teaching example, not population inference.

## Mathematical details

- **Knots.** K interior knots at equally spaced quantiles of age in the 300 people,
  as `bs(df = )` and `ns(df = )` do. Boundary knots at 2 and 80. Quantile knots move
  when K changes, so the training error of a spline is not guaranteed to fall with K.
- **Counts (including the intercept).** Polynomial d + 1; piecewise (K + 1)(d + 1);
  regression spline d + 1 + K (degree 0 unavailable); natural cubic spline K + 2.
- **Why the natural spline is cubic only.** Natural splines are defined for odd degree
  2m − 1, with a degree m − 1 polynomial beyond the boundary knots (the solution of the
  penalised problem with ∫(f⁽ᵐ⁾)²). They are not defined for d = 0 or d = 2. For d = 1
  the natural spline exists (flat tails) but adds no constraint inside the data, so it
  fits exactly like the regression spline; the app shows a message instead. The cubic
  case imposes f''(2) = f''(80) = 0 (two coefficients fewer, K + 2), as the null space
  of those constraints on the truncated-power columns, with straight tails outside
  2–80. It matches `ns()`.
- **Not identified.** A piecewise polynomial of degree d needs d + 1 distinct ages in
  every region. In the second section the piecewise line stops at the last K where
  that holds (K = 8 for d = 3), instead of plotting arbitrary minimum-norm fits.
- **Numerics.** Models 1–5 use `numpy.linalg.lstsq` on truncated-power columns of the
  rescaled age u = (age − 41)/39 ∈ [−1, 1]. The polynomial curve in the second section
  uses Chebyshev columns (same fitted polynomials, stable up to degree 12).
- **Smoothing spline with repeated ages.** The objective is
  Σᵢ (yᵢ − f(xᵢ))² + λ ∫ f''(t)² dt with age in years. People with the same age share a
  fitted value, so it is solved on the distinct ages with weights (counts) and mean
  heights: (W + λK) g = W ȳ, K the natural cubic roughness matrix. This has the same
  solution as the individual-level problem. Effective df runs from 77 (λ = 0: the mean
  height at each age) to 2 (the least-squares line). λ slider positions correspond to
  effective df 2.05 to 70.

## Verify

```bash
uv run python verify_app.py
uv run marimo check app.py
```

`verify_app.py` runs the notebook and checks: the embedded data (300 and 3,607
people, 77 distinct ages); coefficient counts for every d = 0–3 and K = 1–6; identical
predictions of the two piecewise bases; smooth joins at interior knots; straight,
straight natural cubic tails with zero boundary curvature; agreement with R `bs()`, `ns()` (both to
1e-8) and `smooth.spline(all.knots = TRUE)` (1e-3 cm, the precision of R's reported
df); the smoothing limits (age means at λ = 0, the OLS line for large λ), that the
fitted values minimise the individual-level penalised objective, and that training
loss never rises for nested models.

## Browser export

```bash
bash export_wasm.sh
python -m http.server 8000 --directory build
```

Open [the local preview](http://localhost:8000). Rebuild `build/` after every change
to `app.py`; the folder is generated and ignored. **The first browser load requires
internet** for Marimo/Pyodide. [Marimo WASM export documentation](https://docs.marimo.io/guides/exporting/webassembly_html/).

Sources: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction
to Statistical Learning*, 2nd ed., Chapter 7.
[DOI: 10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1).
