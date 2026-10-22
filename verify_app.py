"""Check the models in app.py against their definitions and against R.

Run with: uv run python verify_app.py
"""

import unittest
from pathlib import Path

import numpy as np

from app import app

# Reference values computed in R 4.x with splines::bs(), splines::ns() and
# stats::smooth.spline(all.knots = TRUE) on the same 300 people, knots at the
# quartiles of age (11, 24, 49), Boundary.knots = c(2, 80).
R_GRID = np.array([0, 2, 5, 10, 20, 40, 60, 80, 90, 100.0])
R_BS3 = np.array(
    [85.4775991434, 94.131683863, 112.7211115591, 146.2738182241, 170.4213489948,
     168.0483882001, 167.6709632236, 160.4567619157, 153.1818320486, 142.9143052877]
)
R_NS3 = np.array(
    [75.410138101, 91.1125481052, 114.2951688896, 146.8870384322, 169.4712158561,
     168.695757374, 167.0468196034, 161.5343263739, 158.4349021775, 155.3354779812]
)
# smooth.spline(..., df = 8, all.knots = TRUE) reported df = 8.001182.
R_SS_DF = 8.001182
R_SS_AGES = np.array([2, 5, 10, 20, 40, 60, 80.0])
R_SS = np.array(
    [95.3569260621, 114.8391718036, 144.4547597403, 170.1630475411, 168.9081691903,
     167.96890204, 160.6685234257]
)


class FitChecks(unittest.TestCase):
    """Check the mathematical claims made by the plots and the table."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the notebook once and reuse its functions and data."""
        cls.outputs, cls.defs = app.run()
        d = cls.defs
        cls.x, cls.y = d["ages"], d["heights"]
        cls.design = staticmethod(d["design_matrix"])
        cls.fit = staticmethod(d["fit_models"])
        cls.knots = staticmethod(d["quantile_knots"])

    def test_data(self) -> None:
        """The embedded data are the lecture's 300 training and 3,607 test people."""
        self.assertEqual(len(self.x), 300)
        self.assertEqual(len(self.defs["test_ages"]), 3607)
        self.assertEqual(len(np.unique(self.x)), 77)
        self.assertEqual((self.x.min(), self.x.max()), (2.0, 80.0))
        source = Path(__file__).with_name("app.py").read_text()
        self.assertNotIn("read_csv", source)

    def test_all_cells_execute(self) -> None:
        """The default notebook produces six fits and finite flexibility curves."""
        self.assertEqual(len(self.defs["fits"]), 6)
        for rows in self.defs["curves"].values():
            self.assertTrue(np.isfinite(np.array(rows, dtype=float)).all())

    def test_counts_and_piecewise_equivalence(self) -> None:
        """Ranks match the stated counts; both piecewise bases give the same fit."""
        grid = np.linspace(0, 100, 301)
        for degree in range(4):
            for count in range(1, 7):
                knots = self.knots(self.x, count)
                self.assertEqual(len(knots), count)
                expected = [
                    degree + 1,
                    (count + 1) * (degree + 1),
                    (count + 1) * (degree + 1),
                    degree + count + 1,
                    count + 2,
                ]
                fits = self.fit(self.x, self.y, degree, knots)
                predictions = {}
                for kind, fit in fits.items():
                    if (kind == 3 and degree == 0) or (kind == 4 and degree != 3):
                        self.assertIsNone(fit)
                        continue
                    if not fit["identified"]:
                        self.fail(f"not identified: d={degree}, K={count}, kind={kind}")
                    self.assertEqual(fit["df"], expected[kind], (degree, count, kind))
                    predictions[kind] = self.design(grid, degree, knots, kind) @ fit["beta"]
                    self.assertTrue(np.isfinite(predictions[kind]).all())
                np.testing.assert_allclose(predictions[1], predictions[2], atol=1e-6)

    def test_adjustment_plot_spans_age_range(self) -> None:
        """Baseline and cumulative step adjustments extend to the displayed endpoint."""
        knots = self.knots(self.x, 2)
        fit = self.fit(self.x, self.y, 0, knots)[2]
        for tails, low, high in ((True, 0, 100), (False, 2, 80)):
            fig = self.defs["draw_fit"](
                2, fit, {"degree": 0, "knots": knots, "tails": tails}, (self.x, self.y)
            )
            ax = fig.axes[0]
            baseline = next(line for line in ax.lines if line.get_label() == "Baseline")
            np.testing.assert_allclose(baseline.get_xdata()[[0, -1]], [low, high])
            baseline_height = self.y[self.x < knots[0]].mean()
            np.testing.assert_allclose(baseline.get_ydata(), baseline_height)
            shading = [c for c in ax.collections if hasattr(c, "get_paths")
                       and c.__class__.__name__ == "FillBetweenPolyCollection"]
            self.assertEqual(len(shading), len(knots))
            for patch, left, right in zip(shading, knots, np.r_[knots[1:], high]):
                vertices = patch.get_paths()[0].vertices
                np.testing.assert_allclose(vertices[:, 0].min(), left)
                np.testing.assert_allclose(vertices[:, 0].max(), right)
                region_height = self.y[(self.x >= left) & (self.x < right)].mean()
                if right == 80:
                    region_height = self.y[self.x >= left].mean()
                np.testing.assert_allclose(
                    [vertices[:, 1].min(), vertices[:, 1].max()],
                    sorted([baseline_height, region_height]),
                )

    def test_spline_joins(self) -> None:
        """Values and the first d-1 derivatives agree at every interior knot."""
        knots = self.knots(self.x, 3)
        offsets = -np.linspace(0.1, 0.8, 8)
        for kind, degrees in ((3, (1, 2, 3)), (4, (3,))):
            for degree in degrees:
                for knot in knots:
                    left = self.design(knot + offsets, degree, knots, kind)
                    right = self.design(knot - offsets, degree, knots, kind)
                    a = np.polynomial.polynomial.polyfit(offsets, left, degree)
                    b = np.polynomial.polynomial.polyfit(-offsets, right, degree)
                    np.testing.assert_allclose(a[:degree], b[:degree], atol=1e-6)

    def test_natural_tails(self) -> None:
        """Natural cubic splines: straight outside 2-80, smooth join, zero boundary curvature."""
        knots = self.knots(self.x, 3)
        steps = np.linspace(0.2, 1.6, 8)
        for degree in (3,):
            m = (degree + 1) // 2
            beta = self.fit(self.x, self.y, degree, knots)[4]["beta"]
            f = lambda a: self.design(np.atleast_1d(a), degree, knots, 4) @ beta  # noqa: E731
            for edge, direction in ((2.0, 1.0), (80.0, -1.0)):
                inside = np.polynomial.polynomial.polyfit(
                    direction * steps, f(edge + direction * steps), degree)
                outside = np.polynomial.polynomial.polyfit(
                    -direction * steps, f(edge - direction * steps), degree)
                # Outside: a polynomial of degree m - 1, a straight line for d = 3.
                np.testing.assert_allclose(outside[m:], 0, atol=1e-6)
                # The join is as smooth as an interior knot: derivatives 0..d-1 agree,
                # so derivatives m..d-1 are zero just inside the boundary.
                np.testing.assert_allclose(inside[:degree], outside[:degree], atol=1e-5)

    def test_against_r(self) -> None:
        """bs(), ns() and smooth.spline() in R give the same curves."""
        knots = self.knots(self.x, 3)
        np.testing.assert_allclose(knots, [11, 24, 49])
        predict = self.defs["predict_model"]
        fits = self.fit(self.x, self.y, 3, knots)
        np.testing.assert_allclose(predict(R_GRID, 3, fits[3], 3, knots), R_BS3, atol=1e-8)
        np.testing.assert_allclose(predict(R_GRID, 4, fits[4], 3, knots), R_NS3, atol=1e-8)
        smooth = self.defs["smoother"](self.defs["lambda_for_df"](R_SS_DF))
        np.testing.assert_allclose(
            self.defs["predict_smoother"](R_SS_AGES, smooth), R_SS, atol=1e-3
        )

    def test_smoothing_with_repeated_ages(self) -> None:
        """Limits, edf range and the individual-level objective with repeated ages."""
        smoother, predict = self.defs["smoother"], self.defs["predict_smoother"]
        ages = np.unique(self.x)
        means = np.array([self.y[self.x == a].mean() for a in ages])
        wiggly = smoother(0.0)
        np.testing.assert_allclose(wiggly["values"], means, atol=1e-8)
        self.assertAlmostEqual(wiggly["df"], len(ages))
        straight = smoother(1e12)
        line = np.polynomial.Polynomial.fit(self.x, self.y, 1)(self.x)
        np.testing.assert_allclose(predict(self.x, straight), line, atol=1e-4)
        self.assertAlmostEqual(straight["df"], 2, places=4)
        lambdas = [0, 1e-2, 1, 1e2, 1e4, 1e6]
        fits = [smoother(lam) for lam in lambdas]
        self.assertTrue(np.all(np.diff([f["df"] for f in fits]) < 0))
        rss = [np.sum((predict(self.x, f) - self.y) ** 2) for f in fits]
        self.assertTrue(np.all(np.diff(rss) > 0))
        # Penalised objective on individuals: perturbing the fitted values never helps.
        fit = smoother(100.0)
        t, h = fit["t"], np.diff(fit["t"])

        def objective(values):
            n = len(t)
            q = np.zeros((n, n - 2))
            j = np.arange(n - 2)
            q[j, j], q[j + 1, j], q[j + 2, j] = 1 / h[:-1], -1 / h[:-1] - 1 / h[1:], 1 / h[1:]
            r = np.diag((h[:-1] + h[1:]) / 3) + np.diag(h[1:-1] / 6, 1) + np.diag(h[1:-1] / 6, -1)
            fitted = values[np.searchsorted(t, self.x)]
            return np.sum((self.y - fitted) ** 2) + 100.0 * values @ q @ np.linalg.solve(r, q.T @ values)

        best = objective(fit["values"])
        rng = np.random.default_rng(1)
        for _ in range(20):
            self.assertLess(best, objective(fit["values"] + rng.normal(0, 0.05, len(t))))

    def test_training_loss_nested(self) -> None:
        """Nested models (polynomial degrees, smaller lambda) never increase training loss."""
        curves = self.defs["curves"]
        train_poly = [r[2] for r in curves[0]]
        self.assertTrue(np.all(np.diff(train_poly) <= 1e-9))
        train_smooth = [r[2] for r in curves[5]]
        self.assertTrue(np.all(np.diff(train_smooth) <= 1e-9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
