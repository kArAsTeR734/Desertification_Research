import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image

import numpy as np
from matplotlib.figure import Figure

from app import (
    APP_VERSION,
    PARAMETER_SNAPSHOT_SCHEMA_VERSION,
    DESERT_SPREAD_START,
    DESERT_SPREAD_FULL,
    MOISTURE_SCALE_LOCAL,
    MOISTURE_SCALE_COMMON,
    RAIN_MAP_FIXED,
    RAIN_MAP_MOVING,
    App,
    SCENARIO_OVERRIDES,
    DesertificationModel,
)


def parameters(**overrides):
    p = dict(
        nx=5, ny=5, Lx=10.0, Ly=10.0, dt=0.05,
        Dv=0.05, Dw=0.10, r=0.50, Kw=0.30, Ks=0.30,
        Kd=0.30, m=0.15, P0=0.30, e=0.10, c=0.15,
        rs=0.05, ds=0.08, V0=0.70, W0=0.60, S0=0.80,
        Dcrit=0.70,
        spot_x=5.0, spot_y=5.0, spot_sigma=1.5,
        spot_count=1, spot_strength_spread=0.0,
        spot_size_spread=0.0, spot_min_separation=0.9,
        spot_V_drop=0.25, spot_W_drop=0.25, spot_S_drop=0.20,
        noise=0.0, picard_tol=1e-8, max_picard=50, seed=42,
    )
    p.update(overrides)
    return p


class DesertificationModelTests(unittest.TestCase):
    def landscape_parameters(self, **changes):
        return parameters(**(SCENARIO_OVERRIDES["Случайные очаги"] | {
            "nx": 25, "ny": 25, "spot_count": 5, "seed": 3,
            "noise": 0.0,
        } | changes))

    def make_app(self, model):
        class Value:
            def get(self):
                return "Случайные очаги"
        app = App.__new__(App)
        app.model, app.vars = model, {}
        app.default_values = parameters()
        app.scenario_var = Value()
        return app

    def test_landscape_is_fixed_reproducible_and_seed_sensitive(self):
        model = DesertificationModel(self.landscape_parameters())
        other = DesertificationModel(dict(model.p))
        fields = {key: getattr(model, key).copy()
                  for key in ("rain_factor", "growth_factor", "soil_factor")}
        for key, field in fields.items():
            np.testing.assert_array_equal(field, getattr(other, key))
            self.assertGreater(np.std(field), 0.01)
            self.assertGreater(np.min(field), 0.0)
            self.assertAlmostEqual(np.mean(field), 1.0)
        for _ in range(20):
            model.step()
        for key, field in fields.items():
            np.testing.assert_array_equal(field, getattr(model, key))
        model.reset()
        for key, field in fields.items():
            np.testing.assert_array_equal(field, getattr(model, key))
        changed = DesertificationModel(model.p | {"seed": 4})
        self.assertFalse(np.array_equal(model.rain_factor, changed.rain_factor))

    def test_moving_rain_is_reproducible_and_independent_of_query_order(self):
        p = self.landscape_parameters(rain_map_mode=RAIN_MAP_MOVING)
        first = DesertificationModel(p)
        second = DesertificationModel(p)
        expected = {t: first.get_precipitation_field(t).copy() for t in (0.0, 1.0, 5.0, 30.0)}
        for t in (30.0, 0.0, 5.0, 1.0):
            np.testing.assert_array_equal(second.get_precipitation_field(t), expected[t])
            np.testing.assert_array_equal(first.get_precipitation_field(t), expected[t])
        first.step()
        first.reset()
        np.testing.assert_array_equal(first.get_precipitation_field(5.0), expected[5.0])
        changed = DesertificationModel(p | {"seed": 4})
        self.assertFalse(np.array_equal(changed.get_precipitation_field(0), expected[0.0]))

    def test_moving_rain_is_smooth_in_time_and_moves_with_the_selected_velocity(self):
        model = DesertificationModel(self.landscape_parameters(
            nx=21, ny=21, rain_spatial_amplitude=0.0,
            rain_map_mode=RAIN_MAP_MOVING, rain_motion_speed=0.5, rain_motion_angle=0.0,
        ))
        initial = model.get_rainfall_multiplier(0).copy()
        later = model.get_rainfall_multiplier(1).copy()
        # dx=0.5: after t=1 the pattern is translated by one cell to the right.
        # Crop and normalize to remove the instantaneous whole-domain normalization.
        overlap_initial, overlap_later = initial[:, :-1], later[:, 1:]
        np.testing.assert_allclose(overlap_initial / np.mean(overlap_initial),
                                   overlap_later / np.mean(overlap_later), atol=1e-13)
        self.assertGreater(np.max(np.abs(later - initial)), 0.1)
        nearby = model.get_rainfall_multiplier(1 + 1e-6)
        self.assertLess(np.max(np.abs(nearby - later)), 1e-4)

    def test_moving_rain_preserves_mean_in_every_climate_mode(self):
        for mode in ("Постоянные", "Сезонные", "Линейное уменьшение", "Линейное увеличение"):
            model = DesertificationModel(self.landscape_parameters(
                rain_map_mode=RAIN_MAP_MOVING, precipitation_mode=mode,
                P0=0.03, A=0.03, Tseason=10.0,
            ))
            for t in (0, 2.5, 7.5, 30, 100):
                rain = model.get_precipitation_field(t)
                self.assertTrue(np.all(np.isfinite(rain)))
                self.assertGreaterEqual(np.min(rain), 0)
                self.assertAlmostEqual(np.mean(rain), model.get_precipitation(t), places=14)
            if mode == "Сезонные":
                np.testing.assert_array_equal(model.get_precipitation_field(7.5), np.zeros_like(model.W))

    def test_moving_weather_preserves_initial_landscape_and_zero_contrast_preserves_old_dynamics(self):
        p = self.landscape_parameters(rain_map_mode=RAIN_MAP_FIXED)
        fixed = DesertificationModel(p)
        moving = DesertificationModel(p | {"rain_map_mode": RAIN_MAP_MOVING})
        disabled = DesertificationModel(p | {"rain_map_mode": RAIN_MAP_MOVING,
                                             "rain_motion_strength": 0.0})
        for name in ("V", "W", "S", "rain_factor", "growth_factor", "soil_factor"):
            np.testing.assert_array_equal(getattr(fixed, name), getattr(moving, name))
        for _ in range(40):
            fixed.step()
            disabled.step()
        for name in ("V", "W", "S", "desertification_index_history", "precipitation_min_history",
                     "precipitation_max_history"):
            np.testing.assert_array_equal(getattr(fixed, name), getattr(disabled, name))

    def test_zero_speed_freezes_weather_and_zero_rain_has_no_local_precipitation(self):
        model = DesertificationModel(self.landscape_parameters(
            rain_map_mode=RAIN_MAP_MOVING, rain_motion_speed=0.0,
        ))
        np.testing.assert_array_equal(model.get_precipitation_field(0), model.get_precipitation_field(100))
        dry = DesertificationModel(model.p | {"P0": 0.0, "rain_motion_speed": 0.5})
        for t in (0, 10):
            np.testing.assert_array_equal(dry.get_precipitation_field(t), np.zeros_like(dry.W))

    def test_moving_local_rain_enters_the_water_equation_and_actual_extrema_history(self):
        model = DesertificationModel(self.landscape_parameters(
            rain_map_mode=RAIN_MAP_MOVING, Dv=0.0, Dw=0.0,
            r=0.0, m=0.0, e=0.0, c=0.0, rs=0.0, ds=0.0,
        ))
        expected = model.W.copy()
        for _ in range(30):
            rain = model.get_precipitation_field(model.t + model.p["dt"])
            expected += model.p["dt"] * rain
            model.step()
            self.assertAlmostEqual(model.precipitation_min_history[-1], np.min(rain))
            self.assertAlmostEqual(model.precipitation_max_history[-1], np.max(rain))
        np.testing.assert_allclose(model.W, expected, atol=1e-14)
        self.assertEqual(len(model.time_history), len(model.precipitation_min_history))

    def test_rain_map_displays_actual_precipitation_and_records_weather_configuration(self):
        model = DesertificationModel(self.landscape_parameters(rain_map_mode=RAIN_MAP_MOVING))
        model.step()
        app = self.make_app(model)
        spec = next(spec for spec in app._map_specs() if spec[0] == "Осадки на территории")
        np.testing.assert_array_equal(spec[3], model.get_precipitation_field(model.t))
        self.assertGreaterEqual(spec[6], np.max(spec[3]))
        saved = json.loads(json.dumps(app._parameter_snapshot()))
        self.assertEqual(saved["applied_run"]["rain_motion"], model.rain_motion_summary)
        self.assertEqual(saved["applied_run"]["rainfall_map"]["quantity"], "P(x,y,t)")

    def test_invalid_moving_rain_parameters_are_rejected(self):
        for key, values in {
            "rain_map_mode": ("bad",), "rain_motion_strength": (-0.1, 1.1, np.nan),
            "rain_motion_speed": (-0.1, np.inf), "rain_motion_length": (0, 11, np.nan),
            "rain_motion_angle": (-1, 361, np.inf),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    DesertificationModel(self.landscape_parameters(**{key: value}))

    def test_rainfall_is_nonnegative_mean_preserving_and_seasonal(self):
        for mode in ("Постоянные", "Линейное уменьшение", "Линейное увеличение", "Сезонные"):
            model = DesertificationModel(self.landscape_parameters(
                precipitation_mode=mode, P0=0.02, A=0.08, Tseason=10.0,
            ))
            for time in (0, 2.5, 7.5, 20, 100):
                field = model.get_precipitation_field(time)
                self.assertGreaterEqual(np.min(field), 0.0)
                self.assertAlmostEqual(np.mean(field), model.get_precipitation(time), places=14)
                np.testing.assert_allclose(field, model.get_precipitation(time) * model.rain_factor)

    def test_local_rainfall_enters_water_equation_every_step(self):
        model = DesertificationModel(self.landscape_parameters(
            Dv=0.0, Dw=0.0, r=0.0, m=0.0, e=0.0, c=0.0, rs=0.0, ds=0.0,
            noise=0.0, P0=0.1,
        ))
        initial = model.W.copy()
        for _ in range(5):
            model.step()
        np.testing.assert_allclose(model.W, initial + model.t * 0.1 * model.rain_factor,
                                   atol=1e-14)

    def test_initial_background_and_persistent_landscape_share_patterns(self):
        model = DesertificationModel(self.landscape_parameters(
            spot_V_drop=0.0, spot_W_drop=0.0, spot_S_drop=0.0,
        ))
        for state, factor in ((model.V, model.growth_factor),
                              (model.W, model.rain_factor), (model.S, model.soil_factor)):
            self.assertGreater(np.corrcoef(state.ravel(), factor.ravel())[0, 1], 0.99)

    def test_heterogeneity_can_be_enabled_without_initial_background(self):
        p = self.landscape_parameters(V_bg_amplitude=0.0, W_bg_amplitude=0.0, S_bg_amplitude=0.0)
        local = DesertificationModel(p)
        uniform = DesertificationModel(p | {
            "rain_spatial_amplitude": 0.0, "growth_spatial_amplitude": 0.0,
            "soil_spatial_amplitude": 0.0,
        })
        np.testing.assert_array_equal(local.V, uniform.V)
        np.testing.assert_array_equal(local.W, uniform.W)
        self.assertGreater(np.std(local.rain_factor), 0.01)
        np.testing.assert_array_equal(uniform.rain_factor, np.ones_like(uniform.W))

    def test_dry_weak_regions_degrade_faster_and_maps_remain_heterogeneous(self):
        model = DesertificationModel(self.landscape_parameters())
        uniform = DesertificationModel(model.p | {
            "rain_spatial_amplitude": 0.0, "growth_spatial_amplitude": 0.0,
            "soil_spatial_amplitude": 0.0,
        })
        dry = model.rain_factor <= np.quantile(model.rain_factor, 0.25)
        wet = model.rain_factor >= np.quantile(model.rain_factor, 0.75)
        for _ in range(1200):
            model.step()
            uniform.step()
        index = model.desertification_index()
        self.assertGreater(np.mean(index[dry]) - np.mean(index[wet]), 0.10)
        self.assertLess(np.mean(model.V[dry]), np.mean(model.V[wet]))
        self.assertLess(np.mean(model.W[dry]), np.mean(model.W[wet]))
        for name in ("V", "W", "S"):
            self.assertGreater(np.std(getattr(model, name)), 3 * np.std(getattr(uniform, name)))

    def test_retired_threshold_parameters_cannot_change_dynamics(self):
        first = DesertificationModel(self.landscape_parameters(Dcrit=0.1, coverage_mode="Бинарная площадь"))
        second = DesertificationModel(self.landscape_parameters(Dcrit=0.9, coverage_mode="Плавная оценка охвата"))
        for _ in range(20):
            first.step()
            second.step()
        for name in ("V", "W", "S", "desertification_index_history"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertNotIn("Dcrit", first.p)
        self.assertNotIn("coverage_mode", first.p)
        np.testing.assert_array_equal(first.desert_spread_distribution(),
                                      second.desert_spread_distribution())

    def test_desert_spread_is_continuous_local_and_reversible(self):
        model = DesertificationModel(self.landscape_parameters(nx=10, ny=10))
        index = np.linspace(0, 1, model.N).reshape(model.X.shape)
        model.V = model.W = model.S = 1 - index
        spread = model.desert_spread_distribution()
        self.assertTrue(np.issubdtype(spread.dtype, np.floating))
        self.assertTrue(np.all(spread[index <= DESERT_SPREAD_START] == 0))
        self.assertTrue(np.all(spread[index >= DESERT_SPREAD_FULL] == 1))
        self.assertGreater(np.count_nonzero((spread > 0) & (spread < 1)), 30)
        self.assertTrue(np.all(np.diff(spread.ravel()) >= 0))
        # Changing distant cells cannot reposition or reweight this cell.
        cell = (6, 0)
        expected = spread[cell]
        for field in (model.V, model.W, model.S):
            field[:3] = 0
        self.assertAlmostEqual(model.desert_spread_distribution()[cell], expected)
        model.V = model.W = model.S = 1 - index
        np.testing.assert_array_equal(model.desert_spread_distribution(), spread)

    def test_landscape_amplitude_validation(self):
        for key in ("rain_spatial_amplitude", "growth_spatial_amplitude", "soil_spatial_amplitude"):
            for value in (-0.1, 0.96, np.nan, np.inf):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    DesertificationModel(self.landscape_parameters(**{key: value}))
        for value in (0.0, 0.95):
            DesertificationModel(self.landscape_parameters(rain_spatial_amplitude=value))

    def test_continuous_histories_and_reset_are_consistent(self):
        model = DesertificationModel(self.landscape_parameters())
        for _ in range(5):
            model.step()
        for name in ("desertification_index_history", "index_p10_history", "index_p90_history",
                     "vegetation_history", "moisture_history", "soil_history", "precipitation_history"):
            self.assertEqual(len(getattr(model, name)), len(model.time_history))
        np.testing.assert_allclose([model.index_p10_history[-1], model.index_p90_history[-1]],
                                   np.quantile(model.desertification_index(), (0.1, 0.9)))
        self.assertAlmostEqual(model.vegetation_history[-1], np.mean(model.V))
        model.reset()
        self.assertEqual(len(model.time_history), 1)
        self.assertEqual(model.time_history[0], 0.0)

    def test_export_has_separate_full_size_figures_and_reproducible_data(self):
        model = DesertificationModel(self.landscape_parameters(nx=10, ny=10, rain_map_mode=RAIN_MAP_MOVING))
        model.step()
        app = self.make_app(model)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            paths = app.export_results(output)
            self.assertEqual(len(paths), 21)
            self.assertEqual(len(list(output.glob("*.png"))), 9)
            self.assertEqual(len(list(output.glob("*.svg"))), 9)
            for path in output.glob("*.png"):
                with Image.open(path) as image:
                    is_map = int(path.name[:2]) <= 5 or path.name.startswith("09_")
                    expected = (3000, 2400) if is_map else (3840, 2160)
                    self.assertEqual(image.size, expected)
                    self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.1)
            with np.load(output / "fields.npz") as data:
                np.testing.assert_array_equal(data["D"], model.desertification_index())
                np.testing.assert_array_equal(data["rain_factor"], model.rain_factor)
                np.testing.assert_array_equal(data["desert_spread"], model.desert_spread_distribution())
                np.testing.assert_array_equal(data["rainfall_multiplier"], model.get_rainfall_multiplier(model.t))
                np.testing.assert_array_equal(data["precipitation"], model.get_precipitation_field(model.t))
            history = np.loadtxt(output / "history.csv", delimiter=",", skiprows=1)
            np.testing.assert_allclose(history[:, 1], model.desertification_index_history)
            saved = json.loads((output / "parameters.json").read_text(encoding="utf-8"))
            recreated = DesertificationModel(saved["applied_run"]["parameters"])
            for _ in range(saved["applied_run"]["completed_steps"]):
                recreated.step()
            np.testing.assert_array_equal(recreated.V, model.V)
            np.testing.assert_array_equal(recreated.W, model.W)
            self.assertNotIn("Dcrit", saved["applied_run"]["parameters"])
            self.assertEqual(saved["visualization"]["moisture_color_scale"],
                             app._moisture_color_scale())
            with self.assertRaises(FileExistsError):
                app.export_results(output)

    def test_moisture_scale_reveals_physical_contrast_without_changing_fields(self):
        model = DesertificationModel(self.landscape_parameters(nx=10, ny=10))
        model.W = np.linspace(0.137, 0.272, model.N).reshape(model.W.shape)
        original = model.W.copy()
        app = self.make_app(model)
        scale = app._moisture_color_scale()
        self.assertEqual(scale["mode"], MOISTURE_SCALE_LOCAL)
        self.assertLessEqual(scale["vmin"], np.min(model.W))
        self.assertGreaterEqual(scale["vmax"], np.max(model.W))
        self.assertGreater(np.ptp(model.W) / (scale["vmax"] - scale["vmin"]), 0.8)
        spec = next(spec for spec in app._map_specs() if spec[0] == "Влажность W")
        self.assertIs(spec[3], model.W)
        self.assertEqual((spec[5], spec[6]), (scale["vmin"], scale["vmax"]))
        self.assertIn("0.137", spec[2])
        self.assertIn("0.272", spec[2])
        app.moisture_scale_var = SimpleNamespace(get=lambda: MOISTURE_SCALE_COMMON)
        common = app._moisture_color_scale()
        self.assertEqual((common["vmin"], common["vmax"]), (0.0, 1.0))
        np.testing.assert_array_equal(model.W, original)

    def test_moisture_scale_handles_uniform_zero_and_high_water_without_false_contrast(self):
        model = DesertificationModel(self.landscape_parameters(nx=10, ny=10))
        app = self.make_app(model)
        for value in (0.0, 0.0001, 0.2, 1.0, 2.5):
            with self.subTest(value=value):
                model.W.fill(value)
                scale = app._moisture_color_scale()
                self.assertGreaterEqual(scale["vmin"], 0)
                self.assertLessEqual(scale["vmin"], value)
                self.assertGreaterEqual(scale["vmax"], value)
                self.assertGreaterEqual(scale["vmax"] - scale["vmin"], 0.02 - 1e-12)
        model.W.fill(0.2)
        model.W[0, 0] += 1e-8
        scale = app._moisture_color_scale()
        self.assertLess(np.ptp(model.W) / (scale["vmax"] - scale["vmin"]), 1e-5)
        model.W.fill(2.5)
        app.moisture_scale_var = SimpleNamespace(get=lambda: MOISTURE_SCALE_COMMON)
        self.assertEqual(app._moisture_color_scale()["vmax"], 2.5)

    def test_series_have_clean_index_legend_and_unbounded_moisture(self):
        model = DesertificationModel(self.landscape_parameters(P0=0.45))
        for _ in range(400):
            model.step()
        app = self.make_app(model)
        for name in ("Динамика индекса", "Динамика V, W, S", "Осадки P(t)"):
            figure = Figure()
            axis = figure.add_subplot(111)
            app._render_series(axis, name)
            self.assertEqual(axis.get_xlim()[0], 0.0)
            self.assertAlmostEqual(axis.get_xlim()[1], model.t)
            if name == "Динамика V, W, S":
                self.assertEqual(len(axis.lines), 3)
                self.assertGreater(axis.get_ylim()[1], max(model.moisture_history))
            elif name == "Осадки P(t)":
                self.assertEqual(len(axis.collections), 1)
            else:
                self.assertEqual(len(axis.collections), 0)
                self.assertEqual(len(axis.lines), 1)
                self.assertEqual([text.get_text() for text in axis.get_legend().get_texts()],
                                 ["Средний индекс D"])
            figure.clear()

    def test_single_spot_initial_state_matches_legacy_formula(self):
        p = parameters(nx=25, ny=25, noise=0.01, seed=42)
        model = DesertificationModel(p)
        spot = np.exp(-(
            (model.X - p["spot_x"]) ** 2
            + (model.Y - p["spot_y"]) ** 2
        ) / (2 * p["spot_sigma"] ** 2))
        rng = np.random.default_rng(p["seed"])
        expected_v = np.clip(
            p["V0"] - p["spot_V_drop"] * spot
            + p["noise"] * rng.normal(size=model.X.shape), 0, 1,
        )
        expected_w = np.maximum(
            p["W0"] - p["spot_W_drop"] * spot
            + p["noise"] * rng.normal(size=model.X.shape), 0,
        )
        expected_s = np.clip(
            p["S0"] - p["spot_S_drop"] * spot
            + p["noise"] * rng.normal(size=model.X.shape), 0, 1,
        )
        np.testing.assert_array_equal(model.V, expected_v)
        np.testing.assert_array_equal(model.W, expected_w)
        np.testing.assert_array_equal(model.S, expected_s)
        self.assertEqual(len(model.spots), 1)

    def test_multi_spots_are_reproducible_separated_and_nonadditive(self):
        p = parameters(
            nx=25, ny=25, noise=0.0, seed=3,
            spot_count=5, **(SCENARIO_OVERRIDES["Случайные очаги"] | {
                "V_bg_amplitude": 0.0, "W_bg_amplitude": 0.0,
                "S_bg_amplitude": 0.0,
            }),
        )
        model = DesertificationModel(p)
        initial_spots = [dict(item) for item in model.spots]
        initial_v = model.V.copy()
        initial_w = model.W.copy()
        initial_s = model.S.copy()
        for first_index, first in enumerate(initial_spots):
            for second in initial_spots[first_index + 1:]:
                distance = np.hypot(
                    first["x"] - second["x"], first["y"] - second["y"],
                )
                self.assertGreaterEqual(
                    distance + 1e-12,
                    p["spot_min_separation"] * (first["sigma"] + second["sigma"]),
                )
        self.assertGreaterEqual(
            np.min(model.V), p["V0"] - max(s["V_drop_peak"] for s in model.spots),
        )
        self.assertGreaterEqual(
            np.min(model.S), p["S0"] - max(s["S_drop_peak"] for s in model.spots),
        )
        model.step()
        model.reset()
        self.assertEqual(model.spots, initial_spots)
        np.testing.assert_array_equal(model.V, initial_v)
        np.testing.assert_array_equal(model.W, initial_w)
        np.testing.assert_array_equal(model.S, initial_s)

        other = DesertificationModel(p | {"seed": 4})
        self.assertNotEqual(other.spots, initial_spots)

    def test_multi_spot_invalid_settings_are_rejected(self):
        multi = SCENARIO_OVERRIDES["Случайные очаги"] | {"spot_count": 5}
        for changed in (
            {"spot_count": 0},
            {"spot_count": 21},
            {"spot_strength_spread": 0.81},
            {"spot_size_spread": 0.76},
            {"spot_min_separation": 0.4},
            {"spot_sigma": 6.0},
            {"spot_V_drop": 0.9},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                DesertificationModel(parameters(**(multi | changed)))

    def test_multi_spot_centers_cross_cutoff_at_different_times(self):
        model = DesertificationModel(parameters(
            nx=25, ny=25, noise=0.01, seed=3,
            spot_count=5, **(SCENARIO_OVERRIDES["Случайные очаги"] | {
                "spot_sigma": 1.1,
                "V_bg_amplitude": 0.0, "W_bg_amplitude": 0.0,
                "S_bg_amplitude": 0.0,
            }),
        ))
        local_times = {spot["id"]: None for spot in model.spots}
        center_cells = {
            spot["id"]: (
                int(np.argmin(abs(model.y - spot["y"]))),
                int(np.argmin(abs(model.x - spot["x"]))),
            ) for spot in model.spots
        }
        for _ in range(1200):
            model.step()
            index = model.desertification_index()
            for spot_id, (row, column) in center_cells.items():
                if local_times[spot_id] is None and index[row, column] >= 0.70:
                    local_times[spot_id] = model.t
        self.assertTrue(all(time is not None for time in local_times.values()))
        self.assertGreater(max(local_times.values()) - min(local_times.values()), 5)

    def test_moisture_rebound_does_not_hide_severe_degradation(self):
        model = DesertificationModel(parameters(
            nx=25, ny=25, P0=0.03, r=0.25, m=0.30, e=0.14, ds=0.13,
            spot_V_drop=0.30, spot_W_drop=0.35, spot_S_drop=0.30,
            spot_sigma=2.0, noise=0.01,
        ))
        old_areas = []
        for _ in range(600):
            model.step()
            old_areas.append(float(np.mean(
                (model.V < 0.20) & (model.W < 0.20) & (model.S < 0.30)
            )))

        self.assertGreater(max(old_areas), 0.9)
        self.assertEqual(old_areas[-1], 0.0)
        self.assertGreater(model.mean_desertification_index(), 0.8)
        self.assertGreater(np.mean(model.W), 0.20)

    def test_parameter_snapshot_distinguishes_displayed_and_applied(self):
        class Value:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        model = DesertificationModel(parameters(
            P0=0.03, precipitation_mode="Постоянные",
        ))
        model.step()
        app = App.__new__(App)
        app.model = model
        app.default_values = parameters(
            precipitation_mode="Постоянные", steps=200,
        )
        app.vars = {
            "P0": Value("0.08"),
            "precipitation_mode": Value("Сезонные"),
            "steps": Value("600"),
        }
        app.scenario_var = Value("Случайные очаги")
        snapshot = app._parameter_snapshot()

        self.assertEqual(snapshot["app_version"], APP_VERSION)
        self.assertEqual(snapshot["displayed_in_ui"]["fields"]["P0"], "0.08")
        self.assertEqual(snapshot["applied_run"]["parameters"]["P0"], 0.03)
        self.assertEqual(snapshot["applied_run"]["completed_steps"], 1)
        self.assertEqual(
            snapshot["applied_run"]["matched_scenario"],
            "Пользовательские параметры",
        )
        self.assertEqual(
            {item["field"] for item in snapshot["differences"]["displayed_fields_not_applied"]},
            {"P0", "precipitation_mode"},
        )
        self.assertFalse(snapshot["differences"]["scenario_selection_not_applied"])
        self.assertTrue(snapshot["differences"]["has_unapplied_changes"])
        self.assertEqual(
            json.loads(json.dumps(snapshot))["schema_version"],
            PARAMETER_SNAPSHOT_SCHEMA_VERSION,
        )
        run = snapshot["applied_run"]
        self.assertEqual(run["persistent_landscape"], model.landscape_summary)
        self.assertEqual(run["displayed_map"]["algorithm"], "continuous_weighted_index_v1")
        self.assertNotIn("desert_area_percent_at_t", run)
        self.assertNotIn("Dcrit", run["parameters"])

    def test_parameter_snapshot_records_spots_and_unapplied_seed(self):
        class Value:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        model = DesertificationModel(parameters(
            nx=25, ny=25, seed=3,
            spot_count=5, **SCENARIO_OVERRIDES["Случайные очаги"],
        ))
        app = App.__new__(App)
        app.model = model
        app.default_values = parameters(steps=200)
        app.vars = {"seed": Value("4"), "steps": Value("1200")}
        app.scenario_var = Value("Случайные очаги")

        snapshot = json.loads(json.dumps(app._parameter_snapshot()))
        self.assertEqual(snapshot["applied_run"]["parameters"]["seed"], 3)
        self.assertEqual(snapshot["displayed_in_ui"]["fields"]["seed"], "4")
        self.assertEqual(snapshot["applied_run"]["initial_spots"], model.spots)
        self.assertEqual(len(snapshot["applied_run"]["initial_spots"]), 5)
        self.assertEqual(
            [item["field"] for item in snapshot["differences"]["displayed_fields_not_applied"]],
            ["seed"],
        )
        self.assertTrue(snapshot["differences"]["has_unapplied_changes"])

    def test_increased_losses_change_continuous_dynamics(self):
        base = DesertificationModel(parameters(
            nx=25, ny=25,
            P0=0.03, seed=42, steps=800, precipitation_mode="Постоянные",
        ))
        demo = DesertificationModel(parameters(
            nx=25, ny=25,
            P0=0.03, seed=42, m=0.18, ds=0.10, e=0.11,
            spot_sigma=2.5, spot_V_drop=0.35, spot_W_drop=0.25,
            spot_S_drop=0.35, steps=800, precipitation_mode="Постоянные",
        ))
        for _ in range(541):
            base.step()
            demo.step()

        self.assertEqual(base.p["P0"], demo.p["P0"])
        self.assertLess(np.mean(demo.V), np.mean(base.V))
        self.assertLess(np.mean(demo.S), np.mean(base.S))
        self.assertGreater(
            demo.mean_desertification_index(),
            base.mean_desertification_index(),
        )

    def test_drought_reaction_stays_negative_above_one(self):
        model = DesertificationModel(parameters(r=0.0, rs=0.0))
        V = np.array([[0.5]])
        W = np.array([[2.0]])
        S = np.array([[0.5]])
        RV, _, RS = model._reaction_terms(V, W, S, 0.0)
        drought = 0.30 / (2.0 + 0.30)
        self.assertAlmostEqual(RV[0, 0], -0.15 * 0.5 * drought)
        self.assertAlmostEqual(RS[0, 0], -0.08 * 0.5 * drought * 0.5)
        self.assertLess(RV[0, 0], 0.0)
        self.assertLess(RS[0, 0], 0.0)

    def test_wet_scenario_allows_water_above_one_and_bounded_index(self):
        model = DesertificationModel(parameters(P0=0.45))
        for _ in range(400):
            model.step()
        self.assertGreater(np.min(model.W), 1.0)
        self.assertTrue(np.all(model.desertification_index() >= 0.0))
        self.assertTrue(np.all(model.desertification_index() <= 1.0))
        self.assertLess(model.last_iterations, model.p["max_picard"])

    def test_dry_scenario_desertifies_more_than_wet(self):
        dry = DesertificationModel(parameters(P0=0.01))
        wet = DesertificationModel(parameters(P0=0.30))
        for _ in range(400):
            dry.step()
            wet.step()
        self.assertGreater(
            dry.mean_desertification_index(),
            wet.mean_desertification_index(),
        )


if __name__ == "__main__":
    unittest.main()
