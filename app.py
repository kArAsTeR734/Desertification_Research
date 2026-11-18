import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import secrets
import queue
import threading
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import factorized

APP_VERSION = "1.10.0"
PARAMETER_SNAPSHOT_SCHEMA_VERSION = 10
BACKGROUND_RNG_STREAM = 6203
RAIN_MOTION_RNG_STREAM = 104729
RAIN_MAP_FIXED = "Неподвижное распределение"
RAIN_MAP_MOVING = "Движущиеся дождевые области"
MOISTURE_SCALE_LOCAL = "Текущий диапазон"
MOISTURE_SCALE_COMMON = "Общая: от 0 до ≥1"
DESERT_SPREAD_START = 0.40
DESERT_SPREAD_FULL = 0.90
DESERT_SPREAD_CMAP = LinearSegmentedColormap.from_list(
    "desert_spread", ["#3b844e", "#b8cf83", "#e8d7a0", "#d49852", "#9b602f"],
)
SPOTS_CENTRAL = "Центральный (1)"
SPOTS_RANDOM_COUNT = "Случайное число 1–8"
SPOTS_FIXED_COUNT = "Задано число (случайно)"
RANDOM_SPOT_SCENARIOS = {"Случайные очаги"}

SCENARIO_OVERRIDES = {
    "Случайные очаги": {
        "P0": 0.03,
        "spot_mode": SPOTS_RANDOM_COUNT,
        "V_bg_amplitude": 0.15,
        "W_bg_amplitude": 0.20,
        "S_bg_amplitude": 0.15,
        "bg_length_scale": 2.0,
        "rain_spatial_amplitude": 0.80,
        "growth_spatial_amplitude": 0.50,
        "soil_spatial_amplitude": 0.40,
        "spot_sigma": 0.65,
        "spot_V_drop": 0.30,
        "spot_W_drop": 0.30,
        "spot_S_drop": 0.30,
        "spot_strength_spread": 0.30,
        "spot_size_spread": 0.30,
        "spot_min_separation": 0.90,
        "steps": 1200,
        "precipitation_mode": "Постоянные",
    },
}


def _same_parameter_value(left, right):
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_number)
        and math.isfinite(right_number)
        and math.isclose(left_number, right_number, rel_tol=1e-12, abs_tol=1e-12)
    )


def identify_applied_scenario(applied_parameters, default_values):
    """Name an exact preset match; otherwise describe the run as custom."""
    for name, overrides in SCENARIO_OVERRIDES.items():
        expected = default_values | overrides
        if all(
            _same_parameter_value(value, applied_parameters[key])
            for key, value in expected.items()
            if key in applied_parameters
            and not (name in RANDOM_SPOT_SCENARIOS and key in {"seed", "spot_count"})
        ):
            return name
    return "Пользовательские параметры"


def displayed_parameter_differences(displayed_fields, applied_parameters):
    """List visible values that would not reproduce the active model."""
    differences = []
    for key, applied_value in applied_parameters.items():
        if key not in displayed_fields:
            continue
        displayed_value = displayed_fields[key]
        if not _same_parameter_value(displayed_value, applied_value):
            differences.append({
                "field": key,
                "displayed": displayed_value,
                "applied": applied_value,
            })
    return differences


class DesertificationModel:
    """2-D model of vegetation, soil moisture and soil state.

    Time discretisation:
      - diffusion: implicit;
      - nonlinear reaction terms: Picard-linearised;
      - S has no diffusion, so its local nonlinear equation is
        handled by the same Picard iteration.
    """

    def __init__(self, p):
        p = dict(p)
        # Old parameter files reproduce homogeneous kinetics unless the new
        # landscape amplitudes are explicitly enabled. Thresholds are retired.
        for key in ("Dcrit", "coverage_mode", "coverage_width"):
            p.pop(key, None)
        for key in ("rain_spatial_amplitude", "growth_spatial_amplitude",
                    "soil_spatial_amplitude"):
            p.setdefault(key, 0.0)
        p.setdefault("rain_map_mode", RAIN_MAP_FIXED)
        p.setdefault("rain_motion_strength", 0.75)
        p.setdefault("rain_motion_speed", 0.50)
        p.setdefault("rain_motion_length", min(1.0, p["Lx"], p["Ly"]))
        p.setdefault("rain_motion_angle", 30.0)
        self._validate_spot_settings(p)
        self._validate_background_settings(p)
        self._validate_rain_motion_settings(p)
        self.p = p
        self.nx = p["nx"]
        self.ny = p["ny"]
        self.Lx = p["Lx"]
        self.Ly = p["Ly"]
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.N = self.nx * self.ny

        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.L = self._neumann_laplacian()
        self._build_solvers()
        self.reset()

    @staticmethod
    def _validate_spot_settings(p):
        count = p.get("spot_count", 1)
        mode = p.get(
            "spot_mode", SPOTS_CENTRAL if count == 1 else SPOTS_FIXED_COUNT,
        )
        seed = p["seed"]
        if not isinstance(count, (int, np.integer)) or not 1 <= count <= 20:
            raise ValueError("Число очагов должно быть целым от 1 до 20.")
        if mode not in (SPOTS_CENTRAL, SPOTS_RANDOM_COUNT, SPOTS_FIXED_COUNT):
            raise ValueError("Выберите корректный режим начальных очагов.")
        if mode == SPOTS_CENTRAL and count != 1:
            raise ValueError("Для центрального очага число очагов должно быть 1.")
        if mode == SPOTS_RANDOM_COUNT and count > 8:
            raise ValueError("В режиме случайного числа допускается от 1 до 8 очагов.")
        if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
            raise ValueError("Seed должен быть целым от 0 до 4294967295.")
        if p["Lx"] <= 0 or p["Ly"] <= 0:
            raise ValueError("Lx и Ly должны быть положительными.")
        if p["spot_sigma"] <= 0:
            raise ValueError("Размер очага σ должен быть положительным.")
        strength_spread = p.get("spot_strength_spread", 0.0)
        size_spread = p.get("spot_size_spread", 0.0)
        separation = p.get("spot_min_separation", 0.9)
        if not 0 <= strength_spread <= 0.8:
            raise ValueError("Разброс силы очагов должен быть от 0 до 0.8.")
        if not 0 <= size_spread <= 0.75:
            raise ValueError("Разброс размера очагов должен быть от 0 до 0.75.")
        if not 0.5 <= separation <= 1.5:
            raise ValueError("Минимальное расстояние очагов должно быть от 0.5 до 1.5.")
        for key in ("spot_V_drop", "spot_W_drop", "spot_S_drop"):
            if not 0 <= p[key] <= 1:
                raise ValueError(f"{key} должен быть от 0 до 1.")
            if mode != SPOTS_CENTRAL and p[key] * (1 + strength_spread) > 1:
                raise ValueError(
                    f"{key} с учётом разброса силы не должен превышать 1."
                )

    @staticmethod
    def _validate_background_settings(p):
        for key in ("V_bg_amplitude", "W_bg_amplitude", "S_bg_amplitude"):
            amplitude = p.get(key, 0.0)
            if not math.isfinite(amplitude) or not 0 <= amplitude <= 1:
                raise ValueError(f"{key} должен быть от 0 до 1.")
        length = p.get("bg_length_scale", 2.0)
        if not math.isfinite(length) or not 0 < length <= min(p["Lx"], p["Ly"]):
            raise ValueError("Масштаб фоновой неоднородности должен быть > 0 и не больше размера области.")
        for key in ("rain_spatial_amplitude", "growth_spatial_amplitude",
                    "soil_spatial_amplitude"):
            amplitude = p.get(key, 0.0)
            if not math.isfinite(amplitude) or not 0 <= amplitude <= 0.95:
                raise ValueError(f"{key} должен быть от 0 до 0.95.")

    @staticmethod
    def _validate_rain_motion_settings(p):
        if p["rain_map_mode"] not in (RAIN_MAP_FIXED, RAIN_MAP_MOVING):
            raise ValueError("Выберите корректное пространственное распределение осадков.")
        strength = p["rain_motion_strength"]
        speed = p["rain_motion_speed"]
        length = p["rain_motion_length"]
        angle = p["rain_motion_angle"]
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Контраст дождевых областей должен быть от 0 до 1.")
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Скорость дождевых областей должна быть неотрицательной.")
        if not math.isfinite(length) or not 0 < length <= min(p["Lx"], p["Ly"]):
            raise ValueError("Масштаб дождевых областей должен быть > 0 и не больше размера области.")
        if not math.isfinite(angle) or not 0 <= angle <= 360:
            raise ValueError("Направление дождевых областей должно быть от 0 до 360 градусов.")

    def _make_spots(self):
        """Build fixed initial spots; only central mode keeps old placement."""
        p = self.p
        count = p.get("spot_count", 1)
        mode = p.get(
            "spot_mode", SPOTS_CENTRAL if count == 1 else SPOTS_FIXED_COUNT,
        )
        if mode == SPOTS_CENTRAL:
            return [{
                "id": 1,
                "x": float(p["spot_x"]),
                "y": float(p["spot_y"]),
                "sigma": float(p["spot_sigma"]),
                "strength": 1.0,
                "V_drop_peak": float(p["spot_V_drop"]),
                "W_drop_peak": float(p["spot_W_drop"]),
                "S_drop_peak": float(p["spot_S_drop"]),
            }]

        layout_rng = np.random.default_rng(
            np.random.SeedSequence([int(p["seed"]), 7301])
        )
        size_spread = p.get("spot_size_spread", 0.0)
        strength_spread = p.get("spot_strength_spread", 0.0)
        separation = p.get("spot_min_separation", 0.9)
        if count == 1:
            size_factors = np.array([layout_rng.uniform(1 - size_spread, 1 + size_spread)])
            strength_factors = np.array([layout_rng.uniform(1 - strength_spread, 1 + strength_spread)])
        else:
            size_factors = np.linspace(1 - size_spread, 1 + size_spread, count)
            strength_factors = np.linspace(1 - strength_spread, 1 + strength_spread, count)
        layout_rng.shuffle(size_factors)
        layout_rng.shuffle(strength_factors)
        spots = []
        for index in range(count):
            sigma = float(p["spot_sigma"] * size_factors[index])
            strength = float(strength_factors[index])
            if 2 * sigma >= min(self.Lx, self.Ly):
                raise ValueError("Очаг слишком широк для расчётной области.")

            for _ in range(20000):
                x = float(layout_rng.uniform(sigma, self.Lx - sigma))
                y = float(layout_rng.uniform(sigma, self.Ly - sigma))
                if all(
                    math.hypot(x - other["x"], y - other["y"])
                    >= separation * (sigma + other["sigma"])
                    for other in spots
                ):
                    break
            else:
                raise ValueError(
                    "Не удалось разместить очаги без сильного перекрытия. "
                    "Уменьшите число, σ или минимальное расстояние."
                )

            spots.append({
                "id": index + 1,
                "x": x,
                "y": y,
                "sigma": sigma,
                "strength": strength,
                "V_drop_peak": float(p["spot_V_drop"] * strength),
                "W_drop_peak": float(p["spot_W_drop"] * strength),
                "S_drop_peak": float(p["spot_S_drop"] * strength),
            })
        return spots

    def _neumann_1d(self, n, h):
        """1-D second derivative with zero-flux Neumann boundaries."""
        main = -2.0 * np.ones(n)
        upper = np.ones(n - 1)
        lower = np.ones(n - 1)
        A = diags(
            [lower, main, upper],
            [-1, 0, 1],
            shape=(n, n),
            format="lil",
        )
        A[0, 1] = 2.0
        A[-1, -2] = 2.0
        return csc_matrix(A / (h * h))

    def _neumann_laplacian(self):
        Lx = self._neumann_1d(self.nx, self.dx)
        Ly = self._neumann_1d(self.ny, self.dy)
        Ix = eye(self.nx, format="csc")
        Iy = eye(self.ny, format="csc")

        # Arrays are flattened in C-order: (y, x).
        return kron(Iy, Lx) + kron(Ly, Ix)

    def _build_solvers(self):
        dt = self.p["dt"]

        AV = eye(self.N, format="csc") - dt * self.p["Dv"] * self.L
        AW = eye(self.N, format="csc") - dt * self.p["Dw"] * self.L

        # Sparse LU factorisation is done once because A does not change
        # while dt, Dv, Dw and the grid stay fixed.
        self.solve_V = factorized(AV)
        self.solve_W = factorized(AW)

    def _make_initial_background(self):
        """Shared seed-fixed landscape for initial fields and local kinetics."""
        p = self.p
        amplitudes = (
            p.get("V_bg_amplitude", 0.0),
            p.get("W_bg_amplitude", 0.0),
            p.get("S_bg_amplitude", 0.0),
        )
        zeros = np.zeros_like(self.X)
        background_rng = np.random.default_rng(
            np.random.SeedSequence([int(p["seed"]), BACKGROUND_RNG_STREAM])
        )
        length = p.get("bg_length_scale", 2.0)
        pixel_sigma = (length / self.dy, length / self.dx)

        def smooth_unit_field():
            field = gaussian_filter(
                background_rng.normal(size=self.X.shape),
                sigma=pixel_sigma,
                mode="reflect",
            )
            field -= np.mean(field)
            peak = float(np.max(np.abs(field)))
            return field / peak if peak > np.finfo(float).eps else zeros.copy()

        wetness = smooth_unit_field()
        vegetation_independent = smooth_unit_field()
        soil_independent = smooth_unit_field()
        vegetation = 0.65 * wetness + 0.35 * vegetation_independent
        self.rain_factor = 1.0 + p["rain_spatial_amplitude"] * wetness
        self.growth_factor = 1.0 + p["growth_spatial_amplitude"] * vegetation
        self.soil_factor = 1.0 + p["soil_spatial_amplitude"] * soil_independent
        self.landscape_summary = {
            "algorithm": "fixed_landscape_v1",
            "seed_sequence": [int(p["seed"]), BACKGROUND_RNG_STREAM],
            "length_scale_physical": float(length),
            "time_dependence": "fixed climatic component; optional moving weather is applied separately",
            "rainfall_interpretation": "relative local input, not individual storm frequency",
            "factors": {
                name: {"min": float(np.min(field)),
                       "mean": float(np.mean(field)),
                       "max": float(np.max(field))}
                for name, field in (("rain", self.rain_factor),
                                    ("growth", self.growth_factor),
                                    ("soil_recovery", self.soil_factor))
            },
        }
        V_background = amplitudes[0] * vegetation
        W_background = amplitudes[1] * wetness
        S_background = amplitudes[2] * soil_independent
        return V_background, W_background, S_background

    def reset(self):
        p = self.p
        rng = np.random.default_rng(p["seed"])

        # Spots specify initial degradation; landscape factors persist in steps.
        self.spots = self._make_spots()
        V_loss = np.zeros_like(self.X)
        W_loss = np.zeros_like(self.X)
        S_loss = np.zeros_like(self.X)
        for item in self.spots:
            spot = np.exp(
                -((self.X - item["x"]) ** 2 + (self.Y - item["y"]) ** 2)
                / (2 * item["sigma"] ** 2)
            )
            # Maximum prevents overlaps from silently summing large losses.
            np.maximum(V_loss, item["V_drop_peak"] * spot, out=V_loss)
            np.maximum(W_loss, item["W_drop_peak"] * spot, out=W_loss)
            np.maximum(S_loss, item["S_drop_peak"] * spot, out=S_loss)

        V_background, W_background, S_background = self._make_initial_background()
        self._build_rain_motion()
        self.initial_background_summary = {
            "algorithm": "gaussian_filter_reflect_v1; centered and max-abs scaled",
            "seed_sequence": [int(p["seed"]), BACKGROUND_RNG_STREAM],
            "length_scale_physical": float(p.get("bg_length_scale", 2.0)),
            "vegetation_wetness_weight": 0.65,
            "offset_min_max": {
                name: [float(np.min(field)), float(np.max(field))]
                for name, field in (
                    ("V", V_background),
                    ("W", W_background),
                    ("S", S_background),
                )
            },
        }

        self.V = np.clip(
            p["V0"]
            + V_background
            - V_loss
            + p["noise"] * rng.normal(size=self.X.shape),
            0,
            1,
        )
        self.W = np.maximum(
            p["W0"]
            + W_background
            - W_loss
            + p["noise"] * rng.normal(size=self.X.shape),
            0.0,
        )
        self.S = np.clip(
            p["S0"]
            + S_background
            - S_loss
            + p["noise"] * rng.normal(size=self.X.shape),
            0,
            1,
        )

        self.t = 0.0
        self.time_history = []
        self.desertification_index_history = []
        self.index_p10_history = []
        self.index_p90_history = []
        self.vegetation_history = []
        self.moisture_history = []
        self.soil_history = []
        self.precipitation_history = []
        self.precipitation_min_history = []
        self.precipitation_max_history = []
        self._record_state()
        self.last_iterations = 0

    def _record_state(self):
        index = self.desertification_index()
        p10, p90 = np.quantile(index, (0.10, 0.90))
        self.time_history.append(float(self.t))
        self.desertification_index_history.append(float(np.mean(index)))
        self.index_p10_history.append(float(p10))
        self.index_p90_history.append(float(p90))
        self.vegetation_history.append(float(np.mean(self.V)))
        self.moisture_history.append(float(np.mean(self.W)))
        self.soil_history.append(float(np.mean(self.S)))
        self.precipitation_history.append(self.get_precipitation(self.t))
        rain = self.get_precipitation_field(self.t)
        self.precipitation_min_history.append(float(np.min(rain)))
        self.precipitation_max_history.append(float(np.max(rain)))

    def get_precipitation(self, t):
        """Return non-negative precipitation for the selected climate mode."""
        p = self.p
        p0 = p.get("P0", p.get("P", 0.30))
        mode = p.get("precipitation_mode", "Постоянные")

        if mode == "Линейное уменьшение":
            value = max(p.get("Pmin", 0.10), p0 - p.get("alpha", 0.005) * t)
        elif mode == "Линейное увеличение":
            value = min(p.get("Pmax", 0.40), p0 + p.get("alpha", 0.005) * t)
        elif mode == "Сезонные":
            period = p.get("Tseason", 10.0)
            amplitude = p.get("A", 0.08)
            value = p0 + amplitude * np.sin(2.0 * np.pi * t / period)
        else:
            value = p0

        return float(max(0.0, value))

    def _build_rain_motion(self):
        """A seeded smooth weather field advected across an unbounded plane."""
        p = self.p
        rng = np.random.default_rng(
            np.random.SeedSequence([int(p["seed"]), RAIN_MOTION_RNG_STREAM])
        )
        modes = 12
        wave_vectors = rng.normal(size=(modes, 2)) / p["rain_motion_length"]
        weights = rng.uniform(0.5, 1.0, size=modes)
        weights /= np.sqrt(0.5 * np.sum(weights ** 2))
        angle = np.deg2rad(p["rain_motion_angle"])
        vx, vy = p["rain_motion_speed"] * np.array([np.cos(angle), np.sin(angle)])
        self._rain_motion_phase = (
            wave_vectors[:, 0, None, None] * self.X
            + wave_vectors[:, 1, None, None] * self.Y
            + rng.uniform(0, 2 * np.pi, size=modes)[:, None, None]
        )
        self._rain_motion_frequency = wave_vectors @ np.array([vx, vy])
        self._rain_motion_weights = weights
        self._rain_motion_cache = None
        self.rain_motion_summary = {
            "algorithm": "advected_fourier_weather_v1",
            "mode": p["rain_map_mode"],
            "seed_sequence": [int(p["seed"]), RAIN_MOTION_RNG_STREAM],
            "fourier_modes": modes,
            "velocity_xy": [float(vx), float(vy)],
            "spatial_scale": p["rain_motion_length"],
            "contrast": p["rain_motion_strength"],
            "mean_normalization": "instantaneous spatial mean of the rainfall multiplier is 1",
            "boundary": "weather advects on an unbounded plane; no edge wrapping",
            "note": "Synthetic moving rainfall regions, not an atmospheric forecast or a storm-frequency model.",
        }

    def get_rainfall_multiplier(self, t):
        """Combine fixed climatic differences and smoothly moving weather."""
        p = self.p
        if p["rain_map_mode"] == RAIN_MAP_FIXED or p["rain_motion_strength"] == 0:
            return self.rain_factor
        if self._rain_motion_cache is not None and self._rain_motion_cache[0] == t:
            return self._rain_motion_cache[1]
        phases = self._rain_motion_phase - self._rain_motion_frequency[:, None, None] * t
        weather = np.einsum("i,ijk->jk", self._rain_motion_weights, np.cos(phases))
        local = self.rain_factor * np.exp(3.0 * p["rain_motion_strength"] * np.tanh(weather))
        local /= np.mean(local)
        local.setflags(write=False)
        self._rain_motion_cache = (float(t), local)
        return local

    def get_precipitation_field(self, t):
        """Spatial rainfall whose arithmetic mean is P(t) at every instant."""
        return self.get_precipitation(t) * self.get_rainfall_multiplier(t)

    def desertification_index(self):
        """Continuous weighted degradation index D(x, y, t) in [0, 1]."""
        index = (
            0.4 * (1.0 - self.V)
            + 0.3 * (1.0 - np.minimum(self.W, 1.0))
            + 0.3 * (1.0 - self.S)
        )
        return index

    def mean_desertification_index(self):
        """Spatial mean of the continuous desertification index."""
        return float(np.mean(self.desertification_index()))

    def desert_spread_distribution(self):
        """Local smooth display weights; no binary mask or area rounding."""
        z = np.clip(
            (self.desertification_index() - DESERT_SPREAD_START)
            / (DESERT_SPREAD_FULL - DESERT_SPREAD_START), 0.0, 1.0,
        )
        return z * z * (3.0 - 2.0 * z)

    def _reaction_terms(self, V, W, S, precipitation):
        """Evaluate the three local reaction terms at one Picard iterate."""
        p = self.p
        fw = W / (W + p["Kw"])
        fs = S / (S + p["Ks"])
        drought = p["Kd"] / (W + p["Kd"])

        RV = p["r"] * self.growth_factor * V * (1.0 - V) * fw * fs - p["m"] * V * drought
        RW = precipitation - p["e"] * W - p["c"] * V * W
        RS = p["rs"] * self.soil_factor * V * (1.0 - S) - p["ds"] * (1.0 - V) * drought * S
        return RV, RW, RS

    def step(self):
        p = self.p
        dt = p["dt"]
        current_P = self.get_precipitation_field(self.t + dt)

        Vn = self.V.copy()
        Wn = self.W.copy()
        Sn = self.S.copy()

        # Initial Picard guess: previous time level.
        Vk = Vn.copy()
        Wk = Wn.copy()
        Sk = Sn.copy()

        for k in range(p["max_picard"]):
            RV, RW, RS = self._reaction_terms(Vk, Wk, Sk, current_P)

            # Implicit diffusion:
            #
            # (I - dt*D*L) u^(k+1) = u^n + dt*R(u^k)
            Vnew = self.solve_V(
                Vn.ravel() + dt * RV.ravel()
            ).reshape(Vn.shape)

            Wnew = self.solve_W(
                Wn.ravel() + dt * RW.ravel()
            ).reshape(Wn.shape)

            # S has no spatial diffusion.
            Snew = Sn + dt * RS

            # V and S are fractions; W can exceed 1 under sustained rain.
            Vnew = np.clip(Vnew, 0.0, 1.0)
            Wnew = np.maximum(Wnew, 0.0)
            Snew = np.clip(Snew, 0.0, 1.0)

            err = max(
                np.max(np.abs(Vnew - Vk)),
                np.max(np.abs(Wnew - Wk)),
                np.max(np.abs(Snew - Sk)),
            )

            Vk, Wk, Sk = Vnew, Wnew, Snew

            if err < p["picard_tol"]:
                self.last_iterations = k + 1
                break
        else:
            self.last_iterations = p["max_picard"]

        self.V, self.W, self.S = Vk, Wk, Sk
        self.t += dt

        self._record_state()


class App:
    def __init__(self, root):
        self.root = root
        root.title(
            f"Модель опустынивания v{APP_VERSION} — неявная конечно-разностная схема"
        )
        root.geometry("1400x900")
        root.minsize(1100, 700)

        self.running = False
        self.simulation_thread = None
        self.simulation_cancel = threading.Event()
        self.simulation_queue = queue.Queue()
        self._build_variables()
        self._build_ui()
        self._create_model()
        self.update_plots()

    def _build_variables(self):
        self.vars = {}

        defaults = {
            "nx": 80,
            "ny": 80,
            "Lx": 10.0,
            "Ly": 10.0,
            "dt": 0.05,
            "Dv": 0.05,
            "Dw": 0.10,
            "r": 0.50,
            "Kw": 0.30,
            "Ks": 0.30,
            "Kd": 0.30,
            "m": 0.15,
            "P0": 0.30,
            "Pmin": 0.10,
            "Pmax": 0.40,
            "alpha": 0.005,
            "A": 0.08,
            "Tseason": 10.0,
            "e": 0.10,
            "c": 0.15,
            "rs": 0.05,
            "ds": 0.08,
            "V0": 0.70,
            "W0": 0.60,
            "S0": 0.80,
            "seed": 42,
            "spot_mode": SPOTS_CENTRAL,
            "spot_count": 1,
            "spot_sigma": 1.5,
            "spot_V_drop": 0.25,
            "spot_W_drop": 0.25,
            "spot_S_drop": 0.20,
            "spot_strength_spread": 0.0,
            "spot_size_spread": 0.0,
            "spot_min_separation": 0.9,
            "V_bg_amplitude": 0.0,
            "W_bg_amplitude": 0.0,
            "S_bg_amplitude": 0.0,
            "bg_length_scale": 2.0,
            "rain_spatial_amplitude": 0.0,
            "growth_spatial_amplitude": 0.0,
            "soil_spatial_amplitude": 0.0,
            "noise": 0.01,
            "picard_tol": 1e-5,
            "max_picard": 50,
            "steps": 200,
            "precipitation_mode": "Постоянные",
            "rain_map_mode": RAIN_MAP_FIXED,
            "rain_motion_strength": 0.75,
            "rain_motion_speed": 0.50,
            "rain_motion_length": 1.0,
            "rain_motion_angle": 30.0,
        }

        defaults.update(SCENARIO_OVERRIDES["Случайные очаги"])
        self.default_values = defaults.copy()
        self.session_multi_seed = secrets.randbelow(2**32)
        self.session_spot_count = secrets.randbelow(8) + 1
        initial_values = (
            defaults
            | {"seed": self.session_multi_seed, "spot_count": self.session_spot_count}
        )

        for key, value in initial_values.items():
            self.vars[key] = tk.StringVar(value=str(value))

        self.scenario_var = tk.StringVar(value="Случайные очаги")
        self.moisture_scale_var = tk.StringVar(value=MOISTURE_SCALE_LOCAL)

        self.entries = []
        self.comboboxes = []

    def _add_entry(self, parent, row, label, key):
        ttk.Label(parent, text=label).grid(
            row=row,
            column=0,
            sticky="w",
            padx=5,
            pady=2,
        )
        entry = ttk.Entry(
            parent,
            textvariable=self.vars[key],
            width=10,
        )
        entry.grid(
            row=row,
            column=1,
            sticky="ew",
            padx=5,
            pady=2,
        )
        self.entries.append(entry)
        if key == "spot_count":
            self.spot_count_entry = entry

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        controls_pane = ttk.Frame(main, width=375)
        controls_pane.pack(
            side="left",
            fill="y",
            padx=(0, 8),
        )
        controls_pane.pack_propagate(False)

        controls_canvas = tk.Canvas(
            controls_pane,
            highlightthickness=0,
            width=330,
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_pane,
            orient="vertical",
            command=controls_canvas.yview,
        )
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_canvas.pack(side="left", fill="both", expand=True)
        controls_scrollbar.pack(side="right", fill="y")

        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw",
        )
        controls.bind(
            "<Configure>",
            lambda _event: controls_canvas.configure(
                scrollregion=controls_canvas.bbox("all")
            ),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(
                controls_window,
                width=event.width,
            ),
        )

        controls.columnconfigure(1, weight=1)

        ttk.Label(
            controls,
            text=f"Параметры модели · v{APP_VERSION}",
            font=("TkDefaultFont", 12, "bold"),
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 8),
        )

        ttk.Label(controls, text="Базовый сценарий").grid(
            row=1, column=0, sticky="w", padx=5, pady=2,
        )
        self.scenario_label = ttk.Label(
            controls,
            textvariable=self.scenario_var,
        )
        self.scenario_label.grid(
            row=1, column=1, sticky="ew", padx=5, pady=2,
        )
        self.scenario_apply_btn = ttk.Button(
            controls,
            text="Восстановить базовые параметры",
            command=self.apply_scenario,
        )
        self.scenario_apply_btn.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=(2, 8),
        )
        self.new_spots_btn = ttk.Button(
            controls,
            text="Новые случайные очаги (1–8)",
            command=self.new_random_spots,
        )
        self.new_spots_btn.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 8),
        )

        sections = [
            (
                "Сетка и время",
                [
                    ("Nx", "nx"),
                    ("Ny", "ny"),
                    ("Lx", "Lx"),
                    ("Ly", "Ly"),
                    ("Δt", "dt"),
                ],
            ),
            (
                "Диффузия",
                [
                    ("Dv", "Dv"),
                    ("Dw", "Dw"),
                ],
            ),
            (
                "Растительность",
                [
                    ("r — рост", "r"),
                    ("Kw", "Kw"),
                    ("Ks", "Ks"),
                    ("Kd — засуха", "Kd"),
                    ("m — гибель", "m"),
                ],
            ),
            (
                "Вода",
                [
                    ("e — испарение", "e"),
                    ("c — расход", "c"),
                ],
            ),
            (
                "Осадки",
                [
                    ("P0 — базовые осадки", "P0"),
                    ("Pmin — минимум", "Pmin"),
                    ("Pmax — максимум", "Pmax"),
                    ("α — скорость изменения", "alpha"),
                    ("A — амплитуда сезона", "A"),
                    ("Tseason — период", "Tseason"),
                ],
            ),
            (
                "Движение дождевых областей",
                [
                    ("Контраст областей (0–1)", "rain_motion_strength"),
                    ("Скорость (коорд./t)", "rain_motion_speed"),
                    ("Масштаб областей", "rain_motion_length"),
                    ("Направление (градусы)", "rain_motion_angle"),
                ],
            ),
            (
                "Почва",
                [
                    ("rs — восстановление", "rs"),
                    ("ds — деградация", "ds"),
                ],
            ),
            (
                "Постоянные различия территории",
                [
                    ("Разброс осадков ±", "rain_spatial_amplitude"),
                    ("Разброс роста V ±", "growth_spatial_amplitude"),
                    ("Разброс восстановления S ±", "soil_spatial_amplitude"),
                ],
            ),
            (
                "Начальные условия",
                [
                    ("V0", "V0"),
                    ("W0", "W0"),
                    ("S0", "S0"),
                    ("Seed (для повторения)", "seed"),
                    ("Число очагов (факт./зад.)", "spot_count"),
                    ("Падение V в пятне", "spot_V_drop"),
                    ("Падение W в пятне", "spot_W_drop"),
                    ("Падение S в пятне", "spot_S_drop"),
                    ("σ пятна", "spot_sigma"),
                    ("Разброс силы", "spot_strength_spread"),
                    ("Разброс размера", "spot_size_spread"),
                    ("Мин. расстояние", "spot_min_separation"),
                    ("Фон V ±", "V_bg_amplitude"),
                    ("Фон W ±", "W_bg_amplitude"),
                    ("Фон S ±", "S_bg_amplitude"),
                    ("Масштаб фона", "bg_length_scale"),
                    ("Шум", "noise"),
                ],
            ),
            (
                "Решатель",
                [
                    ("Допуск Пикара", "picard_tol"),
                    ("Макс. итераций", "max_picard"),
                ],
            ),
        ]

        row = 4

        for title, items in sections:
            ttk.Label(
                controls,
                text=title,
                font=("TkDefaultFont", 10, "bold"),
            ).grid(
                row=row,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(7, 2),
            )
            row += 1

            for label, key in items:
                self._add_entry(
                    controls,
                    row,
                    label,
                    key,
                )
                row += 1

        ttk.Label(controls, text="Режим осадков").grid(
            row=row,
            column=0,
            sticky="w",
            padx=5,
            pady=2,
        )
        self.precipitation_combo = ttk.Combobox(
            controls,
            textvariable=self.vars["precipitation_mode"],
            values=(
                "Постоянные",
                "Линейное уменьшение",
                "Линейное увеличение",
                "Сезонные",
            ),
            state="readonly",
            width=22,
        )
        self.precipitation_combo.grid(
            row=row,
            column=1,
            sticky="ew",
            padx=5,
            pady=2,
        )
        self.comboboxes.append(self.precipitation_combo)
        row += 1

        ttk.Label(controls, text="Карта осадков").grid(
            row=row, column=0, sticky="w", padx=5, pady=2,
        )
        self.rain_map_combo = ttk.Combobox(
            controls, textvariable=self.vars["rain_map_mode"],
            values=(RAIN_MAP_FIXED, RAIN_MAP_MOVING), state="readonly", width=25,
        )
        self.rain_map_combo.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        self.comboboxes.append(self.rain_map_combo)
        row += 1

        ttk.Label(
            controls,
            text=("Фон территории закреплён по seed. Дождевые области могут двигаться; "
                  "средние осадки остаются равны P(t). После изменений примените сброс."),
            wraplength=325,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 6))
        row += 1

        ttk.Label(controls, text="Режим начальных очагов").grid(
            row=row, column=0, sticky="w", padx=5, pady=2,
        )
        self.spot_mode_combo = ttk.Combobox(
            controls,
            textvariable=self.vars["spot_mode"],
            values=(SPOTS_CENTRAL, SPOTS_RANDOM_COUNT, SPOTS_FIXED_COUNT),
            state="readonly",
            width=25,
        )
        self.spot_mode_combo.grid(
            row=row, column=1, sticky="ew", padx=5, pady=2,
        )
        self.spot_mode_combo.bind(
            "<<ComboboxSelected>>", self._on_spot_mode_changed,
        )
        self.comboboxes.append(self.spot_mode_combo)
        row += 1

        ttk.Label(controls, text="Шкала влажности").grid(
            row=row, column=0, sticky="w", padx=5, pady=2,
        )
        self.moisture_scale_combo = ttk.Combobox(
            controls, textvariable=self.moisture_scale_var,
            values=(MOISTURE_SCALE_LOCAL, MOISTURE_SCALE_COMMON),
            state="readonly", width=25,
        )
        self.moisture_scale_combo.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        self.moisture_scale_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_plots())
        self.comboboxes.append(self.moisture_scale_combo)
        row += 1
        ttk.Label(
            controls, text="В текущем диапазоне сравнивайте влажность по числам на цветовой шкале.",
            wraplength=325,
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 6))

        row += 1
        ttk.Label(controls, text="Шагов моделирования").grid(
            row=row,
            column=0,
            sticky="w",
            padx=5,
            pady=2,
        )
        self.steps_entry = ttk.Spinbox(
            controls,
            from_=1,
            to=1_000_000,
            textvariable=self.vars["steps"],
            width=10,
        )
        self.steps_entry.grid(
            row=row,
            column=1,
            sticky="ew",
            padx=5,
            pady=2,
        )
        self.entries.append(self.steps_entry)

        row += 1
        button_frame = ttk.Frame(controls)
        button_frame.grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=10,
        )

        for column in range(2):
            button_frame.columnconfigure(column, weight=1)

        self.reset_btn = ttk.Button(
            button_frame,
            text="Сброс / применить",
            command=self.reset_model,
        )
        self.reset_btn.grid(
            row=0,
            column=0,
            padx=2,
            pady=2,
            sticky="ew",
        )

        self.run_btn = ttk.Button(
            button_frame,
            text="Запустить",
            command=self.toggle_run,
        )
        self.run_btn.grid(
            row=0,
            column=1,
            padx=2,
            pady=2,
            sticky="ew",
        )

        self.step_btn = ttk.Button(
            button_frame,
            text="Один шаг",
            command=self.one_step,
        )
        self.step_btn.grid(
            row=1,
            column=0,
            padx=2,
            pady=2,
            sticky="ew",
        )

        self.ten_steps_btn = ttk.Button(
            button_frame,
            text="10 шагов",
            command=lambda: self.manual_steps(10),
        )
        self.ten_steps_btn.grid(
            row=1,
            column=1,
            padx=2,
            pady=2,
            sticky="ew",
        )

        self.simulate_btn = ttk.Button(
            button_frame,
            text="Рассчитать заданное число шагов",
            command=self.start_simulation,
        )
        self.simulate_btn.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=2,
            pady=(7, 2),
            sticky="ew",
        )

        self.cancel_simulation_btn = ttk.Button(
            button_frame,
            text="Остановить расчёт",
            command=self.cancel_simulation,
            state="disabled",
        )
        self.cancel_simulation_btn.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=2,
            pady=2,
            sticky="ew",
        )

        self.save_btn = ttk.Button(
            button_frame,
            text="Сохранить отдельно для слайдов…",
            command=self.save_results,
        )
        self.save_btn.grid(
            row=4,
            column=0,
            columnspan=2,
            padx=2,
            pady=2,
            sticky="ew",
        )

        self.save_parameters_btn = ttk.Button(
            button_frame,
            text="Сохранить параметры JSON…",
            command=self.save_parameters_json,
        )
        self.save_parameters_btn.grid(
            row=5,
            column=0,
            columnspan=2,
            padx=2,
            pady=(2, 7),
            sticky="ew",
        )

        self.progress = ttk.Progressbar(
            controls,
            mode="determinate",
            maximum=100,
            value=0,
        )
        self.progress.grid(
            row=row + 1,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=5,
            pady=(4, 0),
        )

        self.status = tk.StringVar(value="t = 0.00")

        ttk.Label(
            controls,
            textvariable=self.status,
            wraplength=300,
        ).grid(
            row=row + 2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=8,
        )

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.tabs = {}

        for name in [
            "Растительность V",
            "Влажность W",
            "Почва S",
            "Индекс опустынивания D",
            "Распространение пустыни",
            "Осадки на территории",
            "Динамика индекса",
            "Динамика V, W, S",
            "Осадки P(t)",
        ]:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(
                frame,
                text=name,
            )
            self.tabs[name] = frame

        self.figures = {}
        self.axes = {}
        self.canvases = {}

        # Each tab has its own matplotlib figure.
        for name, frame in self.tabs.items():
            figure, axis = plt.subplots(
                figsize=(8, 6),
                constrained_layout=True,
            )

            canvas = FigureCanvasTkAgg(
                figure,
                master=frame,
            )

            canvas.get_tk_widget().pack(
                fill="both",
                expand=True,
            )

            self.figures[name] = figure
            self.axes[name] = axis
            self.canvases[name] = canvas

        self._sync_spot_count_entry_state()

    def _sync_spot_count_entry_state(self):
        if self.simulation_thread is None:
            state = "normal" if self.vars["spot_mode"].get() == SPOTS_FIXED_COUNT else "readonly"
            self.spot_count_entry.configure(state=state)

    def _on_spot_mode_changed(self, _event):
        mode = self.vars["spot_mode"].get()
        if mode == SPOTS_CENTRAL:
            self.vars["spot_count"].set("1")
        elif mode == SPOTS_RANDOM_COUNT:
            try:
                count = int(self.vars["spot_count"].get())
            except ValueError:
                count = 0
            if not 1 <= count <= 8:
                self.vars["spot_count"].set(str(self.session_spot_count))
        self._sync_spot_count_entry_state()

    def _get_params(self):
        def f(key):
            return float(self.vars[key].get())

        def i(key):
            return int(float(self.vars[key].get()))

        def exact_i(key):
            value = float(self.vars[key].get())
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError(f"{key} должен быть целым числом.")
            return int(value)

        p = {
            "nx": i("nx"),
            "ny": i("ny"),
            "Lx": f("Lx"),
            "Ly": f("Ly"),
            "dt": f("dt"),
            "Dv": f("Dv"),
            "Dw": f("Dw"),
            "r": f("r"),
            "Kw": f("Kw"),
            "Ks": f("Ks"),
            "Kd": f("Kd"),
            "m": f("m"),
            "P0": f("P0"),
            "Pmin": f("Pmin"),
            "Pmax": f("Pmax"),
            "alpha": f("alpha"),
            "A": f("A"),
            "Tseason": f("Tseason"),
            "precipitation_mode": self.vars["precipitation_mode"].get(),
            "rain_map_mode": self.vars["rain_map_mode"].get(),
            "rain_motion_strength": f("rain_motion_strength"),
            "rain_motion_speed": f("rain_motion_speed"),
            "rain_motion_length": f("rain_motion_length"),
            "rain_motion_angle": f("rain_motion_angle"),
            "e": f("e"),
            "c": f("c"),
            "rs": f("rs"),
            "ds": f("ds"),
            "V0": f("V0"),
            "W0": f("W0"),
            "S0": f("S0"),
            "spot_mode": self.vars["spot_mode"].get(),
            "spot_count": exact_i("spot_count"),
            "spot_x": f("Lx") / 2,
            "spot_y": f("Ly") / 2,
            "spot_sigma": f("spot_sigma"),
            "spot_V_drop": f("spot_V_drop"),
            "spot_W_drop": f("spot_W_drop"),
            "spot_S_drop": f("spot_S_drop"),
            "spot_strength_spread": f("spot_strength_spread"),
            "spot_size_spread": f("spot_size_spread"),
            "spot_min_separation": f("spot_min_separation"),
            "V_bg_amplitude": f("V_bg_amplitude"),
            "W_bg_amplitude": f("W_bg_amplitude"),
            "S_bg_amplitude": f("S_bg_amplitude"),
            "bg_length_scale": f("bg_length_scale"),
            "rain_spatial_amplitude": f("rain_spatial_amplitude"),
            "growth_spatial_amplitude": f("growth_spatial_amplitude"),
            "soil_spatial_amplitude": f("soil_spatial_amplitude"),
            "noise": f("noise"),
            "picard_tol": f("picard_tol"),
            "max_picard": i("max_picard"),
            "seed": exact_i("seed"),
        }

        if p["nx"] < 3 or p["ny"] < 3:
            raise ValueError("Nx и Ny должны быть не меньше 3.")

        if p["dt"] <= 0:
            raise ValueError("Δt должен быть > 0.")

        if p["Kw"] <= 0 or p["Ks"] <= 0 or p["Kd"] <= 0:
            raise ValueError("Kw, Ks и Kd должны быть > 0.")

        DesertificationModel._validate_background_settings(p)
        DesertificationModel._validate_rain_motion_settings(p)

        if min(p["P0"], p["Pmin"], p["Pmax"], p["alpha"], p["A"]) < 0:
            raise ValueError("Параметры осадков должны быть неотрицательными.")

        if p["Pmin"] > p["Pmax"]:
            raise ValueError("Pmin не может превышать Pmax.")

        if p["Tseason"] <= 0:
            raise ValueError("Tseason должен быть > 0.")

        if p["precipitation_mode"] not in {
            "Постоянные",
            "Линейное уменьшение",
            "Линейное увеличение",
            "Сезонные",
        }:
            raise ValueError("Выберите корректный режим осадков.")

        return p

    def _create_model(self):
        params = self._get_params()
        self.model = DesertificationModel(params)

    def apply_scenario(self):
        if self.simulation_thread is not None:
            return
        selected = self.scenario_var.get()
        values = self.default_values | SCENARIO_OVERRIDES[selected]
        if selected in RANDOM_SPOT_SCENARIOS:
            values["seed"] = self.session_multi_seed
            values["spot_count"] = self.session_spot_count
        for key, value in values.items():
            self.vars[key].set(str(value))
        self._sync_spot_count_entry_state()
        self.reset_model()

    def new_random_spots(self):
        if self.simulation_thread is not None:
            return
        if self.vars["spot_mode"].get() == SPOTS_CENTRAL:
            # Keep kinetics; enter the random initial-state configuration.
            for key in (
                "spot_sigma", "spot_strength_spread", "spot_size_spread",
                "spot_min_separation", "V_bg_amplitude", "W_bg_amplitude",
                "S_bg_amplitude", "bg_length_scale",
            ):
                self.vars[key].set(str(SCENARIO_OVERRIDES["Случайные очаги"][key]))
        self.vars["spot_mode"].set(SPOTS_RANDOM_COUNT)
        old_seed = self.vars["seed"].get()
        new_seed = secrets.randbelow(2**32)
        while str(new_seed) == old_seed:
            new_seed = secrets.randbelow(2**32)
        self.session_multi_seed = new_seed
        self.session_spot_count = secrets.randbelow(8) + 1
        self.vars["seed"].set(str(self.session_multi_seed))
        self.vars["spot_count"].set(str(self.session_spot_count))
        self._sync_spot_count_entry_state()
        self.reset_model()

    def reset_model(self):
        try:
            self.running = False
            self.run_btn.configure(text="Запустить")
            self.simulate_btn.configure(state="normal")
            self._create_model()
            if self.model.p["spot_mode"] != SPOTS_CENTRAL:
                self.session_multi_seed = self.model.p["seed"]
                if self.model.p["spot_count"] <= 8:
                    self.session_spot_count = self.model.p["spot_count"]
            self._sync_spot_count_entry_state()
            self.update_plots()
        except Exception as exc:
            messagebox.showerror(
                "Ошибка параметров",
                str(exc),
            )

    def one_step(self):
        if self.running:
            return

        try:
            self.model.step()
            self.update_plots()
        except Exception as exc:
            messagebox.showerror(
                "Ошибка расчёта",
                str(exc),
            )

    def manual_steps(self, count):
        if self.running:
            return

        try:
            for _ in range(count):
                self.model.step()

            self.update_plots()
        except Exception as exc:
            messagebox.showerror(
                "Ошибка расчёта",
                str(exc),
            )

    def toggle_run(self):
        if self.simulation_thread is not None:
            return

        self.running = not self.running

        self.run_btn.configure(
            text="Пауза" if self.running else "Запустить"
        )
        self.simulate_btn.configure(
            state="disabled" if self.running else "normal"
        )

        if self.running:
            self._tick()

    def start_simulation(self):
        if self.running or self.simulation_thread is not None:
            return

        try:
            steps = int(self.vars["steps"].get())
            if steps < 1:
                raise ValueError("Число шагов моделирования должно быть не меньше 1.")
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Ошибка параметров", str(exc))
            return

        self.simulation_cancel.clear()
        self.simulation_queue = queue.Queue()
        self.progress.configure(maximum=steps, value=0)

        for widget in (
            *self.entries,
            self.reset_btn,
            self.run_btn,
            self.step_btn,
            self.ten_steps_btn,
            self.simulate_btn,
            self.save_btn,
            self.save_parameters_btn,
            self.scenario_apply_btn,
            self.new_spots_btn,
        ):
            widget.configure(state="disabled")
        for widget in self.comboboxes:
            widget.configure(state="disabled")
        self.cancel_simulation_btn.configure(state="normal")
        self.status.set(f"Расчёт: 0 / {steps} шагов…")

        self.simulation_thread = threading.Thread(
            target=self._simulation_worker,
            args=(steps,),
            daemon=True,
        )
        self.simulation_thread.start()
        self.root.after(50, self._poll_simulation)

    def _simulation_worker(self, steps):
        report_every = max(1, steps // 100)
        completed = 0

        try:
            for index in range(steps):
                if self.simulation_cancel.is_set():
                    break

                self.model.step()
                completed = index + 1

                if completed % report_every == 0 or completed == steps:
                    self.simulation_queue.put(("progress", completed))

            self.simulation_queue.put(("done", completed))
        except Exception as exc:
            self.simulation_queue.put(("error", str(exc)))

    def cancel_simulation(self):
        if self.simulation_thread is not None:
            self.simulation_cancel.set()
            self.cancel_simulation_btn.configure(state="disabled")
            self.status.set("Остановка после завершения текущего шага…")

    def _spatial_status_text(self):
        return (
            f"D: 10–90% клеток = {self.model.index_p10_history[-1]:.3f}–"
            f"{self.model.index_p90_history[-1]:.3f}"
        )

    def _poll_simulation(self):
        finished = False
        completed = 0
        error = None

        while True:
            try:
                kind, value = self.simulation_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "progress":
                self.progress.configure(value=value)
                self.status.set(
                    f"Расчёт: {value} шагов | "
                    f"t = {self.model.t:.3f} | "
                    f"P = {self.model.precipitation_history[-1]:.3f} | "
                    f"Dmean = {self.model.desertification_index_history[-1]:.3f} | "
                    f"{self._spatial_status_text()}"
                )
            elif kind == "done":
                finished = True
                completed = value
            elif kind == "error":
                finished = True
                error = value

        if not finished:
            self.root.after(50, self._poll_simulation)
            return

        if self.simulation_thread is not None:
            self.simulation_thread.join()
        self.simulation_thread = None
        self.cancel_simulation_btn.configure(state="disabled")
        for widget in (
            *self.entries,
            self.reset_btn,
            self.run_btn,
            self.step_btn,
            self.ten_steps_btn,
            self.simulate_btn,
            self.save_btn,
            self.save_parameters_btn,
            self.scenario_apply_btn,
            self.new_spots_btn,
        ):
            widget.configure(state="normal")
        for widget in self.comboboxes:
            widget.configure(state="readonly")
        self._sync_spot_count_entry_state()

        self.update_plots()
        if error is not None:
            messagebox.showerror("Ошибка расчёта", error)
        elif self.simulation_cancel.is_set():
            self.status.set(
                f"Расчёт остановлен: выполнено {completed} шагов | "
                f"t = {self.model.t:.3f} | "
                f"P = {self.model.precipitation_history[-1]:.3f} | "
                f"Dmean = {self.model.desertification_index_history[-1]:.3f} | "
                f"{self._spatial_status_text()}"
            )
        else:
            self.progress.configure(value=completed)
            self.status.set(
                f"Расчёт завершён: {completed} шагов | "
                f"t = {self.model.t:.3f} | "
                f"P = {self.model.precipitation_history[-1]:.3f} | "
                f"Dmean = {self.model.desertification_index_history[-1]:.3f} | "
                f"{self._spatial_status_text()}"
            )

    def _parameter_snapshot(self):
        displayed_fields = {key: variable.get() for key, variable in self.vars.items()}
        applied_parameters = dict(self.model.p)
        matched_scenario = identify_applied_scenario(
            applied_parameters, self.default_values,
        )
        selected_scenario = self.scenario_var.get()
        field_differences = displayed_parameter_differences(
            displayed_fields, applied_parameters,
        )
        # The scenario name is now a fixed baseline label. Applied custom
        # parameters are not an unapplied change to a scenario selector.
        scenario_difference = (
            matched_scenario != "Пользовательские параметры"
            and selected_scenario != matched_scenario
        )

        return {
            "format": "desertification-model-parameter-snapshot",
            "schema_version": PARAMETER_SNAPSHOT_SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "program_file": str(Path(__file__).resolve()),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "visualization": {"moisture_color_scale": self._moisture_color_scale()},
            "displayed_in_ui": {
                "selected_scenario": selected_scenario,
                "fields": displayed_fields,
                "steps_field_applies_to_next_calculation": True,
            },
            "applied_run": {
                "matched_scenario": matched_scenario,
                "parameters": applied_parameters,
                "actual_spot_count": len(self.model.spots),
                "initial_spots": self.model.spots,
                "initial_background": self.model.initial_background_summary,
                "persistent_landscape": self.model.landscape_summary,
                "rain_motion": self.model.rain_motion_summary,
                "local_precipitation_min_max_at_t": [
                    float(np.min(self.model.get_precipitation_field(self.model.t))),
                    float(np.max(self.model.get_precipitation_field(self.model.t))),
                ],
                "rainfall_map": {
                    "quantity": "P(x,y,t)",
                    "note": "Текущие местные осадки; rain_factor в NPZ — постоянный климатический фон.",
                },
                "spot_overlap_rule": "maximum loss per field, not a sum",
                "time_t": float(self.model.t),
                "completed_steps": len(self.model.time_history) - 1,
                "precipitation_at_t": float(self.model.precipitation_history[-1]),
                "mean_desertification_index_at_t": (
                    self.model.mean_desertification_index()
                ),
                "spatial_index_p10_p90_at_t": np.quantile(
                    self.model.desertification_index(), (0.10, 0.90),
                ).tolist(),
                "displayed_map": {
                    "algorithm": "continuous_weighted_index_v1",
                    "range": [0, 1],
                    "note": "Непрерывный индекс D; бинарная классификация удалена.",
                },
                "desert_spread_map": {
                    "algorithm": "local_smoothstep_distribution_v1",
                    "transition_start_D": DESERT_SPREAD_START,
                    "transition_full_D": DESERT_SPREAD_FULL,
                    "range": [0, 1],
                    "note": "Плавная местная оценка распространения; не бинарная площадь.",
                },
            },
            "differences": {
                "displayed_fields_not_applied": field_differences,
                "scenario_selection_not_applied": scenario_difference,
                "has_unapplied_changes": bool(
                    field_differences or scenario_difference
                ),
            },
            "reproduction_note": (
                "Для воспроизведения используйте applied_run.parameters, "
                "запустите расчёт заново с t=0 и выполните "
                "applied_run.completed_steps шагов. Поля displayed_in_ui "
                "могут содержать ещё не применённые изменения."
            ),
        }

    def save_parameters_json(self):
        if self.simulation_thread is not None:
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить параметры и состояние расчёта",
            initialfile=f"desertification_parameters_t{self.model.t:.2f}.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(
                    self._parameter_snapshot(), stream,
                    ensure_ascii=False, indent=2, allow_nan=False,
                )
                stream.write("\n")
            self.status.set(f"Параметры сохранены: {path}")
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Ошибка сохранения параметров", str(exc))

    def save_results(self):
        if self.simulation_thread is not None:
            return
        parent = filedialog.askdirectory(title="Папка для отдельных карт и графиков")
        if not parent:
            return
        stem = f"desertification_t{self.model.t:.2f}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        output = Path(parent) / stem
        suffix = 1
        while output.exists():
            output = Path(parent) / f"{stem}_{suffix}"
            suffix += 1
        try:
            paths = self.export_results(output)
            self.status.set(f"Сохранено {len(paths)} файлов: {output}")
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", str(exc))

    def _moisture_color_scale(self):
        """Use physical W values; local limits expose contrast without rescaling W."""
        variable = getattr(self, "moisture_scale_var", None)
        mode = variable.get() if variable is not None else MOISTURE_SCALE_LOCAL
        data_min = float(np.min(self.model.W))
        data_max = float(np.max(self.model.W))
        if mode == MOISTURE_SCALE_COMMON:
            lower, upper = 0.0, max(1.0, data_max)
        else:
            mode = MOISTURE_SCALE_LOCAL
            # A nearly uniform field should stay visually uniform, including
            # during dry seasons when W is close to zero.
            minimum_span = max(0.02, 0.05 * data_max)
            padding = max(0.0, (minimum_span - (data_max - data_min)) / 2)
            lower = max(0.0, data_min - padding)
            upper = max(data_max + padding, lower + minimum_span)
            ticks = MaxNLocator(nbins=6, min_n_ticks=3).tick_values(lower, upper)
            lower, upper = max(0.0, float(ticks[0])), float(ticks[-1])
        return {"mode": mode, "vmin": lower, "vmax": upper,
                "data_min": data_min, "data_max": data_max,
                "values_are_physical_W": True}

    def _map_specs(self):
        model = self.model
        moisture_scale = self._moisture_color_scale()
        moisture_title = (
            "Влажность почвы W\n"
            f"min = {moisture_scale['data_min']:.3f}; max = {moisture_scale['data_max']:.3f}"
        )
        rain = model.get_precipitation_field(model.t)
        reference_P = model.p.get("P0", model.p.get("P", 0.30))
        if model.p.get("precipitation_mode") == "Сезонные":
            reference_P += model.p.get("A", 0.08)
        reference_P = max(reference_P, max(model.precipitation_history), 1e-6)
        moving = model.p["rain_map_mode"] == RAIN_MAP_MOVING
        rain_vmax = max(
            reference_P * (4.0 if moving else 2.0),
            max(model.precipitation_max_history), float(np.max(rain)),
        )
        return [
            ("Растительность V", "01_vegetation", "Растительность V", model.V,
             "YlGn", 0.0, 1.0, "Плотность растительности V"),
            ("Влажность W", "02_moisture", moisture_title, model.W,
             "Blues", moisture_scale["vmin"], moisture_scale["vmax"], "Влажность W"),
            ("Почва S", "03_soil", "Качество почвы S", model.S,
             "YlOrBr_r", 0.0, 1.0, "Качество почвы S"),
            ("Индекс опустынивания D", "04_desertification_index", "Индекс опустынивания D",
             model.desertification_index(), "magma", 0.0, 1.0, "Степень деградации D"),
            ("Распространение пустыни", "09_desert_spread", "Распространение пустыни",
             model.desert_spread_distribution(), DESERT_SPREAD_CMAP, 0.0, 1.0,
             "Плавная оценка распространения"),
            ("Осадки на территории", "05_rainfall_distribution",
             f"Осадки на территории\nСреднее P(t) = {model.get_precipitation(model.t):.3f}",
             rain, "Blues", 0.0, rain_vmax, "Местные осадки P(x, y, t)"),
        ]

    def _render_series(self, axis, name, fontsize=12):
        model = self.model
        times = np.asarray(model.time_history)
        if name == "Динамика индекса":
            axis.plot(times, model.desertification_index_history, color="#7a3e9d",
                      linewidth=2.4, label="Средний индекс D")
            axis.set_title("Динамика индекса опустынивания", fontsize=fontsize + 3, pad=14)
            axis.set_ylabel("Индекс D [0, 1]", fontsize=fontsize)
            axis.set_ylim(0, 1)
        elif name == "Динамика V, W, S":
            for values, color, label in (
                (model.vegetation_history, "#25824c", "Растительность V"),
                (model.moisture_history, "#2878a5", "Влажность W"),
                (model.soil_history, "#996b30", "Качество почвы S"),
            ):
                axis.plot(times, values, color=color, linewidth=2.4, label=label)
            axis.set_title("Динамика состояния территории", fontsize=fontsize + 3, pad=14)
            axis.set_ylabel("Среднее значение по территории", fontsize=fontsize)
            axis.set_ylim(0, max(1.0, 1.05 * max(model.moisture_history)))
        elif name == "Осадки P(t)":
            rain = np.asarray(model.precipitation_history)
            axis.fill_between(times, model.precipitation_min_history,
                              model.precipitation_max_history, color="#2878a5", alpha=0.18,
                              label="Минимум–максимум по территории")
            axis.plot(times, rain, color="#2878a5", linewidth=2.4, label="Средние осадки P(t)")
            axis.set_title("Осадки: среднее и пространственный диапазон", fontsize=fontsize + 3, pad=14)
            axis.set_ylabel("Интенсивность осадков P", fontsize=fontsize)
            axis.set_ylim(bottom=0)
        else:
            raise ValueError(f"Неизвестный график: {name}")
        axis.set_xlabel("Время t (безразмерное)", fontsize=fontsize)
        axis.set_xlim(0, max(float(times[-1]), model.p["dt"]))
        axis.tick_params(labelsize=fontsize)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=fontsize - 1, loc="best")

    def _export_map(self, directory, spec):
        """Render the same physical field and limits used by the GUI."""
        _name, stem, title, data, cmap, vmin, vmax, label = spec
        paths = []
        figure = Figure(figsize=(10, 8), dpi=300, layout="constrained")
        FigureCanvasAgg(figure)
        try:
            axis = figure.add_subplot(111)
            image = axis.imshow(data, origin="lower", extent=(0, self.model.Lx, 0, self.model.Ly),
                                cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
            time_label = f"t = {self.model.t:.2f}"
            axis.set_title(f"{title}\n{time_label}", fontsize=20, pad=14)
            axis.set_xlabel("x", fontsize=17)
            axis.set_ylabel("y", fontsize=17)
            axis.tick_params(labelsize=16)
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label(label, fontsize=16, labelpad=14)
            colorbar.ax.tick_params(labelsize=15)
            for extension in ("png", "svg"):
                path = Path(directory) / f"{stem}.{extension}"
                figure.savefig(path, dpi=300, facecolor="white")
                paths.append(path)
        finally:
            figure.clear()
        return paths

    def export_results(self, directory):
        """Write one presentation-sized PNG/SVG per map or chart, plus data."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=False)
        paths = []
        for spec in self._map_specs():
            paths.extend(self._export_map(output, spec))
        for name, stem in (("Динамика индекса", "06_index_dynamics"),
                           ("Динамика V, W, S", "07_state_dynamics"),
                           ("Осадки P(t)", "08_precipitation")):
            figure = Figure(figsize=(12.8, 7.2), dpi=300, layout="constrained")
            FigureCanvasAgg(figure)
            try:
                self._render_series(figure.add_subplot(111), name, fontsize=17)
                for extension in ("png", "svg"):
                    path = output / f"{stem}.{extension}"
                    figure.savefig(path, dpi=300, facecolor="white")
                    paths.append(path)
            finally:
                figure.clear()
        parameter_path = output / "parameters.json"
        parameter_path.write_text(json.dumps(self._parameter_snapshot(), ensure_ascii=False,
                                            indent=2, allow_nan=False) + "\n", encoding="utf-8")
        paths.append(parameter_path)
        field_path = output / "fields.npz"
        np.savez_compressed(field_path, x=self.model.x, y=self.model.y, V=self.model.V,
                            W=self.model.W, S=self.model.S, D=self.model.desertification_index(),
                            desert_spread=self.model.desert_spread_distribution(),
                            rain_factor=self.model.rain_factor, growth_factor=self.model.growth_factor,
                            soil_factor=self.model.soil_factor,
                            rainfall_multiplier=self.model.get_rainfall_multiplier(self.model.t),
                            precipitation=self.model.get_precipitation_field(self.model.t))
        paths.append(field_path)
        history_path = output / "history.csv"
        np.savetxt(history_path, np.column_stack((
            self.model.time_history, self.model.desertification_index_history,
            self.model.index_p10_history, self.model.index_p90_history,
            self.model.vegetation_history, self.model.moisture_history,
            self.model.soil_history, self.model.precipitation_history,
            self.model.precipitation_min_history, self.model.precipitation_max_history,
        )), delimiter=",", header="t,D_mean,D_p10,D_p90,V_mean,W_mean,S_mean,P_mean,P_min,P_max", comments="")
        paths.append(history_path)
        return paths

    def _tick(self):
        if not self.running:
            return

        try:
            self.model.step()
            self.update_plots()
            self.root.after(20, self._tick)
        except Exception as exc:
            self.running = False
            self.run_btn.configure(text="Запустить")
            self.simulate_btn.configure(state="normal")
            messagebox.showerror(
                "Ошибка расчёта",
                str(exc),
            )

    def _draw_field(
        self,
        name,
        data,
        title,
        cmap="viridis",
        vmin=0,
        vmax=1,
    ):
        axis = self.axes[name]
        extent = [0, self.model.Lx, 0, self.model.Ly]
        image = getattr(axis, "_model_image", None)

        if image is None:
            image = axis.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="nearest",
            )
            axis._model_image = image
            axis._model_cbar = self.figures[name].colorbar(image, ax=axis)
        else:
            image.set_data(data)
            image.set_extent(extent)
            image.set_cmap(cmap)
            image.set_clim(vmin, vmax)
            axis._model_cbar.update_normal(image)

        axis.set_xlim(0, self.model.Lx)
        axis.set_ylim(0, self.model.Ly)

        axis.set_title(
            f"{title}, t = {self.model.t:.2f}"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")

        self.canvases[name].draw_idle()

    def update_plots(self):
        model = self.model
        for name, _stem, title, data, cmap, vmin, vmax, label in self._map_specs():
            self._draw_field(name, data, title, cmap=cmap, vmin=vmin, vmax=vmax)
            self.axes[name]._model_cbar.set_label(label)
        for name in ("Динамика индекса", "Динамика V, W, S", "Осадки P(t)"):
            axis = self.axes[name]
            axis.clear()
            self._render_series(axis, name)
            self.canvases[name].draw_idle()
        self.status.set(
            f"t = {model.t:.3f} | "
            f"P = {model.precipitation_history[-1]:.3f} | "
            f"Dmean = {model.desertification_index_history[-1]:.3f} | "
            f"{self._spatial_status_text()} | "
            f"итераций Пикара = {model.last_iterations}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
