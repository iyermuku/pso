from __future__ import annotations

import numpy as np

from landscape_core import LandscapeProblem


def _problem(problem_id: str, label: str, lo: np.ndarray, hi: np.ndarray, fn) -> LandscapeProblem:
    def evaluate(x: np.ndarray):
        value = float(fn(np.asarray(x, dtype=float)))
        return value, value, 0.0

    return LandscapeProblem(
        problem_id=problem_id,
        label=label,
        lo=np.asarray(lo, dtype=float),
        hi=np.asarray(hi, dtype=float),
        evaluate=evaluate,
        calibrate=None,
    )


def _bounds(low, high, dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if dim is None:
        return np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    return np.full(dim, low, dtype=float), np.full(dim, high, dtype=float)


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def _ellipsoid(x: np.ndarray) -> float:
    dim = len(x)
    weights = np.power(1e6, np.arange(dim) / max(dim - 1, 1)) if dim > 1 else np.array([1.0])
    return float(np.sum(weights * x * x))


def _sum_of_different_powers(x: np.ndarray) -> float:
    powers = np.arange(2, len(x) + 2, dtype=float)
    return float(np.sum(np.abs(x) ** powers))


def _zakharov(x: np.ndarray) -> float:
    idx = np.arange(1, len(x) + 1, dtype=float)
    s1 = np.sum(x * x)
    s2 = np.sum(0.5 * idx * x)
    return float(s1 + s2 * s2 + s2**4)


def _rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def _step(x: np.ndarray) -> float:
    return float(np.sum(np.floor(x + 0.5) ** 2))


def _quartic(x: np.ndarray) -> float:
    idx = np.arange(1, len(x) + 1, dtype=float)
    return float(np.sum(idx * x**4))


def _schwefel_222(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


def _schwefel_12(x: np.ndarray) -> float:
    c = np.cumsum(x)
    return float(np.sum(c * c))


def _schwefel_221(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def _rastrigin(x: np.ndarray) -> float:
    return float(10.0 * len(x) + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def _ackley(x: np.ndarray) -> float:
    dim = len(x)
    s1 = np.sum(x * x)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(s1 / dim)) - np.exp(s2 / dim) + 20.0 + np.e)


def _griewank(x: np.ndarray) -> float:
    idx = np.arange(1, len(x) + 1, dtype=float)
    return float(np.sum(x * x) / 4000.0 - np.prod(np.cos(x / np.sqrt(idx))) + 1.0)


def _levy(x: np.ndarray) -> float:
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    term2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2)) if len(x) > 1 else 0.0
    return float(term1 + term2 + term3)


def _michalewicz(x: np.ndarray, m: float = 10.0) -> float:
    idx = np.arange(1, len(x) + 1, dtype=float)
    return float(-np.sum(np.sin(x) * np.sin(idx * x * x / np.pi) ** (2.0 * m)))


def _alpine1(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))


def _alpine2(x: np.ndarray) -> float:
    return float(np.prod(np.sqrt(np.abs(x)) * np.sin(x)))


def _bent_cigar(x: np.ndarray) -> float:
    return float(x[0] ** 2 + 1.0e6 * np.sum(x[1:] ** 2))


def _discus(x: np.ndarray) -> float:
    return float(1.0e6 * x[0] ** 2 + np.sum(x[1:] ** 2))


def _weierstrass(x: np.ndarray, a: float = 0.5, b: float = 3.0, kmax: int = 20) -> float:
    idx = np.arange(0, kmax + 1, dtype=float)
    coeff_a = a**idx
    coeff_b = b**idx
    term1 = np.sum([np.sum(coeff_a * np.cos(2.0 * np.pi * coeff_b * (xi + 0.5))) for xi in x])
    term2 = len(x) * np.sum(coeff_a * np.cos(np.pi * coeff_b))
    return float(term1 - term2)


def _happycat(x: np.ndarray) -> float:
    dim = len(x)
    s1 = np.sum(x * x)
    return float(((s1 - dim) ** 2) ** 0.25 + (0.5 * s1 + np.sum(x)) / dim + 0.5)


def _hgbat(x: np.ndarray) -> float:
    dim = len(x)
    s1 = np.sum(x * x)
    s2 = np.sum(x)
    return float(np.abs(s1 * s1 - s2 * s2) ** 0.5 + (0.5 * s1 + s2) / dim + 0.5)


def _qing(x: np.ndarray) -> float:
    idx = np.arange(1, len(x) + 1, dtype=float)
    return float(np.sum((x * x - idx) ** 2))


def _salomon(x: np.ndarray) -> float:
    r = np.sqrt(np.sum(x * x))
    return float(1.0 - np.cos(2.0 * np.pi * r) + 0.1 * r)


def _xy(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        raise ValueError("Benchmark function requires at least 2 dimensions")
    return float(x[0]), float(x[1])


def _bohachevsky(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    return float(x1 * x1 + 2.0 * x2 * x2 - 0.3 * np.cos(3.0 * np.pi * x1) - 0.4 * np.cos(4.0 * np.pi * x2) + 0.7)


def _booth(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    return float((x1 + 2.0 * x2 - 7.0) ** 2 + (2.0 * x1 + x2 - 5.0) ** 2)


def _matyas(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    return float(0.26 * (x1 * x1 + x2 * x2) - 0.48 * x1 * x2)


def _three_hump_camel(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    return float(2.0 * x1 * x1 - 1.05 * x1**4 + x1**6 / 6.0 + x1 * x2 + x2 * x2)


def _six_hump_camel(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    return float((4.0 - 2.1 * x1 * x1 + x1**4 / 3.0) * x1 * x1 + x1 * x2 + (-4.0 + 4.0 * x2 * x2) * x2 * x2)


def _goldstein_price(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    a = 1.0 + (x1 + x2 + 1.0) ** 2 * (19.0 - 14.0 * x1 + 3.0 * x1**2 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2**2)
    b = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (18.0 - 32.0 * x1 + 12.0 * x1**2 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2**2)
    return float(a * b)


def _branin(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    a = x2 - 5.1 / (4.0 * np.pi**2) * x1**2 + 5.0 * x1 / np.pi - 6.0
    b = 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(x1)
    return float(a * a + b + 10.0)


def _shubert(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    s1 = np.sum([i * np.cos((i + 1.0) * x1 + i) for i in range(1, 6)])
    s2 = np.sum([i * np.cos((i + 1.0) * x2 + i) for i in range(1, 6)])
    return float(s1 * s2)


def _himmelblau(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    return float((x1 * x1 + x2 - 11.0) ** 2 + (x1 + x2 * x2 - 7.0) ** 2)


def _easom(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    return float(-np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2)))


def _cross_in_tray(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    term = np.abs(100.0 - np.sqrt(x1 * x1 + x2 * x2) / np.pi)
    inner = np.abs(np.sin(x1) * np.sin(x2) * np.exp(term))
    return float(-0.0001 * (inner + 1.0) ** 0.1)


def _holder_table(x: np.ndarray) -> float:
    x1, x2 = _xy(x)
    term = np.abs(1.0 - np.sqrt(x1 * x1 + x2 * x2) / np.pi)
    return float(-np.abs(np.sin(x1) * np.cos(x2) * np.exp(term)))


def get_all_benchmark_problems() -> list[LandscapeProblem]:
    problems: list[LandscapeProblem] = []

    def add(problem_id: str, label: str, lo, hi, fn) -> None:
        problems.append(_problem(problem_id, label, lo, hi, fn))

    add("bench01_sphere", "Sphere", *_bounds(-5.12, 5.12, 2), _sphere)
    add("bench02_ellipsoid", "Ellipsoid", *_bounds(-5.12, 5.12, 2), _ellipsoid)
    add("bench03_sum_different_powers", "Sum of Different Powers", *_bounds(-1.0, 1.0, 2), _sum_of_different_powers)
    add("bench04_zakharov", "Zakharov", *_bounds(-5.0, 10.0, 2), _zakharov)
    add("bench05_rosenbrock", "Rosenbrock", *_bounds(-2.048, 2.048, 2), _rosenbrock)
    add("bench06_step", "Step", *_bounds(-5.12, 5.12, 2), _step)
    add("bench07_quartic", "Quartic", *_bounds(-1.28, 1.28, 2), _quartic)
    add("bench08_schwefel_222", "Schwefel 2.22", *_bounds(-10.0, 10.0, 2), _schwefel_222)
    add("bench09_schwefel_12", "Schwefel 1.2", *_bounds(-100.0, 100.0, 2), _schwefel_12)
    add("bench10_schwefel_221", "Schwefel 2.21", *_bounds(-100.0, 100.0, 2), _schwefel_221)
    add("bench11_rastrigin", "Rastrigin", *_bounds(-5.12, 5.12, 2), _rastrigin)
    add("bench12_ackley", "Ackley", *_bounds(-32.768, 32.768, 2), _ackley)
    add("bench13_griewank", "Griewank", *_bounds(-600.0, 600.0, 2), _griewank)
    add("bench14_levy", "Levy", *_bounds(-10.0, 10.0, 2), _levy)
    add("bench15_michalewicz", "Michalewicz", *_bounds(0.0, np.pi, 2), _michalewicz)
    add("bench16_alpine1", "Alpine 1", *_bounds(-10.0, 10.0, 2), _alpine1)
    add("bench17_alpine2", "Alpine 2", *_bounds(0.0, 10.0, 2), _alpine2)
    add("bench18_bent_cigar", "Bent Cigar", *_bounds(-10.0, 10.0, 3), _bent_cigar)
    add("bench19_discus", "Discus", *_bounds(-10.0, 10.0, 3), _discus)
    add("bench20_weierstrass", "Weierstrass", *_bounds(-0.5, 0.5, 4), _weierstrass)
    add("bench21_happycat", "HappyCat", *_bounds(-10.0, 10.0, 4), _happycat)
    add("bench22_hgbat", "HGBat", *_bounds(-10.0, 10.0, 4), _hgbat)
    add("bench23_qing", "Qing", *_bounds(-500.0, 500.0, 4), _qing)
    add("bench24_salomon", "Salomon", *_bounds(-100.0, 100.0, 5), _salomon)
    add("bench25_bohachevsky", "Bohachevsky", *_bounds(-100.0, 100.0, 5), _bohachevsky)
    add("bench26_booth", "Booth", *_bounds(-10.0, 10.0, 5), _booth)
    add("bench27_matyas", "Matyas", *_bounds(-10.0, 10.0, 10), _matyas)
    add("bench28_three_hump_camel", "Three-hump Camel", *_bounds(-5.0, 5.0, 10), _three_hump_camel)
    add("bench29_six_hump_camel", "Six-hump Camel", *_bounds(-5.0, 5.0, 10), _six_hump_camel)
    add("bench30_goldstein_price", "Goldstein-Price", *_bounds(-2.0, 2.0, 10), _goldstein_price)
    add("bench31_branin", "Branin", *_bounds(-5.0, 10.0, 30), _branin)
    add("bench32_shubert", "Shubert", *_bounds(-10.0, 10.0, 30), _shubert)
    add("bench33_himmelblau", "Himmelblau", *_bounds(-6.0, 6.0, 30), _himmelblau)
    add("bench34_easom", "Easom", *_bounds(-100.0, 100.0, 30), _easom)
    add("bench35_cross_in_tray", "Cross-in-Tray", *_bounds(-10.0, 10.0, 30), _cross_in_tray)
    add("bench36_holder_table", "Holder Table", *_bounds(-10.0, 10.0, 30), _holder_table)

    return problems
