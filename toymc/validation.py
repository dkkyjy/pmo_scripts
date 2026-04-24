"""
统计校验与分布测试工具
"""
import numpy as np

VALIDATION_MIN_SAMPLE_SIZE = 20
VALIDATION_KS_PVALUE_THRESHOLD = 0.01
VALIDATION_KDE_MAX_SAMPLE_SIZE = 100_000


def _ks_pvalue_from_lambda(lam):
    if lam <= 0:
        return 1.0
    series = 0.0
    for k in range(1, 101):
        series += ((-1) ** (k - 1)) * np.exp(-2.0 * (k ** 2) * (lam ** 2))
    return float(max(0.0, min(1.0, 2.0 * series)))


def ks_test_exponential(intervals):
    values = np.asarray(intervals, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size < VALIDATION_MIN_SAMPLE_SIZE:
        return {"status": "INSUFFICIENT_DATA", "n": int(values.size)}
    if np.allclose(values, values[0]):
        return {"status": "INSUFFICIENT_DATA", "n": int(values.size)}
    sorted_values = np.sort(values)
    n_samples = sorted_values.size
    scale = float(np.mean(sorted_values))
    if scale <= 0:
        return {"status": "INSUFFICIENT_DATA", "n": int(n_samples)}
    cdf_theory = 1.0 - np.exp(-sorted_values / scale)
    cdf_upper = np.arange(1, n_samples + 1, dtype=float) / n_samples
    cdf_lower = np.arange(0, n_samples, dtype=float) / n_samples
    d_stat = float(max(np.max(np.abs(cdf_upper - cdf_theory)), np.max(np.abs(cdf_theory - cdf_lower))))
    lam = (np.sqrt(n_samples) + 0.12 + 0.11 / np.sqrt(n_samples)) * d_stat
    p_value = _ks_pvalue_from_lambda(lam)
    return {"status": "OK", "n": int(n_samples), "d_stat": d_stat, "p_value": float(p_value), "scale": scale}


def record_direction_sample(validation_state, source, r_m, theta_deg, phi_deg):
    if validation_state is None:
        return
    prefix = "signal" if source == "signal" else "cosmic"
    validation_state[f"{prefix}_r_m"].append(float(r_m))
    validation_state[f"{prefix}_theta_deg"].append(float(theta_deg))
    validation_state[f"{prefix}_phi_deg"].append(float(phi_deg))
