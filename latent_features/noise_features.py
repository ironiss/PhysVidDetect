import numpy as np
from scipy import stats as sp_stats


def safeee(val):
    return float(val) if np.isfinite(val) else float("nan")


def entropy_from_histt(x, bins=32):
    hist, _ = np.histogram(x.flatten(), bins=bins, density=True)
    hist = hist[hist>0]
    if hist.size==0:
        return float("nan")
    hist = hist/hist.sum()
    return safeee(-np.sum(hist*np.log2(hist +1e-12)))



#group1
def temporal_noise_consistency(eps_stack):
    """eps diffs between frames -- stable vs jumpy noise"""
    T = eps_stack.shape[0]
    nan_result = {k: float("nan") for k in [
        "eps_diff_norm_mean", "eps_diff_norm_std",
        "eps_diff_norm_skew", "eps_diff_norm_kurt",
        "eps_diff_cosine_mean", "eps_diff_cosine_std",
        "eps_diff_energy_mean", "eps_diff_entropy",
    ]}

    if T<2 or not np.isfinite(eps_stack).all():
        return nan_result
    flat = eps_stack.reshape(T, -1).astype(np.float64)
    diffs = flat[1:]-flat[:-1]
    diff_norms = np.linalg.norm(diffs, axis=1)

    if T>=3:
        cos_vals = []
        for i in range(len(diffs)-1):
            n1 = np.linalg.norm(diffs[i])
            n2 = np.linalg.norm(diffs[i+1])
            if n1>1e-12 and n2>1e-12:
                cos_vals.append(np.dot(diffs[i], diffs[i+1])/(n1*n2))
        cos_vals = np.array(cos_vals) if cos_vals else np.array([0.0])
        cos_mean = safeee(np.mean(cos_vals))
        cos_std = safeee(np.std(cos_vals))
    else:
        cos_mean = float("nan")
        cos_std = float("nan")

    diff_energy = np.mean(diffs **2, axis=1)
    if len(diff_norms)>=4:
        skew = safeee(sp_stats.skew(diff_norms))
        kurt = safeee(sp_stats.kurtosis(diff_norms))
    else:
        skew = float("nan")
        kurt = float("nan")

    return {
        "eps_diff_norm_mean": safeee(np.mean(diff_norms)),
        "eps_diff_norm_std": safeee(np.std(diff_norms)),
        "eps_diff_norm_skew": skew,
        "eps_diff_norm_kurt": kurt,
        "eps_diff_cosine_mean": cos_mean,
        "eps_diff_cosine_std": cos_std,
        "eps_diff_energy_mean": safeee(np.mean(diff_energy)),
        "eps_diff_entropy": entropy_from_histt(diff_norms, bins=min(16, max(T//2, 2))),
    }



#group 2
def noise_trajectory_stability(eps_stack):
    """motion of eps in latent space -- smooth or chaotic"""
    T = eps_stack.shape[0]
    nan_result = {k: float("nan") for k in [
        "eps_velocity_mean", "eps_velocity_std",
        "eps_acceleration_mean", "eps_acceleration_std",
        "eps_jerk_mean",
        "eps_smoothness", "eps_normalized_temporal_var",
    ]}

    if T<2 or not np.isfinite(eps_stack).all():
        return nan_result
    flat = eps_stack.reshape(T, -1).astype(np.float64)
    velocity = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    if T>=3:
        accel = np.diff(velocity)
        accel_mean = safeee(np.mean(np.abs(accel)))
        accel_std = safeee(np.std(accel))
    else:
        accel_mean = float("nan")
        accel_std = float("nan")
    if T>=4:
        jerk = np.diff(np.diff(velocity))
        jerk_mean = safeee(np.mean(np.abs(jerk)))
    else:
        jerk_mean = float("nan")

    vel_std = np.std(velocity)
    smoothness = safeee(np.mean(velocity)/(vel_std+1e-12)) if vel_std>0 else float("nan")
    mean_v = np.mean(velocity)
    norm_var = safeee(np.var(velocity)/(mean_v**2 +1e-12)) if mean_v>0 else float("nan")

    return {
        "eps_velocity_mean": safeee(np.mean(velocity)),
        "eps_velocity_std": safeee(np.std(velocity)),
        "eps_acceleration_mean": accel_mean,
        "eps_acceleration_std": accel_std,
        "eps_jerk_mean": jerk_mean,
        "eps_smoothness": smoothness,
        "eps_normalized_temporal_var": norm_var,
    }



#group 3
def noise_frequency_features(eps_stack):
    """freq behavior of eps -- temporal+spatial hf stuff"""
    T = eps_stack.shape[0]
    nan_result = {k: float("nan") for k in [
        "eps_temporal_spectral_slope",
        "eps_temporal_hf_energy_ratio",
        "eps_temporal_spectral_entropy",
        "eps_spatial_hf_ratio_mean",
        "eps_spatial_hf_ratio_std",
        "eps_spatial_hf_ratio_temporal_var",
    ]}

    if T<3 or not np.isfinite(eps_stack).all():
        return nan_result
    flat = eps_stack.reshape(T, -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1)
    y = norms-norms.mean()
    if len(y)>=4:
        Y = np.fft.rfft(y)
        P = np.abs(Y)**2
        P_copy = P.copy()
        P_copy[0] = 0.0
        total = P_copy.sum()+1e-12
        split = max(2, int(0.35*len(P_copy)))
        hf_energy = P_copy[split:].sum()
        hf_ratio = safeee(hf_energy/total)

        pn = P_copy/total
        pn = pn[pn>0]
        spec_entropy = safeee(-np.sum(pn*np.log2(pn +1e-12)))

        freqs = np.fft.rfftfreq(len(y), d=1.0)[1:]
        p = P_copy[1:]
        mask = (p>1e-12) & (freqs>0)
        if mask.sum()>=3:
            Xl = np.log(freqs[mask]+1e-12)
            Yl = np.log(p[mask] + 1e-12)
            A = np.vstack([Xl, np.ones_like(Xl)]).T
            slope = np.linalg.lstsq(A, Yl, rcond=None)[0][0]
            spec_slope = safeee(slope)
        else:
            spec_slope = float("nan")
    else:
        hf_ratio = float("nan")
        spec_entropy = float("nan")
        spec_slope = float("nan")

    hf_ratios_per_frame = []
    for t in range(T):
        spatial = eps_stack[t].mean(axis=0).astype(np.float64)
        h, w = spatial.shape
        if h<4 or w<4:
            continue

        F = np.fft.fftshift(np.fft.fft2(spatial-spatial.mean()))
        psd = np.abs(F) **2
        total_e = psd.sum()+1e-12

        cy, cx = h//2, w // 2
        Y_grid, X_grid = np.ogrid[:h, :w]
        r = np.sqrt((Y_grid-cy)**2 + (X_grid - cx)**2)
        r_norm = r/(r.max()+1e-12)

        hf_mask = r_norm>0.5
        hf_ratios_per_frame.append(psd[hf_mask].sum()/total_e)

    hf_arr = np.array(hf_ratios_per_frame)
    if len(hf_arr)>=2:
        spatial_hf_mean = safeee(np.mean(hf_arr))
        spatial_hf_std = safeee(np.std(hf_arr))
        spatial_hf_tvar = safeee(np.var(hf_arr)/(np.mean(hf_arr)**2 +1e-12))
    else:
        spatial_hf_mean = float("nan")
        spatial_hf_std = float("nan")
        spatial_hf_tvar = float("nan")

    return {
        "eps_temporal_spectral_slope": spec_slope,
        "eps_temporal_hf_energy_ratio": hf_ratio,
        "eps_temporal_spectral_entropy": spec_entropy,
        "eps_spatial_hf_ratio_mean": spatial_hf_mean,
        "eps_spatial_hf_ratio_std": spatial_hf_std,
        "eps_spatial_hf_ratio_temporal_var": spatial_hf_tvar,
    }
