import numpy as np
from scipy import stats as sp_stats

def safeee(val):
    """cast to float -- nan if not finite"""
    return float(val) if np.isfinite(val) else float("nan")


def entropy_from_histtt(x, bins=64):
    """entropy via histogram -- avoid zero probs"""
    hist, _ = np.histogram(x.flatten(), bins=bins, density=True)
    hist = hist[hist>0]
    if hist.size==0:
        return float("nan")
    hist = hist / hist.sum()
    return safeee(-np.sum(hist*np.log2(hist +1e-12)))



#group1
def basic_latent_stats(z0_stack):
    """global latent stats -- mean std skew etc"""
    flat = z0_stack.flatten().astype(np.float64)
    norms=np.linalg.norm(z0_stack.reshape(len(z0_stack), -1), axis=1)
    return {
        "latent_mean": safeee(np.mean(flat)),
        "latent_std": safeee(np.std(flat)),
        "latent_variance": safeee(np.var(flat)),
        "latent_skewness": safeee(sp_stats.skew(flat)),
        "latent_kurtosis": safeee(sp_stats.kurtosis(flat)),
        "latent_l2_norm": safeee(np.mean(norms)),
    }



#group2
def per_channel_stats(z0_stack):
    """per-channel stats -- mean variance energy"""
    feats={}
    for c in range(4):
        ch = z0_stack[:, c].flatten().astype(np.float64)
        feats[f"ch{c}_mean"] = safeee(np.mean(ch))
        feats[f"ch{c}_variance"]=safeee(np.var(ch))
        feats[f"ch{c}_energy"]=safeee(np.mean(ch**2))
    return feats



#group3
def spatial_statistics(z0_stack):
    """spatial structure -- entropy autocorr gradients"""
    spatial = z0_stack.mean(axis=(0, 1)).astype(np.float64)
    s_entropy = entropy_from_histtt(spatial)
    s_var=safeee(np.var(spatial))

    h, w = spatial.shape
    if h>2 and w>2:
        shifted_h = spatial[1:, :]*spatial[:-1, :]
        shifted_w=spatial[:, 1:] * spatial[:, :-1]
        autocorr = (np.mean(shifted_h) + np.mean(shifted_w))/2.0
        autocorr=safeee(autocorr / (np.var(spatial)+1e-12))
    else:
        autocorr=float("nan")
    grad_h = np.diff(z0_stack.astype(np.float64), axis=2)
    grad_w = np.diff(z0_stack.astype(np.float64), axis=3)
    min_h=min(grad_h.shape[2], grad_w.shape[2])
    min_w = min(grad_h.shape[3], grad_w.shape[3])
    grad_mag = np.sqrt(grad_h[:, :, :min_h, :min_w]**2 +grad_w[:, :, :min_h, :min_w] **2)

    return {
        "spatial_entropy": s_entropy,
        "spatial_variance": s_var,
        "spatial_autocorrelation": autocorr,
        "gradient_magnitude_mean": safeee(np.mean(grad_mag)),
        "gradient_magnitude_std": safeee(np.std(grad_mag)),
    }



#group4
def frequency_features(z0_stack):
    """frequency features via fft -- energy spectrum slope"""
    spatial = z0_stack.mean(axis=0).mean(axis=0).astype(np.float64)
    h, w=spatial.shape
    F = np.fft.fftshift(np.fft.fft2(spatial-spatial.mean()))
    psd = np.abs(F) **2
    total_energy=safeee(np.sum(psd))

    cy, cx = h//2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y-cy)**2 + (X - cx)**2)
    r_max=r.max()
    r_norm = r/(r_max + 1e-12)

    lf_mask = r_norm<0.25
    hf_mask=r_norm>0.5
    lf_energy = np.sum(psd[lf_mask])
    hf_energy=np.sum(psd[hf_mask])
    hf_ratio = safeee(hf_energy/(total_energy+1e-12))
    lf_ratio=safeee(lf_energy / (total_energy + 1e-12))

    psd_flat = psd.flatten()
    psd_norm = psd_flat / (psd_flat.sum()+1e-12)
    psd_norm = psd_norm[psd_norm>0]
    spec_entropy = safeee(-np.sum(psd_norm*np.log2(psd_norm + 1e-12)))

    nbins=min(30, max(h, w)//2)
    bins = np.linspace(0, r_max, nbins+1)
    radii = 0.5*(bins[:-1]+bins[1:])
    psd_radial = np.zeros(nbins)
    counts=np.zeros(nbins)
    idx = np.clip(np.digitize(r.flatten(), bins)-1, 0, nbins-1)
    np.add.at(psd_radial, idx, psd.flatten())
    np.add.at(counts, idx, 1.0)
    psd_radial /= (counts + 1e-12)

    valid = (radii>0) & (psd_radial>0)
    if valid.sum()>=3:
        log_r = np.log(radii[valid])
        log_p=np.log(psd_radial[valid])
        A = np.vstack([log_r, np.ones_like(log_r)]).T
        slope = np.linalg.lstsq(A, log_p, rcond=None)[0][0]
        spec_slope=safeee(slope)
    else:
        spec_slope = float("nan")
    return {
        "fft_energy_total": safeee(total_energy),
        "high_freq_energy_ratio": hf_ratio,
        "low_freq_energy_ratio": lf_ratio,
        "spectral_entropy": spec_entropy,
        "spectral_slope": spec_slope,
    }



#group5
def noise_residual_features(eps_stack):
    """noise residual stats -- check gaussian-like noise"""
    nan_result = {
        "noise_residual_mean": float("nan"),
        "noise_residual_std": float("nan"),
        "noise_residual_energy": float("nan"),
        "noise_spatial_correlation": float("nan"),
    }

    if eps_stack is None or not np.isfinite(eps_stack).any():
        return nan_result
    flat = eps_stack.flatten().astype(np.float64)
    finite=flat[np.isfinite(flat)]
    if finite.size==0:
        return nan_result

    eps_mean=np.nanmean(eps_stack, axis=(0, 1)).astype(np.float64)
    h, w = eps_mean.shape
    if h>2 and w>2:
        shifted = eps_mean[1:, :]*eps_mean[:-1, :]
        var_eps=np.var(eps_mean)
        spatial_corr = safeee(np.mean(shifted)/(var_eps +1e-12)) if np.isfinite(var_eps) else float("nan")
    else:
        spatial_corr=float("nan")

    return {
        "noise_residual_mean": safeee(np.mean(finite)),
        "noise_residual_std": safeee(np.std(finite)),
        "noise_residual_energy": safeee(np.mean(finite **2)),
        "noise_spatial_correlation": spatial_corr,
    }



#group6
def patch_consistency_features(z0_stack, patch_size=4):
    """patch consistency -- variance similarity entropy"""
    T, C, H, W = z0_stack.shape
    ph, pw=patch_size, patch_size
    patch_vars = []
    patch_means = []

    for t in range(T):
        for i in range(0, H - ph + 1, ph):
            for j in range(0, W - pw + 1, pw):
                patch = z0_stack[t, :, i:i +ph, j:j+pw].flatten().astype(np.float64)
                patch_vars.append(np.var(patch))
                patch_means.append(np.mean(patch))

    patch_vars=np.array(patch_vars)
    patch_means=np.array(patch_means)
    if patch_vars.size<2:
        return {
            "patch_variance_mean": float("nan"),
            "patch_variance_std": float("nan"),
            "patch_similarity": float("nan"),
            "patch_entropy": float("nan"),
        }

    diffs = np.abs(patch_means[:, None] - patch_means[None, :])
    np.fill_diagonal(diffs, np.nan)
    patch_sim = safeee(1.0/(np.nanmean(diffs) + 1e-12))

    return {
        "patch_variance_mean": safeee(np.mean(patch_vars)),
        "patch_variance_std": safeee(np.std(patch_vars)),
        "patch_similarity": patch_sim,
        "patch_entropy": entropy_from_histtt(patch_means),
    }



#group7
def temporal_features(z0_stack):
    """temporal dynamics -- velocity acceleration smoothness """
    T = z0_stack.shape[0]
    if T<2:
        return {k: float("nan") for k in ["latent_frame_distance_mean", "latent_frame_distance_std", "latent_velocity_mean", "latent_velocity_std", "latent_acceleration_mean", "latent_acceleration_std", "latent_temporal_entropy", "latent_jerk_mean", "latent_smoothness", "normalized_temporal_variance", "trajectory_curvature_consistency"]}

    flat = z0_stack.reshape(T, -1).astype(np.float64)
    dists=np.linalg.norm(np.diff(flat, axis=0), axis=1)
    velocity=dists

    if T>=3:
        accel = np.diff(velocity)
        accel_mean = safeee(np.mean(np.abs(accel)))
        accel_std=safeee(np.std(accel))
    else:
        accel_mean = float("nan")
        accel_std=float("nan")
    temp_entropy = entropy_from_histtt(dists, bins=min(16, max(T // 2, 2)))

    if T>=4:
        accel = np.diff(velocity)
        jerk = np.diff(accel)
        jerk_mean=safeee(np.mean(np.abs(jerk)))
    else:
        jerk_mean = float("nan")

    vel_std = np.std(velocity)
    latent_smoothness = safeee(np.mean(velocity)/(vel_std+1e-12)) if vel_std>0 else float("nan")
    mean_d=np.mean(dists)
    norm_temp_var = safeee(np.var(dists)/(mean_d**2 +1e-12)) if mean_d>0 else float("nan")
    displacements = np.diff(flat, axis=0)
    if T>=3:
        cos_angles = []
        for i in range(len(displacements) - 1):
            d1 = displacements[i]
            d2 = displacements[i + 1]
            n1 = np.linalg.norm(d1)
            n2=np.linalg.norm(d2)
            if n1>1e-12 and n2>1e-12:
                cos_angles.append(np.dot(d1, d2)/(n1*n2))
        if cos_angles:
            curvature_consistency = safeee(np.std(cos_angles))
        else:
            curvature_consistency=float("nan")
    else:
        curvature_consistency = float("nan")

    return {
        "latent_frame_distance_mean": safeee(np.mean(dists)),
        "latent_frame_distance_std": safeee(np.std(dists)),
        "latent_velocity_mean": safeee(np.mean(velocity)),
        "latent_velocity_std": safeee(np.std(velocity)),
        "latent_acceleration_mean": accel_mean,
        "latent_acceleration_std": accel_std,
        "latent_temporal_entropy": temp_entropy,
        "latent_jerk_mean": jerk_mean,
        "latent_smoothness": latent_smoothness,
        "normalized_temporal_variance": norm_temp_var,
        "trajectory_curvature_consistency": curvature_consistency,
    }



#group8
def correlation_features(z0_stack):
    """inter-channel relations -- corr mi covariance"""
    T, C, H, W = z0_stack.shape
    ch_flat=z0_stack.reshape(T, C, -1).astype(np.float64)
    corrs = []
    mis=[]
    cov_traces = []

    for t in range(T):
        cc = np.corrcoef(ch_flat[t])
        upper = cc[np.triu_indices(C, k=1)]
        corrs.append(upper)
        cov=np.cov(ch_flat[t])
        cov_traces.append(np.trace(cov))

        mi_pairs=[]
        for i in range(C):
            for j in range(i+1, C):
                hist_2d, _, _ = np.histogram2d(ch_flat[t, i], ch_flat[t, j], bins=32)
                pxy = hist_2d / (hist_2d.sum() + 1e-12)
                px = pxy.sum(axis=1)
                py = pxy.sum(axis=0)
                mask = pxy>0
                mi = np.sum(pxy[mask] * np.log2(pxy[mask] / (px[:, None] * py[None, :] + 1e-12)[mask] + 1e-12))
                mi_pairs.append(mi)
        mis.append(np.mean(mi_pairs))

    corrs = np.concatenate(corrs)
    return {
        "channel_correlation_mean": safeee(np.mean(corrs)),
        "channel_correlation_std": safeee(np.std(corrs)),
        "channel_mutual_information": safeee(np.mean(mis)),
        "covariance_trace": safeee(np.mean(cov_traces)),
    }



#group9
def manifold_features(z0_stack):
    """latent space geometry -- mahalanobis density nn dist"""
    T = z0_stack.shape[0]
    flat = z0_stack.reshape(T, -1).astype(np.float64)

    if T<3:
        return {
            "mahalanobis_distance": float("nan"),
            "latent_density_score": float("nan"),
            "nearest_neighbor_distance": float("nan"),
        }

    mu=flat.mean(axis=0)
    var = flat.var(axis=0) + 1e-12
    maha_dists=np.sqrt(np.mean((flat-mu)**2 / var, axis=1))

    pw_dists = np.zeros((T, T))
    for i in range(T):
        for j in range(i+1, T):
            d = np.linalg.norm(flat[i] - flat[j])
            pw_dists[i, j] = d
            pw_dists[j, i] = d

    np.fill_diagonal(pw_dists, np.inf)
    nn_dists = pw_dists.min(axis=1)

    np.fill_diagonal(pw_dists, 0)
    mean_pw = np.mean(pw_dists[np.triu_indices(T, k=1)])
    density=1.0/(mean_pw+1e-12)

    return {
        "mahalanobis_distance": safeee(np.mean(maha_dists)),
        "latent_density_score": safeee(density),
        "nearest_neighbor_distance": safeee(np.mean(nn_dists)),
    }



#group10
def object_latent_features(z0_stack, masks_per_frame):
    """object-level stats -- variance entropy motion consistency"""
    nan_result = {
        "object_latent_variance": float("nan"),
        "object_latent_entropy": float("nan"),
        "object_latent_temporal_variance": float("nan"),
        "object_latent_motion_consistency": float("nan"),
        "object_motion_smoothness": float("nan"),
    }

    if masks_per_frame is None or len(masks_per_frame)==0:
        return nan_result
    T, C, H, W=z0_stack.shape

    obj_variances = []
    obj_entropies = []
    obj_temporal={}
    for t in range(min(T, len(masks_per_frame))):
        frame_masks = masks_per_frame[t]
        if not frame_masks:
            continue

        for obj_id, mask in frame_masks.items():
            if mask.shape != (H, W):
                from cv2 import resize, INTER_NEAREST
                mask = resize(mask.astype(np.uint8), (W, H), interpolation=INTER_NEAREST).astype(bool)

            if mask.sum()<4:
                continue

            vals = z0_stack[t][:, mask].flatten().astype(np.float64)
            obj_variances.append(np.var(vals))
            obj_entropies.append(entropy_from_histtt(vals, bins=32))

            if obj_id not in obj_temporal:
                obj_temporal[obj_id] = []
            obj_temporal[obj_id].append(np.mean(vals))

    if not obj_variances:
        return nan_result

    temp_vars = []
    motion_consistencies=[]
    obj_smoothnesses = []
    for obj_id, series in obj_temporal.items():
        if len(series)>=2:
            s=np.array(series)
            temp_vars.append(np.var(s))
            diffs = np.diff(s)
            if len(diffs)>=2:
                motion_consistencies.append(np.std(diffs)/(np.mean(np.abs(diffs))+1e-12))
                abs_diffs = np.abs(diffs)
                d_std = np.std(abs_diffs)
                if d_std>0:
                    obj_smoothnesses.append(np.mean(abs_diffs)/(d_std +1e-12))

    return {
        "object_latent_variance": safeee(np.mean(obj_variances)),
        "object_latent_entropy": safeee(np.mean(obj_entropies)),
        "object_latent_temporal_variance": safeee(np.mean(temp_vars)) if temp_vars else float("nan"),
        "object_latent_motion_consistency": safeee(np.mean(motion_consistencies)) if motion_consistencies else float("nan"),
        "object_motion_smoothness": safeee(np.mean(obj_smoothnesses)) if obj_smoothnesses else float("nan"),
    }
