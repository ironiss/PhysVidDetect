import cv2
import numpy as np
from helpers import safe_div, sobel_mag_theta, make_ring, resize_mask, lap_var, iou, circ_mean_R, circ_diff, nanstd, nanmedian, zscore, basic_stats, diff_stats, highpass, crop_bbox, radial_psd_slope

PHYSICS_FEATURE_KEYS=[
    #group 2 
    # inter-object consistency
    "inter_obj_corr_median", "inter_obj_corr_mean", "inter_obj_corr_min", "inter_obj_pairs",
    # boundary orientation phase noise
    "orient_phase_noise_objmedian", "orient_phase_noise_objspread", "orient_R_median_objmedian", "orient_R_median_objspread",
    #edge-flow alignment
    "edgeflow_align_median_objmedian", "edgeflow_align_median_objspread", "edgeflow_align_diff_std_objmedian", "edgeflow_align_diff_std_objspread",
    #inside-outside contrast
    "inside_out_dI_diffz_std_objmedian", "inside_out_dI_diffz_std_objspread", "inside_out_dG_diffz_std_objmedian", "inside_out_dG_diffz_std_objspread",
    
    #group 3
    # HF noise variance ratio
    "hf_var_ratio_median_objmedian", "hf_var_ratio_median_objspread", "hf_var_ratio_std_objmedian", "hf_var_ratio_std_objspread",
    #PSD slope difference
    "psd_slope_diff_median_objmedian", "psd_slope_diff_median_objspread", "psd_slope_diff_std_objmedian", "psd_slope_diff_std_objspread",
    #valid fraction
    "valid_frac_objmedian", "valid_frac_objspread"
]



#group1 (actually we do not use it, but for experiments can be added)
def extract_features_from_tracks(tracks, frame_paths, fps, ring_r=3, grad_top_pct=80.0, bg_ring_r=10):
    """per-object motion + shape over time (speed area masks gradients)"""
    per_obj_feats, track_lengths = [], []
    num_objs = 0

    for _, seq in tracks.items():
        if not seq:
            continue
        
        seq = sorted(seq, key=lambda d: int(d["t"]))
        num_objs += 1
        track_lengths.append(len(seq))

        cx = np.array([float(d["centroid"][0]) for d in seq], dtype=np.float32)
        cy = np.array([float(d["centroid"][1]) for d in seq], dtype=np.float32)
        area = np.array([float(d.get("area", np.nan)) for d in seq], dtype=np.float32)
        perim = np.array([float(d.get("perimeter", np.nan)) for d in seq], dtype=np.float32)
        
        aspect = np.array([max(1., float(d["bbox"][2] - d["bbox"][0] + 1)) / max(1., float(d["bbox"][3] - d["bbox"][1] + 1)) for d in seq], dtype=np.float32)
        circularity = (4.0*np.pi*area) / (perim*perim + 1e-9)

        dx, dy = np.diff(cx), np.diff(cy)
        speed = np.sqrt(dx*dx + dy*dy)
        accel = np.diff(speed)
        jerk = np.diff(accel)

        ious, comp_counts, holes_counts = [], [], []
        for k in range(1, len(seq)):
            ious.append(iou(seq[k - 1]["mask"], seq[k]["mask"]))
            m_u8 = seq[k]["mask"].astype(np.uint8)
            n_cc, _ = cv2.connectedComponents(m_u8)
            comp_counts.append(float(max(0, n_cc - 1)))
            x1, y1, x2, y2 = [int(v) for v in seq[k]["bbox"]]
            roi = m_u8[max(0, y1): y2 + 1, max(0, x1): x2 + 1]
            if roi.size < 50:
                holes_counts.append(float("nan"))
            else:
                n_bg, _ = cv2.connectedComponents((roi == 0).astype(np.uint8))
                holes_counts.append(float(max(0, n_bg-1)))

        ious = np.array(ious, dtype=np.float32)
        comp_counts = np.array(comp_counts, dtype=np.float32)
        holes_counts = np.array(holes_counts, dtype=np.float32)

        edge_mean, edge_std, edge_R = [], [], []
        grad_res_bnd, grad_obj_mean = [], []
        sharp_obj, sharp_bg, sharp_ratio = [], [], []
        prev_mag, prev_ring = None, None

        for k, d in enumerate(seq):
            fi = int(d["t"])
            nan_entry = lambda: (
                edge_mean.append(float("nan")), edge_std.append(float("nan")),
                edge_R.append(float("nan")), grad_res_bnd.append(float("nan")),
                grad_obj_mean.append(float("nan")), sharp_obj.append(float("nan")),
                sharp_bg.append(float("nan")), sharp_ratio.append(float("nan")),
            )
            if fi < 0 or fi >= len(frame_paths):
                nan_entry()
                prev_mag, prev_ring = None, None
                continue

            frame = cv2.imread(frame_paths[fi], cv2.IMREAD_COLOR)
            if frame is None:
                nan_entry()
                prev_mag, prev_ring = None, None
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = resize_mask(d["mask"].astype(bool), gray)
            ring = make_ring(mask,r=ring_r)
            bg_ring = make_ring(mask,r=bg_ring_r) & (~mask)
            mag, theta = sobel_mag_theta(gray)

            vals = mag[ring]
            if vals.size < 20:
                edge_mean.append(float("nan"))
                edge_std.append(float("nan"))
                edge_R.append(float("nan"))
            else:
                edge_mean.append(float(np.mean(vals)))
                edge_std.append(float(np.std(vals)))
                thr = np.percentile(vals, grad_top_pct)
                sel = ring & (mag >= thr)
                _, R = circ_mean_R(theta[sel], mag[sel])
                edge_R.append(R)

            obj_vals = mag[mask]
            grad_obj_mean.append(float(np.mean(obj_vals)) if obj_vals.size >= 20 else float("nan"))

            if prev_mag is not None and prev_ring is not None:
                v = np.abs(mag-prev_mag)[ring]
                grad_res_bnd.append(float(np.mean(v)) if v.size >= 20 else float("nan"))
            else:
                grad_res_bnd.append(float("nan"))

            s_obj = lap_var(gray, mask)
            s_bg  = lap_var(gray, bg_ring) if bg_ring.sum() > 50 else float("nan")
            sharp_obj.append(s_obj)
            sharp_bg.append(s_bg)
            sharp_ratio.append(safe_div(s_obj, s_bg) if np.isfinite(s_obj) and np.isfinite(s_bg) else float("nan"))
            prev_mag, prev_ring = mag, ring

        edge_mean = np.array(edge_mean, dtype=np.float32)
        edge_std = np.array(edge_std, dtype=np.float32)
        edge_R = np.array(edge_R, dtype=np.float32)
        grad_res_bnd = np.array(grad_res_bnd, dtype=np.float32)
        grad_obj_mean = np.array(grad_obj_mean, dtype=np.float32)
        sharp_obj = np.array(sharp_obj, dtype=np.float32)
        sharp_bg= np.array(sharp_bg, dtype=np.float32)
        sharp_ratio = np.array(sharp_ratio, dtype=np.float32)

        f = {
            "track_len": float(len(seq)),
            "track_seconds": float(len(seq) / max(1e-9, fps)),
        }

        for name, values in [
            ("speed", speed),
            ("accel", accel),
            ("jerk", jerk),
            ("area", area),
            ("perim", perim),
            ("circ", circularity),
            ("aspect", aspect),
            ("iou", ious),
            ("cc", comp_counts),
            ("holes", holes_counts),
            ("edge_mean", edge_mean),
            ("edge_std", edge_std),
            ("edge_R", edge_R),
            ("gradres_bnd", grad_res_bnd),
            ("grad_obj", grad_obj_mean),
            ("sharp_obj", sharp_obj),
            ("sharp_bg", sharp_bg),
            ("sharp_ratio", sharp_ratio),
        ]: f.update({f"{name}_{k}": v for k, v in basic_stats(values).items()})

        for name, values in [
            ("speed", speed),
            ("area", area),
            ("circ", circularity),
            ("edge_mean", edge_mean),
            ("grad_obj", grad_obj_mean),
            ("sharp_ratio", sharp_ratio),
        ]: f.update({f"{name}_{k}": v for k, v in diff_stats(values).items()})

        f["iou_low_frac"] = float(np.nanmean(ious<0.4)) if ious.size else float("nan")

        per_obj_feats.append(f)

    out={"num_objects": float(num_objs), "track_len_mean": float(np.mean(track_lengths)) if track_lengths else float("nan"), "track_len_std":  float(np.std(track_lengths))  if track_lengths else float("nan")}
    if not per_obj_feats:
        return out
    
    all_keys = sorted({k for d in per_obj_feats for k in d})
    
    for k in all_keys:
        vals = np.array([d.get(k, np.nan) for d in per_obj_feats], dtype=np.float32)
        out[f"{k}__obj_mean"] = float(np.nanmean(vals))
        out[f"{k}__obj_std"] = float(np.nanstd(vals))
        out[f"{k}__obj_median"] = float(np.nanmedian(vals))
    return out



#group2
def boundary_edge_series(tracks, frame_paths, ring_r=3, min_vals=20):
    """build per-object edge strength over time (boundary gradients)"""
    out = {}
    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])
        if not seq:
            out[obj_id] = np.array([], dtype=float)
            continue

        tmax   = max(int(d["t"]) for d in seq)
        series = np.full((tmax + 1,), np.nan, dtype=float)
        for d in seq:
            t = int(d["t"])
            if t < 0 or t >= len(frame_paths): 
                continue

            frame = cv2.imread(frame_paths[t])
            if frame is None: 
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mag, _ = sobel_mag_theta(gray)
            mask = resize_mask(d["mask"], gray)
            ring = make_ring(mask, r=ring_r)
            vals = mag[ring]
            if vals.size >= min_vals:
                series[t] = float(vals.mean())
        out[obj_id] = series
    return out



def inter_object_consistency_features(tracks, frame_paths, ring_r=3):
    """edge energy correlation over time"""
    series_dict = boundary_edge_series(tracks, frame_paths, ring_r=ring_r)
    obj_ids = [o for o, s in series_dict.items() if s.size >= 5]

    if len(obj_ids) < 2:
        return {"inter_obj_corr_median": np.nan, "inter_obj_corr_mean": np.nan, "inter_obj_corr_min": np.nan, "inter_obj_pairs": 0}

    maxlen = max(series_dict[o].size for o in obj_ids)
    M = []
    for o in obj_ids:
        s = series_dict[o]
        pad = np.full((maxlen,), np.nan)
        pad[:s.size] = s
        dz = np.diff(zscore(pad))
        M.append(dz)
    M = np.stack(M, axis=0)

    cors = []
    for i in range(len(obj_ids)):
        for j in range(i+1, len(obj_ids)):
            a, b = M[i], M[j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 10: 
                continue

            c = np.corrcoef(a[ok], b[ok])[0, 1]
            if np.isfinite(c): 
                cors.append(float(c))

    if not cors:
        return {"inter_obj_corr_median": np.nan, "inter_obj_corr_mean": np.nan, "inter_obj_corr_min": np.nan, "inter_obj_pairs": 0}
    cors = np.array(cors)

    return {
        "inter_obj_corr_median": float(np.nanmedian(cors)),
        "inter_obj_corr_mean": float(np.nanmean(cors)),
        "inter_obj_corr_min": float(np.nanmin(cors)),
        "inter_obj_pairs": int(cors.size),
    }



def boundary_orientation_phase_noise_features(tracks, frame_paths, ring_r=3, min_vals=50):
    """boundary orientation drift"""
    per_obj = {}
    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])
        mus, Rs = [], []
        for d in seq:
            t = int(d["t"])
            if t<0 or t>=len(frame_paths): 
                continue

            frame = cv2.imread(frame_paths[t])
            if frame is None: 
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mag, theta = sobel_mag_theta(gray)
            mask = resize_mask(d["mask"], gray)
            ring = make_ring(mask, r=ring_r)
            th, w = theta[ring], mag[ring]
            if th.size < min_vals: 
                continue

            mu, R = circ_mean_R(th, w)
            if np.isfinite(mu): 
                mus.append(mu)
                Rs.append(R)

        if len(mus) < 2:
            per_obj[obj_id] = {"phase_noise": np.nan, "R_median": np.nan}
            continue

        dmu = np.array([circ_diff(mus[k], mus[k - 1]) for k in range(1, len(mus))])
        per_obj[obj_id] = {"phase_noise": nanstd(dmu), "R_median":  nanmedian(np.array(Rs))}

    pn = np.array([v["phase_noise"] for v in per_obj.values()], dtype=float)
    Rv = np.array([v["R_median"] for v in per_obj.values()], dtype=float)
    return {
        "orient_phase_noise_objmedian": nanmedian(pn),
        "orient_phase_noise_objspread": nanstd(pn),
        "orient_R_median_objmedian": nanmedian(Rv),
        "orient_R_median_objspread": nanstd(Rv),
    }



def edge_flow_alignment_features(tracks, frame_paths, ring_r=3, min_vals=50):
    """flow vs gradient alignment"""
    per_obj = {}
    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])

        if len(seq) < 2:
            per_obj[obj_id] = {"align_median": np.nan, "align_diff_std": np.nan}
            continue

        align_series = []
        for k in range(1, len(seq)):
            t0, t1 = int(seq[k - 1]["t"]), int(seq[k]["t"])

            if not (0 <= t0 < len(frame_paths) and 0 <= t1 < len(frame_paths)):
                align_series.append(np.nan)
                continue
            f0, f1 = cv2.imread(frame_paths[t0]), cv2.imread(frame_paths[t1])
            if f0 is None or f1 is None:
                align_series.append(np.nan)
                continue

            g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            phi = np.arctan2(flow[..., 1], flow[..., 0])

            mag, theta = sobel_mag_theta(g1)
            mask = resize_mask(seq[k]["mask"], g1)
            ring = make_ring(mask, r=ring_r)
            th, ph, w = theta[ring], phi[ring], mag[ring]

            if th.size < min_vals:
                align_series.append(np.nan)
                continue
            a = np.abs(np.cos(th - ph))
            align_series.append(float(np.sum(a * w) / (np.sum(w)+1e-12)))

        a = np.array(align_series, dtype=float)
        per_obj[obj_id] = {"align_median": nanmedian(a), "align_diff_std": nanstd(np.diff(a))}

    v1 = np.array([v["align_median"] for v in per_obj.values()], dtype=float)
    v2 = np.array([v["align_diff_std"] for v in per_obj.values()], dtype=float)

    return {
        "edgeflow_align_median_objmedian":   nanmedian(v1),
        "edgeflow_align_median_objspread":   nanstd(v1),
        "edgeflow_align_diff_std_objmedian": nanmedian(v2),
        "edgeflow_align_diff_std_objspread": nanstd(v2),
    }



def inside_outside_contrast_features(tracks, frame_paths, ring_r=3, bg_ring_r=10, min_vals=50):
    """brightness + gradient stability inside vs outside object"""
    per_obj = {}
    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])
        deltaI, deltaG = [], []

        for d in seq:
            t = int(d["t"])
            if t < 0 or t >= len(frame_paths): 
                continue

            frame = cv2.imread(frame_paths[t])
            if frame is None: 
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mag, _ = sobel_mag_theta(gray.astype(np.uint8))
            mask = resize_mask(d["mask"], gray)
            m = mask.astype(np.uint8)
            ki  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_r   + 1, 2 * ring_r + 1))
            kbg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bg_ring_r + 1, 2 * bg_ring_r + 1))
            ero_in  = cv2.erode(m, ki, iterations=1).astype(bool)
            dil_in  = cv2.dilate(m, ki, iterations=1).astype(bool)
            dil_bg  = cv2.dilate(m, kbg, iterations=1).astype(bool)
            inside  = mask & (~ero_in)
            outside = dil_bg & (~dil_in)
            if inside.sum() < min_vals or outside.sum() < min_vals: 
                continue

            deltaI.append(float(gray[inside].mean()- gray[outside].mean()))
            deltaG.append(float(mag[inside].mean() - mag[outside].mean()))

        dI = np.array(deltaI, dtype=float)
        dG = np.array(deltaG, dtype=float)
        if dI.size < 5:
            per_obj[obj_id] = {"dI": np.nan, "dG": np.nan}
            continue
        per_obj[obj_id] = {"dI": nanstd(np.diff(zscore(dI))), "dG": nanstd(np.diff(zscore(dG)))}

    v1 = np.array([v["dI"] for v in per_obj.values()], dtype=float)
    v2 = np.array([v["dG"] for v in per_obj.values()], dtype=float)

    return {
        "inside_out_dI_diffz_std_objmedian": nanmedian(v1),
        "inside_out_dI_diffz_std_objspread": nanstd(v1),
        "inside_out_dG_diffz_std_objmedian": nanmedian(v2),
        "inside_out_dG_diffz_std_objspread": nanstd(v2),
    }


def extract_boundary_features(tracks, frame_paths, ring_r=3, bg_ring_r=10):
    """combine group 2 features -- inter-object + boundary physics"""
    feats={}
    feats.update(inter_object_consistency_features(tracks, frame_paths, ring_r=ring_r))
    feats.update(boundary_orientation_phase_noise_features(tracks, frame_paths, ring_r=ring_r))
    feats.update(edge_flow_alignment_features(tracks, frame_paths, ring_r=ring_r))
    feats.update(inside_outside_contrast_features(tracks, frame_paths, ring_r=ring_r, bg_ring_r=bg_ring_r))
    return feats



#group3
def extract_noise_bg_features_from_tracks(tracks,frame_paths, inner_r=3, outer_r=12, blur_ksize=7, psd_use_highpass=True, psd_f_low=0.08, psd_f_high=0.45, sample_every=1, min_pixels=200):
    """hf var + psd slope difference between object vs bg ring"""
    per_obj={}

    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])
        ratios, slope_diffs = [], []
        valid = total = 0

        for d in seq:
            t = int(d["t"])
            if t < 0 or t >= len(frame_paths): 
                continue
            if sample_every > 1 and t % sample_every != 0: 
                continue
            frame = cv2.imread(frame_paths[t])
            if frame is None: 
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = resize_mask(d["mask"], gray)
            m = mask.astype(np.uint8)
            ki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner_r + 1, 2 * inner_r + 1))
            ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer_r + 1, 2 * outer_r + 1))
            dil_in = cv2.dilate(m, ki, iterations=1).astype(bool)
            dil_out = cv2.dilate(m, ko, iterations=1).astype(bool)
            bg = dil_out & (~dil_in)

            total += 1
            if mask.sum() < min_pixels or bg.sum() < min_pixels: 
                continue
            valid += 1

            hp = highpass(gray, ksize=blur_ksize)
            ratios.append(float(np.var(hp[mask])) / (float(np.var(hp[bg])) + 1e-9))

            if psd_use_highpass:
                base = hp
            else:
                base = gray
                
            slope_diffs.append(radial_psd_slope(base, mask=mask, f_low=psd_f_low, f_high=psd_f_high) -radial_psd_slope(base, mask=bg, f_low=psd_f_low, f_high=psd_f_high))

        per_obj[obj_id] = {"hf_var_ratio_median": nanmedian(np.array(ratios)), "hf_var_ratio_std": nanstd(np.array(ratios)), "psd_slope_diff_median": nanmedian(np.array(slope_diffs)), "psd_slope_diff_std": nanstd(np.array(slope_diffs)), "valid_frac": float(valid / max(total, 1))}

    keys = ["hf_var_ratio_median", "hf_var_ratio_std", "psd_slope_diff_median", "psd_slope_diff_std", "valid_frac"]
    
    video_feats={}
    for k in keys:
        vals = np.array([per_obj[o][k] for o in per_obj], dtype=float)
        video_feats[f"{k}_objmedian"] = nanmedian(vals)
        video_feats[f"{k}_objspread"] = nanstd(vals)

    return video_feats, per_obj





def extract_all_physics_features(tracks, frame_paths):
    """all together (except first group)"""
    feats={}
    feats.update(extract_boundary_features(tracks, frame_paths))
    noise_feats, _ = extract_noise_bg_features_from_tracks(tracks, frame_paths)
    feats.update(noise_feats)

    return np.array([feats.get(k, float("nan")) for k in PHYSICS_FEATURE_KEYS], dtype=np.float32)
