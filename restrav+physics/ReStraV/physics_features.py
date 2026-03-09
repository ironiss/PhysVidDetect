"""
physics_features.py
===================
Three groups of physics-inspired video features for deepfake detection.

All three groups operate on a ``tracks`` dict produced by video_tracker.py:
    tracks[obj_id] = [
        {"t": int, "mask": np.ndarray(H,W,bool),
         "bbox": (x1,y1,x2,y2), "area": float,
         "centroid": (cx,cy), "perimeter": float},
        ...
    ]

Feature groups
--------------
GROUP 1 – extract_features_from_tracks()
    Per-object motion/shape time series aggregated across objects.
    Covers: speed, acceleration, jerk, area, circularity, aspect ratio,
    mask IoU stability, boundary gradient energy, sharpness ratio (obj/bg).
    Why: Real objects move smoothly and have stable, physically plausible
    shape changes.  AI-generated objects often show flickering masks,
    abrupt boundary changes, or unphysical speed jumps.

GROUP 2 – extract_physics_feature_set_123()
    Combines four sub-features:
      1. inter_object_consistency  – do all tracked objects show the same
         edge-energy pattern over time?  In real video they co-vary (same
         camera / lighting).  In AI video each object may be generated
         independently → low cross-correlation.
      2. boundary_orientation_phase_noise  – how stable is the dominant
         gradient orientation around each object boundary?  AI generators
         often produce randomly drifting boundary textures.
      3. edge_flow_alignment  – how well does the optical flow direction
         at the boundary align with the gradient direction?  In real video
         edges move in the direction of their gradient (physics of edges).
      4. inside_outside_contrast  – is the sharpness / brightness contrast
         between the object interior and the nearby background stable over
         time?  AI compositing artifacts produce inconsistent contrast.

GROUP 3 – extract_noise_bg_features_from_tracks()
    High-frequency noise and PSD slope comparison between object region
    and surrounding background.
    Why: AI generators often produce a different noise texture inside the
    generated region compared to the real background they are composited
    onto.  The PSD slope is a fingerprint of the sensor/generator noise.

Public API
----------
    extract_features_from_tracks(tracks, frame_paths, fps)  → dict
    extract_physics_feature_set_123(tracks, frame_paths)    → dict
    extract_noise_bg_features_from_tracks(tracks, frame_paths) → (dict, per_obj)
    extract_all_physics_features(tracks, frame_paths, fps)  → dict (26 fixed keys)

    PHYSICS_FEATURE_KEYS  – ordered list of the 26 fixed keys returned by
                            extract_all_physics_features().
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple


# ═══════════════════════════════════════════════════════════════
# Shared helpers (de-duplicated; used across all three groups)
# ═══════════════════════════════════════════════════════════════

def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return float(a) / float(b + eps)


def _sobel_mag_theta(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gradient magnitude and orientation (radians) via 3×3 Sobel."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy), np.arctan2(gy, gx)


def _make_ring(mask_bool: np.ndarray, r: int = 3) -> np.ndarray:
    """Thin boundary ring: dilate XOR erode."""
    m = mask_bool.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil = cv2.dilate(m, k, iterations=1).astype(bool)
    ero = cv2.erode(m, k, iterations=1).astype(bool)
    return dil & (~ero)


def _resize_mask(mask_bool: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Resize a boolean mask to match the shape of gray (H, W)."""
    if mask_bool.shape == gray.shape[:2]:
        return mask_bool
    return cv2.resize(
        mask_bool.astype(np.uint8),
        (gray.shape[1], gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _lap_var(gray: np.ndarray, mask: np.ndarray) -> float:
    """Laplacian variance inside mask (proxy for sharpness)."""
    vals = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)[mask]
    return float(vals.var()) if vals.size >= 10 else float("nan")


def _iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    uni = np.logical_or(m1, m2).sum()
    return float(inter / uni) if uni else float("nan")


def _circ_mean_R(theta: np.ndarray, weights: np.ndarray | None = None) -> Tuple[float, float]:
    """
    Circular mean angle (mu) and resultant length R ∈ [0, 1].
    R = 1  →  all angles identical (perfectly consistent boundary)
    R ≈ 0  →  angles uniformly distributed (chaotic boundary texture)
    """
    if theta.size < 10:
        return float("nan"), float("nan")
    w = np.ones_like(theta, dtype=np.float64) if weights is None else weights.astype(np.float64)
    C = np.sum(w * np.cos(theta))
    S = np.sum(w * np.sin(theta))
    mu = float(np.arctan2(S, C))
    R = float(np.sqrt(C * C + S * S) / (np.sum(w) + 1e-9))
    return mu, R


def _circ_diff(a: float, b: float) -> float:
    """Angular difference wrapped to [-π, π]."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def _nanstd(x) -> float:
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x)
    return float(s) if np.isfinite(s) else float("nan")


def _nanmedian(x) -> float:
    x = np.asarray(x, dtype=float)
    m = np.nanmedian(x)
    return float(m) if np.isfinite(m) else float("nan")


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu, sig = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(sig) or sig < 1e-8:
        return np.full_like(x, np.nan)
    return (x - mu) / sig


def _basic_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "p95": float("nan")}
    return {
        "mean":   float(np.nanmean(x)),
        "std":    float(np.nanstd(x)),
        "median": float(np.nanmedian(x)),
        "p95":    float(np.nanpercentile(x, 95)),
    }


def _diff_stats(x: np.ndarray) -> Dict[str, float]:
    """Stats of first differences — captures temporal jumpiness."""
    if x.size < 3:
        return {"diff_mean": float("nan"), "diff_std": float("nan"),
                "diff_abs_mean": float("nan")}
    d = np.diff(x)
    return {
        "diff_mean":     float(np.nanmean(d)),
        "diff_std":      float(np.nanstd(d)),
        "diff_abs_mean": float(np.nanmean(np.abs(d))),
    }


# ═══════════════════════════════════════════════════════════════
# GROUP 1 — Motion / Shape / Sharpness
# ═══════════════════════════════════════════════════════════════

def extract_features_from_tracks(
    tracks: Dict[int, List[Dict[str, Any]]],
    frame_paths: List[str],
    fps: float,
    ring_r: int = 3,
    grad_top_pct: float = 80.0,
    bg_ring_r: int = 10,
) -> Dict[str, float]:
    """
    Per-object time-series features aggregated across all tracked objects.

    Covers motion (speed/accel/jerk), shape (area/circularity/aspect),
    mask temporal stability (IoU between consecutive frames),
    boundary gradient energy & orientation consistency, and
    sharpness ratio (object vs background ring).

    Returns a flat dict; keys end in ``__obj_mean``, ``__obj_std``,
    ``__obj_median`` after aggregation across objects.
    """
    per_obj_feats: List[Dict[str, float]] = []
    num_objs = 0
    track_lengths = []

    for obj_id, seq in tracks.items():
        if not seq:
            continue
        seq = sorted(seq, key=lambda d: int(d["t"]))
        num_objs += 1
        track_lengths.append(len(seq))

        cx   = np.array([float(d["centroid"][0]) for d in seq], dtype=np.float32)
        cy   = np.array([float(d["centroid"][1]) for d in seq], dtype=np.float32)
        area = np.array([float(d.get("area", np.nan)) for d in seq], dtype=np.float32)
        perim = np.array([float(d.get("perimeter", np.nan)) for d in seq], dtype=np.float32)
        aspect = np.array([
            max(1., float(d["bbox"][2] - d["bbox"][0] + 1)) /
            max(1., float(d["bbox"][3] - d["bbox"][1] + 1))
            for d in seq
        ], dtype=np.float32)
        circularity = (4.0 * np.pi * area) / (perim * perim + 1e-9)

        dx, dy = np.diff(cx), np.diff(cy)
        speed  = np.sqrt(dx * dx + dy * dy)
        accel  = np.diff(speed)
        jerk   = np.diff(accel)

        # Mask temporal stability
        ious, comp_counts, holes_counts = [], [], []
        for k in range(1, len(seq)):
            ious.append(_iou(seq[k - 1]["mask"], seq[k]["mask"]))
            m_u8 = seq[k]["mask"].astype(np.uint8)
            n_cc, _ = cv2.connectedComponents(m_u8)
            comp_counts.append(float(max(0, n_cc - 1)))
            x1, y1, x2, y2 = [int(v) for v in seq[k]["bbox"]]
            roi = m_u8[max(0, y1): y2 + 1, max(0, x1): x2 + 1]
            if roi.size < 50:
                holes_counts.append(float("nan"))
            else:
                n_bg, _ = cv2.connectedComponents((roi == 0).astype(np.uint8))
                holes_counts.append(float(max(0, n_bg - 1)))

        ious         = np.array(ious,         dtype=np.float32)
        comp_counts  = np.array(comp_counts,  dtype=np.float32)
        holes_counts = np.array(holes_counts, dtype=np.float32)

        # Per-frame gradient & sharpness
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
                nan_entry(); prev_mag, prev_ring = None, None; continue

            frame = cv2.imread(frame_paths[fi], cv2.IMREAD_COLOR)
            if frame is None:
                nan_entry(); prev_mag, prev_ring = None, None; continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = _resize_mask(d["mask"].astype(bool), gray)
            ring    = _make_ring(mask, r=ring_r)
            bg_ring = _make_ring(mask, r=bg_ring_r) & (~mask)
            mag, theta = _sobel_mag_theta(gray)

            # Boundary gradient stats
            vals = mag[ring]
            if vals.size < 20:
                edge_mean.append(float("nan")); edge_std.append(float("nan")); edge_R.append(float("nan"))
            else:
                edge_mean.append(float(np.mean(vals)))
                edge_std.append(float(np.std(vals)))
                thr = np.percentile(vals, grad_top_pct)
                sel = ring & (mag >= thr)
                _, R = _circ_mean_R(theta[sel], mag[sel])
                edge_R.append(R)

            obj_vals = mag[mask]
            grad_obj_mean.append(float(np.mean(obj_vals)) if obj_vals.size >= 20 else float("nan"))

            if prev_mag is not None and prev_ring is not None:
                v = np.abs(mag - prev_mag)[ring]
                grad_res_bnd.append(float(np.mean(v)) if v.size >= 20 else float("nan"))
            else:
                grad_res_bnd.append(float("nan"))

            s_obj = _lap_var(gray, mask)
            s_bg  = _lap_var(gray, bg_ring) if bg_ring.sum() > 50 else float("nan")
            sharp_obj.append(s_obj)
            sharp_bg.append(s_bg)
            sharp_ratio.append(
                _safe_div(s_obj, s_bg)
                if np.isfinite(s_obj) and np.isfinite(s_bg) else float("nan")
            )
            prev_mag, prev_ring = mag, ring

        edge_mean     = np.array(edge_mean,     dtype=np.float32)
        edge_std      = np.array(edge_std,      dtype=np.float32)
        edge_R        = np.array(edge_R,        dtype=np.float32)
        grad_res_bnd  = np.array(grad_res_bnd,  dtype=np.float32)
        grad_obj_mean = np.array(grad_obj_mean, dtype=np.float32)
        sharp_obj     = np.array(sharp_obj,     dtype=np.float32)
        sharp_bg      = np.array(sharp_bg,      dtype=np.float32)
        sharp_ratio   = np.array(sharp_ratio,   dtype=np.float32)

        f: Dict[str, float] = {}
        f["track_len"]     = float(len(seq))
        f["track_seconds"] = float(len(seq) / max(1e-9, fps))
        f.update({f"speed_{k}":    v for k, v in _basic_stats(speed).items()})
        f.update({f"speed_{k}":    v for k, v in _diff_stats(speed).items()})
        f.update({f"accel_{k}":    v for k, v in _basic_stats(accel).items()})
        f.update({f"jerk_{k}":     v for k, v in _basic_stats(jerk).items()})
        f.update({f"area_{k}":     v for k, v in _basic_stats(area).items()})
        f.update({f"area_{k}":     v for k, v in _diff_stats(area).items()})
        f.update({f"perim_{k}":    v for k, v in _basic_stats(perim).items()})
        f.update({f"circ_{k}":     v for k, v in _basic_stats(circularity).items()})
        f.update({f"circ_{k}":     v for k, v in _diff_stats(circularity).items()})
        f.update({f"aspect_{k}":   v for k, v in _basic_stats(aspect).items()})
        f.update({f"iou_{k}":      v for k, v in _basic_stats(ious).items()})
        f["iou_low_frac"] = float(np.nanmean(ious < 0.4)) if ious.size else float("nan")
        f.update({f"cc_{k}":       v for k, v in _basic_stats(comp_counts).items()})
        f.update({f"holes_{k}":    v for k, v in _basic_stats(holes_counts).items()})
        f.update({f"edge_mean_{k}": v for k, v in _basic_stats(edge_mean).items()})
        f.update({f"edge_mean_{k}": v for k, v in _diff_stats(edge_mean).items()})
        f.update({f"edge_std_{k}":  v for k, v in _basic_stats(edge_std).items()})
        f.update({f"edge_R_{k}":    v for k, v in _basic_stats(edge_R).items()})
        f.update({f"gradres_bnd_{k}": v for k, v in _basic_stats(grad_res_bnd).items()})
        f.update({f"grad_obj_{k}": v for k, v in _basic_stats(grad_obj_mean).items()})
        f.update({f"grad_obj_{k}": v for k, v in _diff_stats(grad_obj_mean).items()})
        f.update({f"sharp_obj_{k}":   v for k, v in _basic_stats(sharp_obj).items()})
        f.update({f"sharp_bg_{k}":    v for k, v in _basic_stats(sharp_bg).items()})
        f.update({f"sharp_ratio_{k}": v for k, v in _basic_stats(sharp_ratio).items()})
        f.update({f"sharp_ratio_{k}": v for k, v in _diff_stats(sharp_ratio).items()})
        per_obj_feats.append(f)

    # Aggregate across objects
    out: Dict[str, float] = {
        "num_objects":    float(num_objs),
        "track_len_mean": float(np.mean(track_lengths)) if track_lengths else float("nan"),
        "track_len_std":  float(np.std(track_lengths))  if track_lengths else float("nan"),
    }
    if not per_obj_feats:
        return out

    all_keys = sorted({k for d in per_obj_feats for k in d})
    for k in all_keys:
        vals = np.array([d.get(k, np.nan) for d in per_obj_feats], dtype=np.float32)
        out[f"{k}__obj_mean"]   = float(np.nanmean(vals))
        out[f"{k}__obj_std"]    = float(np.nanstd(vals))
        out[f"{k}__obj_median"] = float(np.nanmedian(vals))

    return out


# ═══════════════════════════════════════════════════════════════
# GROUP 2 — Inter-object & Boundary Physics  (features 1, 2a, 2b, 3)
# ═══════════════════════════════════════════════════════════════

def _boundary_edge_series(tracks, frame_paths, ring_r=3, min_vals=20):
    """Build per-object time series of mean boundary gradient magnitude."""
    out = {}
    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])
        if not seq:
            out[obj_id] = np.array([], dtype=float); continue
        tmax   = max(int(d["t"]) for d in seq)
        series = np.full((tmax + 1,), np.nan, dtype=float)
        for d in seq:
            t = int(d["t"])
            if t < 0 or t >= len(frame_paths): continue
            frame = cv2.imread(frame_paths[t])
            if frame is None: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mag, _ = _sobel_mag_theta(gray)
            mask = _resize_mask(d["mask"], gray)
            ring = _make_ring(mask, r=ring_r)
            vals = mag[ring]
            if vals.size >= min_vals:
                series[t] = float(vals.mean())
        out[obj_id] = series
    return out


def inter_object_consistency_features(
    tracks: Dict, frame_paths: List[str], ring_r: int = 3
) -> Dict[str, float]:
    """
    Feature 1: Inter-object edge-energy consistency.

    In real video, all visible objects share the same camera, lighting and
    compression — their boundary-gradient energies co-vary over time.
    In AI-generated or composited video, each object may be rendered
    independently → low cross-correlation of boundary-energy time series.

    Returns: inter_obj_corr_{median, mean, min}, inter_obj_pairs
    """
    series_dict = _boundary_edge_series(tracks, frame_paths, ring_r=ring_r)
    obj_ids = [o for o, s in series_dict.items() if s.size >= 5]

    if len(obj_ids) < 2:
        return {"inter_obj_corr_median": np.nan, "inter_obj_corr_mean": np.nan,
                "inter_obj_corr_min": np.nan, "inter_obj_pairs": 0}

    maxlen = max(series_dict[o].size for o in obj_ids)
    M = []
    for o in obj_ids:
        s = series_dict[o]
        pad = np.full((maxlen,), np.nan)
        pad[:s.size] = s
        dz = np.diff(_zscore(pad))
        M.append(dz)
    M = np.stack(M, axis=0)

    cors = []
    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            a, b = M[i], M[j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 10: continue
            c = np.corrcoef(a[ok], b[ok])[0, 1]
            if np.isfinite(c): cors.append(float(c))

    if not cors:
        return {"inter_obj_corr_median": np.nan, "inter_obj_corr_mean": np.nan,
                "inter_obj_corr_min": np.nan, "inter_obj_pairs": 0}
    cors = np.array(cors)
    return {
        "inter_obj_corr_median": float(np.nanmedian(cors)),
        "inter_obj_corr_mean":   float(np.nanmean(cors)),
        "inter_obj_corr_min":    float(np.nanmin(cors)),
        "inter_obj_pairs":       int(cors.size),
    }


def boundary_orientation_phase_noise_features(
    tracks: Dict, frame_paths: List[str], ring_r: int = 3, min_vals: int = 50
) -> Dict[str, float]:
    """
    Feature 2a: Boundary gradient orientation phase noise.

    The dominant gradient direction on an object boundary should be stable
    over time (same object edge, same camera).  In AI video, texture
    generation drifts → high std of the circular-mean direction over frames.
    Low R (resultant length) means boundary orientations are incoherent.

    Returns: orient_phase_noise_{objmedian, objspread},
             orient_R_median_{objmedian, objspread}
    """
    per_obj = {}
    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])
        mus, Rs = [], []
        for d in seq:
            t = int(d["t"])
            if t < 0 or t >= len(frame_paths): continue
            frame = cv2.imread(frame_paths[t])
            if frame is None: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mag, theta = _sobel_mag_theta(gray)
            mask = _resize_mask(d["mask"], gray)
            ring = _make_ring(mask, r=ring_r)
            th, w = theta[ring], mag[ring]
            if th.size < min_vals: continue
            mu, R = _circ_mean_R(th, w)
            if np.isfinite(mu): mus.append(mu); Rs.append(R)

        if len(mus) < 2:
            per_obj[obj_id] = {"phase_noise": np.nan, "R_median": np.nan}
            continue
        dmu = np.array([_circ_diff(mus[k], mus[k - 1]) for k in range(1, len(mus))])
        per_obj[obj_id] = {
            "phase_noise": _nanstd(dmu),
            "R_median":    _nanmedian(np.array(Rs)),
        }

    pn = np.array([v["phase_noise"] for v in per_obj.values()], dtype=float)
    Rv = np.array([v["R_median"]    for v in per_obj.values()], dtype=float)
    return {
        "orient_phase_noise_objmedian": _nanmedian(pn),
        "orient_phase_noise_objspread": _nanstd(pn),
        "orient_R_median_objmedian":    _nanmedian(Rv),
        "orient_R_median_objspread":    _nanstd(Rv),
    }


def edge_flow_alignment_features(
    tracks: Dict, frame_paths: List[str], ring_r: int = 3, min_vals: int = 50
) -> Dict[str, float]:
    """
    Feature 2b: Edge–optical-flow alignment on boundary ring.

    On a real moving boundary, the optical flow direction aligns with the
    gradient direction (edges move in the direction they point).  In AI
    video, flow and edges are computed/generated separately → low alignment
    or high temporal variability of the alignment signal.

    alignment(t) = weighted mean of |cos(θ_grad − θ_flow)| on boundary ring
    Returns: edgeflow_align_{median,diff_std} aggregated across objects.
    """
    per_obj = {}
    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])
        if len(seq) < 2:
            per_obj[obj_id] = {"align_median": np.nan, "align_diff_std": np.nan}; continue

        align_series = []
        for k in range(1, len(seq)):
            t0, t1 = int(seq[k - 1]["t"]), int(seq[k]["t"])
            if not (0 <= t0 < len(frame_paths) and 0 <= t1 < len(frame_paths)):
                align_series.append(np.nan); continue
            f0 = cv2.imread(frame_paths[t0]); f1 = cv2.imread(frame_paths[t1])
            if f0 is None or f1 is None:
                align_series.append(np.nan); continue

            g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            phi = np.arctan2(flow[..., 1], flow[..., 0])

            mag, theta = _sobel_mag_theta(g1)
            mask = _resize_mask(seq[k]["mask"], g1)
            ring = _make_ring(mask, r=ring_r)
            th, ph, w = theta[ring], phi[ring], mag[ring]
            if th.size < min_vals:
                align_series.append(np.nan); continue
            a = np.abs(np.cos(th - ph))
            align_series.append(float(np.sum(a * w) / (np.sum(w) + 1e-12)))

        a = np.array(align_series, dtype=float)
        per_obj[obj_id] = {
            "align_median":   _nanmedian(a),
            "align_diff_std": _nanstd(np.diff(a)),
        }

    v1 = np.array([v["align_median"]   for v in per_obj.values()], dtype=float)
    v2 = np.array([v["align_diff_std"] for v in per_obj.values()], dtype=float)
    return {
        "edgeflow_align_median_objmedian":   _nanmedian(v1),
        "edgeflow_align_median_objspread":   _nanstd(v1),
        "edgeflow_align_diff_std_objmedian": _nanmedian(v2),
        "edgeflow_align_diff_std_objspread": _nanstd(v2),
    }


def inside_outside_contrast_features(
    tracks: Dict, frame_paths: List[str],
    ring_r: int = 3, bg_ring_r: int = 10, min_vals: int = 50
) -> Dict[str, float]:
    """
    Feature 3: Object–background contrast stability.

    In real video, the brightness and gradient-energy difference between
    an object and its local background stays consistent (same scene).
    AI compositing often produces an unstable contrast (object brightens
    or sharpens relative to background in a way that defies physics).

    ΔI(t) = mean(pixel) inside object − mean(pixel) in outer ring
    ΔG(t) = mean(|∇I|) inside object  − mean(|∇I|) in outer ring
    Feature: std( diff( zscore(ΔI) ) )  and similarly for ΔG
    Returns: inside_out_{dI,dG}_diffz_std aggregated across objects.
    """
    per_obj = {}
    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])
        deltaI, deltaG = [], []
        for d in seq:
            t = int(d["t"])
            if t < 0 or t >= len(frame_paths): continue
            frame = cv2.imread(frame_paths[t])
            if frame is None: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mag, _ = _sobel_mag_theta(gray.astype(np.uint8))
            mask = _resize_mask(d["mask"], gray)
            m = mask.astype(np.uint8)
            ki  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_r   + 1, 2 * ring_r   + 1))
            kbg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bg_ring_r + 1, 2 * bg_ring_r + 1))
            ero_in  = cv2.erode(m, ki, iterations=1).astype(bool)
            dil_in  = cv2.dilate(m, ki, iterations=1).astype(bool)
            dil_bg  = cv2.dilate(m, kbg, iterations=1).astype(bool)
            inside  = mask & (~ero_in)
            outside = dil_bg & (~dil_in)
            if inside.sum() < min_vals or outside.sum() < min_vals: continue
            deltaI.append(float(gray[inside].mean()  - gray[outside].mean()))
            deltaG.append(float(mag[inside].mean()   - mag[outside].mean()))

        dI = np.array(deltaI, dtype=float)
        dG = np.array(deltaG, dtype=float)
        if dI.size < 5:
            per_obj[obj_id] = {"dI": np.nan, "dG": np.nan}; continue
        per_obj[obj_id] = {
            "dI": _nanstd(np.diff(_zscore(dI))),
            "dG": _nanstd(np.diff(_zscore(dG))),
        }

    v1 = np.array([v["dI"] for v in per_obj.values()], dtype=float)
    v2 = np.array([v["dG"] for v in per_obj.values()], dtype=float)
    return {
        "inside_out_dI_diffz_std_objmedian": _nanmedian(v1),
        "inside_out_dI_diffz_std_objspread": _nanstd(v1),
        "inside_out_dG_diffz_std_objmedian": _nanmedian(v2),
        "inside_out_dG_diffz_std_objspread": _nanstd(v2),
    }


def extract_physics_feature_set_123(
    tracks: Dict, frame_paths: List[str],
    ring_r: int = 3, bg_ring_r: int = 10
) -> Dict[str, float]:
    """
    Combined wrapper for features 1, 2a, 2b, 3 (16 features total).
    """
    feats: Dict[str, float] = {}
    feats.update(inter_object_consistency_features(tracks, frame_paths, ring_r=ring_r))
    feats.update(boundary_orientation_phase_noise_features(tracks, frame_paths, ring_r=ring_r))
    feats.update(edge_flow_alignment_features(tracks, frame_paths, ring_r=ring_r))
    feats.update(inside_outside_contrast_features(tracks, frame_paths, ring_r=ring_r, bg_ring_r=bg_ring_r))
    return feats


# ═══════════════════════════════════════════════════════════════
# GROUP 3 — Noise Texture & PSD Slope  (object vs background)
# ═══════════════════════════════════════════════════════════════

def _highpass(gray: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Remove low-frequency structure; keep sensor/generator noise."""
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return gray - blur


def _crop_bbox(arr: np.ndarray, mask: np.ndarray, pad: int = 2):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None, None
    x1, x2 = max(0, xs.min() - pad), min(arr.shape[1] - 1, xs.max() + pad)
    y1, y2 = max(0, ys.min() - pad), min(arr.shape[0] - 1, ys.max() + pad)
    return arr[y1: y2 + 1, x1: x2 + 1], mask[y1: y2 + 1, x1: x2 + 1]


def _radial_psd_slope(
    img: np.ndarray, mask: np.ndarray | None = None,
    min_size: int = 32, max_size: int = 128,
    f_low: float = 0.08, f_high: float = 0.45,
) -> float:
    """
    Slope of the radially-averaged log-log PSD.

    A steeper (more negative) slope means more energy at low frequencies —
    typical of smooth AI-generated regions.  Real camera regions have a
    less steep slope due to sensor shot noise.
    Fitting is done on the [f_low, f_high] relative-frequency band.
    """
    if img is None:
        return float("nan")
    if mask is not None:
        crop_img, crop_mask = _crop_bbox(img, mask)
        if crop_img is None or crop_mask.sum() < 20:
            return float("nan")
        m = crop_mask.astype(np.float32)
        mean_val = float(np.sum(crop_img * m) / (m.sum() + 1e-9))
        crop_img = crop_img.copy()
        crop_img[m < 0.5] = mean_val
        img = crop_img

    h, w = img.shape
    if h < min_size or w < min_size:
        return float("nan")
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) - float(np.mean(img))
    F   = np.fft.fftshift(np.fft.fft2(img))
    psd = (np.abs(F) ** 2).astype(np.float32)

    H, W = psd.shape
    cy, cx = H // 2, W // 2
    rr = np.sqrt((np.arange(H)[:, None] - cy) ** 2 + (np.arange(W)[None, :] - cx) ** 2)
    rmax = rr.max()
    if rmax <= 0:
        return float("nan")
    fr = rr / (rmax + 1e-9)

    nbins = 60
    bins = np.linspace(0, 1.0, nbins + 1)
    rad  = 0.5 * (bins[:-1] + bins[1:])
    p    = np.zeros(nbins, dtype=np.float64)
    cnt  = np.zeros(nbins, dtype=np.float64)
    idx  = np.clip(np.digitize(fr.ravel(), bins) - 1, 0, nbins - 1)
    np.add.at(p,   idx, psd.ravel())
    np.add.at(cnt, idx, 1.0)
    p /= (cnt + 1e-9)

    sel = (rad >= f_low) & (rad <= f_high) & (p > 0)
    if sel.sum() < 8:
        return float("nan")
    x  = np.log(rad[sel] + 1e-12)
    y  = np.log(p[sel]   + 1e-12)
    A  = np.vstack([x, np.ones_like(x)]).T
    slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return float(slope)


def extract_noise_bg_features_from_tracks(
    tracks: Dict,
    frame_paths: List[str],
    inner_r: int = 3,
    outer_r: int = 12,
    blur_ksize: int = 7,
    psd_use_highpass: bool = True,
    psd_f_low: float = 0.08,
    psd_f_high: float = 0.45,
    sample_every: int = 1,
    min_pixels: int = 200,
) -> Tuple[Dict[str, float], Dict]:
    """
    Feature group 3: High-frequency noise texture comparison.

    For each object, at each sampled frame:
      hf_var_ratio = Var(highpass, obj) / Var(highpass, bg_ring)
        Detects mismatched noise levels between generated object and real bg.
      psd_slope_diff = PSD_slope(obj) − PSD_slope(bg_ring)
        Detects different frequency roll-off (steeper slope inside AI region).

    Returns (video_features dict, per_object dict).
    """
    per_obj: Dict[int, Dict] = {}

    for obj_id in sorted(tracks):
        seq = sorted(tracks[obj_id], key=lambda d: d["t"])
        ratios, slope_diffs = [], []
        valid = total = 0

        for d in seq:
            t = int(d["t"])
            if t < 0 or t >= len(frame_paths): continue
            if sample_every > 1 and t % sample_every != 0: continue
            frame = cv2.imread(frame_paths[t])
            if frame is None: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = _resize_mask(d["mask"], gray)
            m = mask.astype(np.uint8)
            ki  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner_r + 1, 2 * inner_r + 1))
            ko  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer_r + 1, 2 * outer_r + 1))
            dil_in  = cv2.dilate(m, ki, iterations=1).astype(bool)
            dil_out = cv2.dilate(m, ko, iterations=1).astype(bool)
            bg = dil_out & (~dil_in)

            total += 1
            if mask.sum() < min_pixels or bg.sum() < min_pixels: continue
            valid += 1

            hp = _highpass(gray, ksize=blur_ksize)
            ratios.append(float(np.var(hp[mask])) / (float(np.var(hp[bg])) + 1e-9))

            base = hp if psd_use_highpass else gray
            slope_diffs.append(
                _radial_psd_slope(base, mask=mask, f_low=psd_f_low, f_high=psd_f_high) -
                _radial_psd_slope(base, mask=bg,   f_low=psd_f_low, f_high=psd_f_high)
            )

        per_obj[obj_id] = {
            "hf_var_ratio_median": _nanmedian(np.array(ratios)),
            "hf_var_ratio_std":    _nanstd(np.array(ratios)),
            "psd_slope_diff_median": _nanmedian(np.array(slope_diffs)),
            "psd_slope_diff_std":    _nanstd(np.array(slope_diffs)),
            "valid_frac": float(valid / max(total, 1)),
        }

    keys = ["hf_var_ratio_median", "hf_var_ratio_std",
            "psd_slope_diff_median", "psd_slope_diff_std", "valid_frac"]
    video_feats: Dict[str, float] = {}
    for k in keys:
        vals = np.array([per_obj[o][k] for o in per_obj], dtype=float)
        video_feats[f"{k}_objmedian"] = _nanmedian(vals)
        video_feats[f"{k}_objspread"] = _nanstd(vals)

    return video_feats, per_obj


# ═══════════════════════════════════════════════════════════════
# Combined extractor — 26 fixed-dimension feature vector
# ═══════════════════════════════════════════════════════════════

#: Canonical ordered list of the 26 physics feature keys.
#: Groups 2 (16) + Group 3 (10).  Used to build consistent numpy arrays.
PHYSICS_FEATURE_KEYS: List[str] = [
    # Group 2 — inter-object consistency (4)
    "inter_obj_corr_median", "inter_obj_corr_mean",
    "inter_obj_corr_min", "inter_obj_pairs",
    # Group 2 — boundary orientation phase noise (4)
    "orient_phase_noise_objmedian", "orient_phase_noise_objspread",
    "orient_R_median_objmedian", "orient_R_median_objspread",
    # Group 2 — edge-flow alignment (4)
    "edgeflow_align_median_objmedian", "edgeflow_align_median_objspread",
    "edgeflow_align_diff_std_objmedian", "edgeflow_align_diff_std_objspread",
    # Group 2 — inside-outside contrast (4)
    "inside_out_dI_diffz_std_objmedian", "inside_out_dI_diffz_std_objspread",
    "inside_out_dG_diffz_std_objmedian", "inside_out_dG_diffz_std_objspread",
    # Group 3 — HF noise variance ratio (4)
    "hf_var_ratio_median_objmedian", "hf_var_ratio_median_objspread",
    "hf_var_ratio_std_objmedian", "hf_var_ratio_std_objspread",
    # Group 3 — PSD slope difference (4)
    "psd_slope_diff_median_objmedian", "psd_slope_diff_median_objspread",
    "psd_slope_diff_std_objmedian", "psd_slope_diff_std_objspread",
    # Group 3 — valid fraction (2)
    "valid_frac_objmedian", "valid_frac_objspread",
]
assert len(PHYSICS_FEATURE_KEYS) == 26


def extract_all_physics_features(
    tracks: Dict,
    frame_paths: List[str],
    fps: float = 25.0,
) -> np.ndarray:
    """
    Run Groups 2 + 3 and return a fixed-length float32 array of shape (26,).

    NaN is used for any feature that cannot be computed (e.g., too few frames,
    no objects tracked).  The caller should NaN-fill before passing to the MLP.

    Parameters
    ----------
    tracks :
        Output of video_tracker.get_video_tracks().
    frame_paths :
        Ordered list of frame image paths on disk.
    fps :
        Video frame rate (used only if Group 1 is added later).

    Returns
    -------
    np.ndarray, shape (26,), dtype float32
    """
    feats: Dict[str, float] = {}
    feats.update(extract_physics_feature_set_123(tracks, frame_paths))
    noise_feats, _ = extract_noise_bg_features_from_tracks(tracks, frame_paths)
    feats.update(noise_feats)

    return np.array(
        [feats.get(k, float("nan")) for k in PHYSICS_FEATURE_KEYS],
        dtype=np.float32,
    )
