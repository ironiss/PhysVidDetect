import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation


def camera_positions(extrinsic):
    """camera centers from extrinsics -- world positions"""
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    return np.array([-R[i].T @ t[i] for i in range(len(R))])


def angular_velocities(extrinsic):
    """rotation change between frames -- angle per step"""
    R = extrinsic[:, :3, :3]
    angles = np.empty(len(R)-1)
    for i in range(len(R)-1):
        R_rel = R[i+1] @ R[i].T
        cos_a = np.clip((np.trace(R_rel)-1.0)/2.0, -1.0, 1.0)
        angles[i] = np.arccos(cos_a)
    return angles


def euler_angles(extrinsic):
    """rotation as euler angles -- per frame"""
    R = extrinsic[:, :3, :3]
    return np.array([Rotation.from_matrix(R[i]).as_euler("xyz") for i in range(len(R))])


def reprojection_errors(extrinsic, intrinsic, depth_map, sample_stride=16):
    """reprojection consistency (depth vs projected points)"""
    N, H, W = depth_map.shape
    vs, us = np.mgrid[0:H:sample_stride, 0:W:sample_stride]
    u_flat = us.ravel().astype(np.float64)
    v_flat = vs.ravel().astype(np.float64)

    errors = np.empty(N-1)
    for i in range(N-1):
        d = depth_map[i][vs.ravel(), us.ravel()].astype(np.float64)
        valid = d>1e-6
        if valid.sum()<10:
            errors[i] = 0.0
            continue

        u, v, d = u_flat[valid], v_flat[valid], d[valid]

        K_inv = np.linalg.inv(intrinsic[i].astype(np.float64))
        P_cam = K_inv @ np.stack([u, v, np.ones_like(u)])*d

        Ri, ti = extrinsic[i, :3, :3].astype(np.float64), extrinsic[i, :3, 3].astype(np.float64)
        P_w = Ri.T @ (P_cam - ti[:, None])

        Rn, tn = extrinsic[i+1, :3, :3].astype(np.float64), extrinsic[i+1, :3, 3].astype(np.float64)
        P_next = Rn @ P_w + tn[:, None]

        z_proj = P_next[2]
        in_front = z_proj>1e-6
        if in_front.sum()<10:
            errors[i] = 0.0
            continue

        Kn = intrinsic[i+1].astype(np.float64)
        proj = Kn @ P_next[:, in_front]
        zp = proj[2]
        up = np.clip((proj[0]/zp).astype(int), 0, W-1)
        vp = np.clip((proj[1]/zp).astype(int), 0, H-1)

        d_actual = depth_map[i+1][vp, up].astype(np.float64)
        ok = d_actual>1e-6
        if ok.sum()<10:
            errors[i] = 0.0
            continue

        rel = np.abs(zp[ok] - d_actual[ok]) / np.maximum(zp[ok], d_actual[ok])
        errors[i] = rel.mean()

    return errors


def extract_features(vggt_output):
    """camera + depth features (motion rotation reprojection stats)"""
    ext = vggt_output["extrinsic"]
    intr = vggt_output["intrinsic"]
    dm = vggt_output["depth_map"]
    dc = vggt_output["depth_conf"]
    N = ext.shape[0]
    f = {}

    pos = camera_positions(ext)
    vel = np.diff(pos, axis=0)
    vel_mag = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel, axis=0)
    acc_mag = np.linalg.norm(acc, axis=1)

    f["pos_jitter_mean"] = float(acc_mag.mean()) if len(acc_mag) else 0.0
    f["pos_jitter_max"] = float(acc_mag.max()) if len(acc_mag) else 0.0

    mean_v = vel_mag.mean()
    f["velocity_smoothness"] = float(vel_mag.std()/mean_v) if mean_v>1e-10 else 0.0

    straight = np.linalg.norm(pos[-1] - pos[0])
    path = vel_mag.sum()
    f["trajectory_linearity"] = float(straight/path) if path>1e-10 else 1.0

    ang_v = angular_velocities(ext)
    f["angular_vel_mean"] = float(ang_v.mean())
    f["angular_vel_std"] = float(ang_v.std())

    f["angular_jitter"] = float(np.abs(np.diff(ang_v)).mean()) if len(ang_v)>=2 else 0.0

    eu = euler_angles(ext)
    f["roll_stability"] = float(eu[:, 0].std())

    if len(vel_mag)>=3:
        corr, _ = stats.pearsonr(vel_mag, ang_v)
        f["translation_rotation_corr"] = 0.0 if np.isnan(corr) else float(corr)
    else:
        f["translation_rotation_corr"] = 0.0

    if N>=4:
        ratios = []
        for dim in range(3):
            sig = pos[:, dim] - pos[:, dim].mean()
            pw = np.abs(np.fft.rfft(sig))**2
            total = pw.sum()
            if total>1e-10:
                cutoff = max(1, len(pw)//4)
                ratios.append(pw[cutoff:].sum()/total)
            else:
                ratios.append(0.0)
        f["high_freq_energy_ratio"] = float(np.mean(ratios))
    else:
        f["high_freq_energy_ratio"] = 0.0

    if N>=2:
        diffs = [np.std(np.abs(dm[i] - dm[i+1])) for i in range(N-1)]
        f["depth_temporal_std"] = float(np.mean(diffs))
    else:
        f["depth_temporal_std"] = 0.0

    ranges = []
    for i in range(N):
        v = dm[i][dm[i]>0]
        ranges.append(v.max() - v.min() if len(v)>0 else 0.0)
    f["depth_range_stability"] = float(np.std(ranges))

    per_frame_conf = np.array([dc[i].mean() for i in range(N)])
    f["depth_conf_mean"] = float(per_frame_conf.mean())
    f["depth_conf_std"] = float(per_frame_conf.std())

    fx = np.array([intr[i, 0, 0] for i in range(N)])
    f["focal_length_std"] = float(fx.std())

    cx = np.array([intr[i, 0, 2] for i in range(N)])
    cy = np.array([intr[i, 1, 2] for i in range(N)])
    drift = np.sqrt((cx - cx.mean())**2 + (cy - cy.mean())**2)
    f["principal_point_drift"] = float(drift.max())

    rpe = reprojection_errors(ext, intr, dm)
    f["reprojection_error_mean"] = float(rpe.mean()) if len(rpe) else 0.0
    f["reprojection_error_max"] = float(rpe.max()) if len(rpe) else 0.0

    return f


def compute_timeseries(vggt_output):
    """raw signals over time -- for plotting or analysis"""
    ext = vggt_output["extrinsic"]
    intr = vggt_output["intrinsic"]
    dm = vggt_output["depth_map"]
    N = ext.shape[0]

    pos = camera_positions(ext)
    vel = np.diff(pos, axis=0)
    vel_mag = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel, axis=0)
    acc_mag = np.linalg.norm(acc, axis=1)
    ang_v = angular_velocities(ext)

    if N>=2:
        depth_changes = np.array([np.std(np.abs(dm[i] - dm[i+1])) for i in range(N-1)])
    else:
        depth_changes = np.array([])

    fx = np.array([intr[i, 0, 0] for i in range(N)])
    rpe = reprojection_errors(ext, intr, dm)

    return {
        "positions": pos,
        "velocity_mag": vel_mag,
        "acceleration_mag": acc_mag,
        "angular_velocity": ang_v,
        "depth_change": depth_changes,
        "focal_length": fx,
        "reprojection_error": rpe,
    }
