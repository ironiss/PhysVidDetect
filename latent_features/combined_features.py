import numpy as np
from config import LATENT_FEATURE_KEYS, NOISE_FEATURE_KEYS, N_NOISE_FEATURES
from latent_features import (basic_latent_stats, per_channel_stats, spatial_statistics, frequency_features, noise_residual_features, patch_consistency_features, temporal_features, correlation_features, manifold_features, object_latent_features)
from noise_features import (temporal_noise_consistency, noise_trajectory_stability, noise_frequency_features)


def extract_noise_features(eps_stack):
    """temporal noise features from eps -- consistency trajectory freq"""
    if eps_stack is None or not np.isfinite(eps_stack).any():
        return np.full(N_NOISE_FEATURES, np.nan, dtype=np.float32)

    feats = {}
    feats.update(temporal_noise_consistency(eps_stack))
    feats.update(noise_trajectory_stability(eps_stack))
    feats.update(noise_frequency_features(eps_stack))

    return np.array(
        [feats.get(k, float("nan")) for k in NOISE_FEATURE_KEYS],
        dtype=np.float32,
    )


def extract_all_latent_features(z0_stack, eps_stack, masks_per_frame=None):
    """full feature pipeline -- latent+noise ->single vector"""
    feats = {}
    feats.update(basic_latent_stats(z0_stack))
    feats.update(per_channel_stats(z0_stack))
    feats.update(spatial_statistics(z0_stack))
    feats.update(frequency_features(z0_stack))
    feats.update(noise_residual_features(eps_stack))
    feats.update(patch_consistency_features(z0_stack))
    feats.update(temporal_features(z0_stack))
    feats.update(correlation_features(z0_stack))
    feats.update(manifold_features(z0_stack))
    feats.update(object_latent_features(z0_stack, masks_per_frame))

    vec_original = np.array(
        [feats.get(k, float("nan")) for k in LATENT_FEATURE_KEYS],
        dtype=np.float32,
    )
    vec_noise = extract_noise_features(eps_stack)

    return np.concatenate([vec_original, vec_noise])
