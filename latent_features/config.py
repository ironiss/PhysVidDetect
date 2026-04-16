
LATENT_FEATURE_KEYS = [
    # group 1 -- basic latent stats
    "latent_mean", "latent_std", "latent_variance",
    "latent_skewness", "latent_kurtosis", "latent_l2_norm",
    # group 2 -- per-channel stats
    "ch0_mean", "ch1_mean", "ch2_mean", "ch3_mean",
    "ch0_variance", "ch1_variance", "ch2_variance", "ch3_variance",
    "ch0_energy", "ch1_energy", "ch2_energy", "ch3_energy",
    # group 3 -- spatial statistics
    "spatial_entropy", "spatial_variance", "spatial_autocorrelation",
    "gradient_magnitude_mean", "gradient_magnitude_std",
    # group 4 -- frequency features
    "fft_energy_total", "high_freq_energy_ratio", "low_freq_energy_ratio",
    "spectral_entropy", "spectral_slope",
    # group 5 -- latent noise residuals
    "noise_residual_mean", "noise_residual_std",
    "noise_residual_energy", "noise_spatial_correlation",
    # group 6 -- patch consistency
    "patch_variance_mean", "patch_variance_std",
    "patch_similarity", "patch_entropy",
    # group 7 -- temporal latent features
    "latent_frame_distance_mean", "latent_frame_distance_std",
    "latent_velocity_mean", "latent_velocity_std",
    "latent_acceleration_mean", "latent_acceleration_std",
    "latent_temporal_entropy",
    "latent_jerk_mean", "latent_smoothness",
    "normalized_temporal_variance", "trajectory_curvature_consistency",
    # group 8 -- latent correlation features
    "channel_correlation_mean", "channel_correlation_std",
    "channel_mutual_information", "covariance_trace",
    # group 9 -- latent manifold features
    "mahalanobis_distance", "latent_density_score",
    "nearest_neighbor_distance",
    # group 10 -- object-level latent features
    "object_latent_variance", "object_latent_entropy",
    "object_latent_temporal_variance", "object_latent_motion_consistency",
    "object_motion_smoothness",
]



NOISE_FEATURE_KEYS = [
    # group 1 -- temporal noise consistency
    "eps_diff_norm_mean", "eps_diff_norm_std",
    "eps_diff_norm_skew", "eps_diff_norm_kurt",
    "eps_diff_cosine_mean", "eps_diff_cosine_std",
    "eps_diff_energy_mean", "eps_diff_entropy",
    # group 2 -- noise trajectory stability
    "eps_velocity_mean", "eps_velocity_std",
    "eps_acceleration_mean", "eps_acceleration_std",
    "eps_jerk_mean",
    "eps_smoothness", "eps_normalized_temporal_var",
    # group 3 -- noise frequency properties
    "eps_temporal_spectral_slope",
    "eps_temporal_hf_energy_ratio",
    "eps_temporal_spectral_entropy",
    "eps_spatial_hf_ratio_mean",
    "eps_spatial_hf_ratio_std",
    "eps_spatial_hf_ratio_temporal_var",
]



ALL_FEATURE_KEYS = LATENT_FEATURE_KEYS + NOISE_FEATURE_KEYS
N_NOISE_FEATURES = len(NOISE_FEATURE_KEYS)