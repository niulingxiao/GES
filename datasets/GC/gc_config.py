model = dict(
    results_path='./results',
    gan_type='WGAN',
    gpu_ids="0",
    cudnn_benchmark=True,
    # Training parameters
    epoch=40,
    batch_size=1,
    num_workers=8,
    # Network parameters
    in_channels=4,
    out_channels=3,
    latent_channels=48,
    pad_type='zero',
    activation='elu',
    norm='none',
    init_type='xavier',
    init_gain=0.02,
    # Dataset parameters
    baseroot='../../inpainting/dataset/Places/img_set',
    baseroot_mask='../../inpainting/dataset/Places/img_set'
)

