import configargparse


def parses():
    # Argument Parser
    parser = configargparse.ArgumentParser(description='field_reconstruction')
    # training settings
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)

    # logging and models save settings
    parser.add_argument('--plot_freq', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--exp', type=str, default='recon',
                        help='experiment directory name')
    parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                        help='Save Checkpoint Point')
    parser.add_argument('--tb_path', type=str, default='logs/tb',
                        help='Save Tensorboard Path')
    parser.add_argument('--snapshot', type=str, default=None)

    # training environment settings
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    return args
