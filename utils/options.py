def add_opts(parser):
    # options for wandb
    parser.add_argument("--run_name", dest="run_name", type=str, help="wandb run name")

    # options for dataset
    parser.add_argument("--data_root", type=str, help="dataset root", required=True)
    parser.add_argument(
        "--mano_root", default="assets/mano_models", type=str, help="mano root"
    )
    parser.add_argument(
        "--obj_model_root",
        default="assets/object_models",
        type=str,
        help="object model root",
    )
    parser.add_argument("--inp_res", default=512, type=int, help="input image size")
    ## additional options for the temporal dataset
    parser.add_argument(
        "--T", default=7, type=int, help="the number of frames in a sequence"
    )
    parser.add_argument(
        "--gap", default=10, type=int, help="the gap between two consecutive frames"
    )

    # options for model
    parser.add_argument(
        "--model",
        default="deformer",
        choices=["deformer"],
        help="network core architecture",
    )
    parser.add_argument(
        "--channels",
        default=256,
        type=int,
        help="Number of channels in the hourglass (default: 256)",
    )
    ## additional options for deformer
    parser.add_argument(
        "--transformer_layer",
        default=6,
        type=int,
        help="Number of  transformer encoder/decoder(default:6)",
    )
    parser.add_argument(
        "--transformer_head",
        default=8,
        type=int,
        help="Number of transformer attention head",
    )
    parser.add_argument(
        "--transformer_dim_feedforward",
        default=256,
        type=int,
        help="Transformer feedforward dimension",
    )
    parser.add_argument(
        "--temporal_transformer_layer",
        default=6,
        type=int,
        help="Number of temporal transformer encoder/decoder(default:6)",
    )
    parser.add_argument(
        "--temporal_transformer_head",
        default=8,
        type=int,
        help="Number of temporal transformer attention head",
    )
    parser.add_argument(
        "--temporal_constrained",
        dest="temporal_constrained",
        action="store_true",
        help="Add temporal loss",
    )
    #### additional options for discriminator
    parser.add_argument(
        "--motion_discrimination",
        dest="motion_discrimination",
        action="store_true",
        help="Add motion discriminator",
    )

    # options for loss
    parser.add_argument(
        "--mano_neurons",
        nargs="+",
        default=[1024, 512],
        type=int,
        help="Number of neurons in hidden layer for mano decoder",
    )
    parser.add_argument(
        "--mano_lambda_joints3d",
        default=1e4,
        type=float,
        help="Weight to supervise joints in 3d",
    )
    parser.add_argument(
        "--mano_lambda_verts3d",
        default=1e4,
        type=float,
        help="Weight to supervise vertices in 3d",
    )
    parser.add_argument(
        "--mano_lambda_manopose",
        default=10,
        type=float,
        help="Weight to supervise mano pose parameters",
    )
    parser.add_argument(
        "--mano_lambda_manoshape",
        default=0.1,
        type=float,
        help="Weight to supervise mano shape parameters",
    )
    parser.add_argument(
        "--mano_lambda_regulshape",
        default=1e2,
        type=float,
        help="Weight to regularize hand shapes",
    )
    parser.add_argument(
        "--mano_lambda_regulpose",
        default=1,
        type=float,
        help="Weight to regularize hand pose in axis-angle space",
    )
    parser.add_argument(
        "--lambda_joints2d",
        default=1e2,
        type=float,
        help="Weight to supervise joints in 2d",
    )
    ## temporal constrains
    parser.add_argument(
        "--temporal_mano_lambda_joints3d",
        default=1e4,
        type=float,
        help="Weight to supervise joints in 3d",
    )
    parser.add_argument(
        "--temporal_mano_lambda_verts3d",
        default=1e4,
        type=float,
        help="Weight to supervise vertices in 3d",
    )
    parser.add_argument(
        "--temporal_mano_lambda_manopose",
        default=10,
        type=float,
        help="Weight to supervise mano pose parameters",
    )
    parser.add_argument(
        "--temporal_mano_lambda_manoshape",
        default=0.1,
        type=float,
        help="Weight to supervise mano shape parameters",
    )
    ## dynamic constrains
    parser.add_argument(
        "--dynamic_mano_lambda_joints3d",
        default=1e4,
        type=float,
        help="Weight to supervise joints in 3d",
    )
    parser.add_argument(
        "--dynamic_mano_lambda_verts3d",
        default=1e4,
        type=float,
        help="Weight to supervise vertices in 3d",
    )
    parser.add_argument(
        "--dynamic_mano_lambda_manopose",
        default=10,
        type=float,
        help="Weight to supervise mano pose parameters",
    )
    parser.add_argument(
        "--dynamic_mano_lambda_manoshape",
        default=0.1,
        type=float,
        help="Weight to supervise mano shape parameters",
    )
    parser.add_argument(
        "--end2end_mano_lambda_joints3d",
        default=1e4,
        type=float,
        help="Weight to supervise joints in 3d",
    )
    parser.add_argument(
        "--end2end_mano_lambda_verts3d",
        default=1e4,
        type=float,
        help="Weight to supervise vertices in 3d",
    )
    parser.add_argument(
        "--end2end_mano_lambda_manopose",
        default=10,
        type=float,
        help="Weight to supervise mano pose parameters",
    )
    ## discrimination loss
    parser.add_argument(
        "--motion_dis_loss_weight",
        default=10,
        type=float,
        help="Weight to supervise mano shape parameters",
    )
    ## adaptive loss
    parser.add_argument("--loss_base", default="maxmse", type=str)

    # options for training
    parser.add_argument("--manual_seed", default=0, type=int, help="manual seed")
    parser.add_argument(
        "-j",
        "--workers",
        default=16,
        type=int,
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--train_batch", default=24, type=int, help="Train batch size")
    parser.add_argument(
        "--test_batch", default=16, type=int, metavar="N", help="Test batch size"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr_backbone",
        "--learning-rate-backbone",
        default=1e-4,
        type=float,
        help="initial learning rate for backbone",
    )
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument(
        "--lr_decay_step",
        default=10,
        type=int,
        help="epochs after which to decay learning rate",
    )
    parser.add_argument(
        "--lr_decay_gamma",
        default=0.7,
        type=float,
        help="factor by which to decay the learning rate",
    )
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    ## additional options for discriminator
    parser.add_argument(
        "--motion_dis_lr", default=1e-3, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--motion_dis_lr_decay_step", default=10, type=int, help="initial learning rate"
    )
    parser.add_argument(
        "--motion_dis_weight_decay",
        default=0.0005,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--motion_dis_lr_decay_gamma",
        default=0.7,
        type=float,
        help="initial learning rate",
    )

    # options for exp
    parser.add_argument(
        "--host_folder",
        default="./exp-results",
        type=str,
        help="path to save experiment results",
    )
    parser.add_argument("--resume", type=str, help="path to latest checkpoint")
    parser.add_argument(
        "--evaluate", dest="evaluate", action="store_true", help="evaluate model"
    )
    parser.add_argument(
        "--save_results", default=False, help="save output results of the network"
    )
    parser.add_argument(
        "--test_freq",
        default=10,
        type=int,
        metavar="N",
        help="testing frequency on evaluation dataset",
    )
    parser.add_argument(
        "--snapshot",
        default=10,
        type=int,
        metavar="N",
        help="How often to take a snapshot of the model (0 = never)",
    )
    parser.add_argument(
        "--use_cuda", default=1, type=int, help="use GPU (default: True)"
    )

    # options for DDP
    parser.add_argument("--local_rank", default=-1, type=int)
