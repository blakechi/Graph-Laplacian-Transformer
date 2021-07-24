def set_parser(parser):
    # Logging
    parser.add_argument('--log_dir', type=str, required=True,
                        metavar='LOGDIR', help='path for the logging file')
    parser.add_argument('--log_msg', default='', type=str,
                        metavar='LOGMSG', help='message for the logging file')

    # Device
    parser.add_argument('--cuda_device', default=0, type=int,
                        metavar='N', help='The index of cuda device')

    # Dataset
    parser.add_argument('--dataset_dir', type=str, required=True,
                        metavar='DATASETDIR',help='path to dataset')
    parser.add_argument('--dataset_name', type=str, required=True,
                        metavar='DATASETNAME', help='Graph dataset name in OGB')
    parser.add_argument('--pin_memory', default=False, action='store_true',
                        help='Whether to copy batches into CUDA pinned memory')
    parser.add_argument('--num_workers', default=1, type=int,
                        metavar='N', help='Number of workers for the dataset')

    # Model
    parser.add_argument('--num_token_layer', default=4, type=int,
                        metavar='N', help='The number of token layers')
    parser.add_argument('--num_cls_layer', default=2, type=int,
                        metavar='N', help='The number of class layers')
    parser.add_argument('--dim', default=128, type=int,
                        metavar='N', help='Embedding / node dimension')
    parser.add_argument('--edge_dim', default=None, type=int,
                        metavar='N', help='Embedding / edge dimension')
    parser.add_argument('--heads', default=4, type=int,
                        metavar='N', help='Number of heads')
    parser.add_argument('--alpha', default=1e-4, type=float,
                        metavar='N', help='Lambda in layer scale')
    parser.add_argument('--num_classes', default=128, type=int,
                        metavar='N', help='Number of classes')
    parser.add_argument('--use_bias', action='store_true',
                        help='Whether to use bias for Q, K, V projection layers')
    parser.add_argument('--use_edge_bias', action='store_true',
                        help='Whether to use bias for edge_K, edge_V projection layers')
    parser.add_argument('--use_attn_expand_bias', action='store_true',
                        help='Whether to use bias for the attention expansion layers')
    parser.add_argument('--head_expand_scale', default=1., type=float,
                        metavar='N', help='Head expansion scale. expanded_heads = ceil(head_expand_scale*heads)')
    parser.add_argument('--ff_expand_scale', default=4, type=int,
                        metavar='N', help='Expansion scale of feed forward layers')
    parser.add_argument('--ff_dropout', default=0., type=float,
                        metavar='N', help='')
    parser.add_argument('--attention_dropout', default=0., type=float,
                        metavar='N', help='')
    parser.add_argument('--path_dropout', default=0., type=float,
                        metavar='N', help='')
    parser.add_argument('--grad_clip_value', default=None, type=lambda x: None if x == "None" else x,
                        metavar='N', help='')

    # Optimizer - Adam
    parser.add_argument('--betas', type=float, default=[0.9, 0.999], nargs=2,
                        metavar='N N', help='betas')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        metavar='N', help='weight decay (default: 1e-4)')

    # Procedure
    parser.add_argument('--epochs', type=int, default=100,
                        metavar='E', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        metavar='MINLR', help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--batch_size', type=int, default=32,
                        metavar='B', help='input batch size for training (default: 32)')
    parser.add_argument('--logging_interval', type=int, default=32,
                        metavar='N', help='logging per N batches')

    # Checkpoints
    parser.add_argument('--checkpoint_name', default='', type=str,
                        metavar='CHECKPOINTNAME', help='Checkpoint file name')
    parser.add_argument('--checkpoint_folder', default='', type=str,
                        metavar='CHECKPOINTFOLDER', help='The folder contains the checkpoint')

    return parser