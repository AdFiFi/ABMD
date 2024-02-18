from models import *
from data import DataConfig
import argparse


def init_brain_encoder_config(args, data_config: DataConfig):
    if args.brain_encoder == "EEGNet":
        model_config = EEGNetConfig(node_size=data_config.node_size,
                                    time_series_size=data_config.time_series_size,
                                    node_feature_size=data_config.node_feature_size,
                                    num_classes=data_config.num_classes,
                                    frequency=args.frequency,
                                    D=args.D,
                                    num_kernels=args.num_kernels,
                                    p1=args.p1,
                                    p2=args.p2,
                                    dropout=args.dropout,
                                    fusion=args.brain_fusion,
                                    use_temporal=args.use_temporal)
        model = EEGNet(model_config)
    elif args.brain_encoder == "DFaST":
        model_config = DFaSTConfig(node_size=data_config.node_size,
                                   time_series_size=data_config.time_series_size,
                                   node_feature_size=data_config.node_feature_size,
                                   num_classes=data_config.num_classes,
                                   sparsity=args.sparsity,
                                   frequency=args.frequency,
                                   D=args.D,
                                   p1=args.p1,
                                   p2=args.p2,
                                   k=args.k,
                                   num_kernels=args.num_kernels,
                                   window_size=args.window_size,
                                   num_heads=args.num_heads,
                                   activation=args.activation,
                                   dropout=args.dropout,
                                   fusion=args.brain_fusion,
                                   use_temporal=args.use_temporal)
        model = DFaSTForClassification(model_config)
    elif args.brain_encoder == "LMDA":
        model_config = LMDAConfig(node_size=data_config.node_size,
                                  time_series_size=data_config.time_series_size,
                                  node_feature_size=data_config.node_feature_size,
                                  num_classes=data_config.num_classes,
                                  depth=9,
                                  channel_depth1=args.num_kernels,
                                  channel_depth2=9,
                                  ave_depth=1,
                                  avepool=5,
                                  fusion=args.brain_fusion,
                                  use_temporal=args.use_temporal)
        model = LMDA(model_config)
    elif args.brain_encoder == "ShallowConvNet":
        model_config = ShallowConvNetConfig(node_size=data_config.node_size,
                                            time_series_size=data_config.time_series_size,
                                            node_feature_size=data_config.node_feature_size,
                                            num_classes=data_config.num_classes,
                                            num_kernels=args.num_kernels,
                                            fusion=args.brain_fusion,
                                            use_temporal=args.use_temporal)
        model = ShallowConvNet(model_config)
    elif args.brain_encoder == "DeepConvNet":
        model_config = DeepConvNetConfig(node_size=data_config.node_size,
                                         time_series_size=data_config.time_series_size,
                                         node_feature_size=data_config.node_feature_size,
                                         num_classes=data_config.num_classes,
                                         num_kernels=25,
                                         fusion=args.brain_fusion,
                                         use_temporal=args.use_temporal)
        model = DeepConvNet(model_config)
    elif args.brain_encoder == "MTAMEEGEncoder":
        model_config = MTAMEEGEncoderConfig(num_classes=data_config.num_classes)
        model = MTAMEEGEncoder(model_config)
    elif args.brain_encoder == "EEGChannelNet":
        model_config = EEGChannelNetConfig(node_size=data_config.node_size,
                                           time_series_size=data_config.time_series_size,
                                           node_feature_size=data_config.node_feature_size,
                                           num_classes=data_config.num_classes,
                                           use_temporal=args.use_temporal)
        model_config.class_weight = data_config.class_weight
        model = EEGChannelNet(model_config)
    elif args.brain_encoder == "BNT":
        model_config = BNTConfig(node_size=data_config.node_size,
                                 sizes=(data_config.node_size, data_config.node_size // 2),
                                 num_classes=data_config.num_classes,
                                 pooling=(False, True),
                                 pos_encoding=None,  # identity, none
                                 orthogonal=True,
                                 freeze_center=True,
                                 project_assignment=True,
                                 num_heads=args.num_heads,
                                 pos_embed_dim=data_config.node_size,
                                 )
        model = BNT(model_config)
    elif args.brain_encoder == 'BrainNetCNN':
        model_config = BrainNetCNNConfig(node_size=data_config.node_size,
                                         num_classes=data_config.num_classes)
        model = BrainNetCNN(model_config)
    elif args.brain_encoder == "Graphormer":
        model_config = GraphormerConfig(node_size=data_config.node_size,
                                        num_classes=data_config.num_classes,
                                        node_feature_size=data_config.node_feature_size,
                                        readout='concat',
                                        num_layers=args.num_layers)
        model = Graphormer(model_config)
    else:
        model = None
        model_config = None
    return model, model_config


def init_vision_encoder_config(args, data_config: DataConfig):
    if args.vision_encoder == "ResNet":
        model_config = ResNetConfig(num_classes=data_config.num_classes,
                                    num_channels=1 if args.dataset == "Minist" else 3)
        model = ResNet(model_config)
    else:
        model = None
        model_config = None
    return model, model_config


def init_text_encoder_config(args, data_config: DataConfig):
    if args.text_encoder == "MTAMTextEncoder":
        model_config = MTAMTextEncoderConfig(num_classes=data_config.num_classes,
                                             fusion="none")
        model = MTAMTextEncoder(model_config)
    elif args.text_encoder == "Bert":
        model_config = BertConfig(num_classes=data_config.num_classes,
                                  fusion=args.text_fusion,
                                  extra_layer_num=args.num_layers)
        model = Bert(model_config)
    elif args.text_encoder == "Bart":
        model_config = BartConfig(num_classes=data_config.num_classes,
                                  fusion=args.text_fusion,
                                  extra_layer_num=args.num_layers)
        model = Bart(model_config)
    elif args.text_encoder == "RoBerta":
        model_config = RoBertaConfig(num_classes=data_config.num_classes,
                                     fusion=args.text_fusion,
                                     extra_layer_num=args.num_layers)
        model = RoBerta(model_config)
    else:
        model = None
        model_config = None
    return model, model_config


def init_audio_encoder_config(args, data_config: DataConfig):
    return None, None


def init_config():
    parser = argparse.ArgumentParser()

    global_group = parser.add_argument_group(title="global", description="")
    global_group.add_argument("--project", default="FaST-P-SMR", type=str, help="")
    # global_group.add_argument("--project", default="FaST-P1", type=str, help="")
    # global_group.add_argument("--wandb_entity", default='adfifi', type=str, help="")
    # global_group.add_argument("--wandb_entity", default='13987790547', type=str, help="")
    global_group.add_argument("--wandb_entity", default='cwg', type=str, help="")
    global_group.add_argument("--extra_info", default='', type=str, help="")
    global_group.add_argument("--two_step", action="store_true", help="")
    global_group.add_argument("--log_dir", default="./log", type=str, help="")
    global_group.add_argument("--framework", default="", type=str, help="")
    global_group.add_argument("--use_temporal", action="store_true", help="")
    global_group.add_argument("--use_sequence", action="store_true", help="")
    global_group.add_argument("--no_padding_brain", action="store_true", help="")
    global_group.add_argument("--mode", default="distill", type=str, help="")
    global_group.add_argument("--modality", default="", type=str, help="")
    global_group.add_argument("--brain_encoder", default="", type=str, help="")
    global_group.add_argument("--text_encoder", default="", type=str, help="")
    global_group.add_argument("--vision_encoder", default="", type=str, help="")
    global_group.add_argument("--audio_encoder", default="", type=str, help="")
    global_group.add_argument("--num_repeat", default=10, type=int, help="")
    global_group.add_argument("--within_subject", action="store_true", help="")
    global_group.add_argument("--subject_num", default=10, type=int, help="")
    global_group.add_argument("--visualize", action="store_true", help="")

    data_group = parser.add_argument_group(title="data", description="")
    data_group.add_argument("--dataset", default='Face', type=str, help="")
    data_group.add_argument("--data_dir", default="../../Datasets/Face/Face275.npy", type=str, help="")
    data_group.add_argument("--data_processors", default=0, type=int, help="")
    data_group.add_argument("--batch_size", default=64, type=int, help="")
    data_group.add_argument("--num_pretrain_epochs", default=200, type=int, help="")
    data_group.add_argument("--num_epochs", default=200, type=int, help="")
    data_group.add_argument("--num_first_step_epochs", default=100, type=int, help="")
    # data_group.add_argument("--train_percent", default=1.0, choices=[i / 10 for i in range(1, 11)], type=float, help="")
    data_group.add_argument("--train_percent", default=1.0,
                            choices=[i / 100 for i in range(1, 11)] + [i / 10 for i in range(1, 11)],
                            type=float, help="")
    data_group.add_argument("--drop_last", default=True, type=bool, help="")
    data_group.add_argument("--dynamic", action="store_true", help="")
    data_group.add_argument("--subject_id", default=0, type=int, help="")
    data_group.add_argument("--frequency", default=500, type=int, help="")
    data_group.add_argument("--D", default=2, type=int, help="")
    data_group.add_argument("--F1", default=8, type=int, help="")
    data_group.add_argument("--p1", default=4, type=int, help="")
    data_group.add_argument("--p2", default=8, type=int, help="")

    preprocess_group = parser.add_argument_group(title="preprocess", description="")
    preprocess_group.add_argument("--mix_up", action="store_true", help="")

    framework_group = parser.add_argument_group(title="brain_encoder", description="")
    framework_group.add_argument("--d_model", default=1024, type=int, help="")
    framework_group.add_argument("--d_hidden", default=2048, type=int, help="")

    brain_group = parser.add_argument_group(title="brain_encoder", description="")
    brain_group.add_argument("--k", default=5, type=int, help="")
    brain_group.add_argument("--num_kernels", default=5, type=int, help="")
    brain_group.add_argument("--sparsity", default=0.6, type=float, help="")
    brain_group.add_argument("--window_size", default=50, type=int, help="")
    brain_group.add_argument("--dynamic_length", default=600, type=int, help="")
    brain_group.add_argument("--dynamic_stride", default=1, type=int, help="")
    brain_group.add_argument("--sampling_init", default=None, type=int, help="")
    brain_group.add_argument("--num_heads", default=1, type=int, help="")
    brain_group.add_argument("--num_layers", default=0, type=int, help="")
    brain_group.add_argument("--activation", default="gelu", type=str, help="")
    brain_group.add_argument("--model_dir", default="output", type=str, help="")
    brain_group.add_argument("--dropout", default=0.5, type=float, help="")
    brain_group.add_argument("--initializer", default=None, type=str, help="")
    brain_group.add_argument("--brain_fusion", default="flatten", type=str, help="")

    text_group = parser.add_argument_group(title="text_encoder", description="")
    text_group.add_argument("--text_fusion", default="flatten", type=str, help="")

    train_group = parser.add_argument_group(title="train", description="")
    train_group.add_argument("--do_train", action="store_true", help="")
    train_group.add_argument("--do_parallel", action="store_true", help="")
    train_group.add_argument("--device", default="cuda", type=str, help="")
    train_group.add_argument("--save_steps", default=200, type=int, help="")
    train_group.add_argument("--epsilon_ls", default=0, type=float, help=" label_smoothing")
    train_group.add_argument("--lam", default=0.5, type=float, help="lambda")
    train_group.add_argument("--t", default=3, type=float, help="lambda")
    train_group.add_argument("--alpha", default=1., type=float, help="")
    train_group.add_argument("--beta", default=1., type=float, help="")

    optimizer_group = parser.add_argument_group(title="optimizer", description="")
    optimizer_group.add_argument("--optimizer", default='Adam', type=str, help="")
    optimizer_group.add_argument("--learning_rate", default=1e-4, type=float, help="")
    optimizer_group.add_argument("--target_learning_rate", default=1e-5, type=float, help="")
    optimizer_group.add_argument("--max_learning_rate", default=0.001, type=float, help="")
    optimizer_group.add_argument("--beta1", default=0.9, type=float, help="")
    optimizer_group.add_argument("--beta2", default=0.98, type=float, help="")
    optimizer_group.add_argument("--epsilon", default=1e-9, type=float, help="")
    optimizer_group.add_argument("--schedule", default='cos', type=str, help="")
    optimizer_group.add_argument("--weight_decay", default=1e-4, type=float, help="")
    optimizer_group.add_argument("--no_weight_decay", action="store_true", help="")
    optimizer_group.add_argument("--label_smoothing", default=0, type=float, help="")

    evaluate_group = parser.add_argument_group(title="evaluate", description="")
    evaluate_group.add_argument("--do_evaluate", action="store_true", help="")
    evaluate_group.add_argument("--do_test", action="store_true", help="")

    return parser.parse_args()
