
class DataConfig:
    def __init__(self, args, time_series_size=0, node_size=0, node_feature_size=0, num_classes=2):
        self.dataset = args.dataset
        self.mode = args.mode
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.drop_last = args.drop_last
        self.alpha = args.alpha
        self.beta = args.beta
        self.n_splits = args.num_repeat
        self.train_percent = args.train_percent
        self.time_series_size = time_series_size
        self.node_size = node_size
        self.node_feature_size = node_feature_size
        self.num_classes = num_classes
        self.dynamic = args.dynamic
        self.subject_id = args.subject_id
        self.class_weight = [1]*num_classes
        self.label_names = []
        self.augmentation = args.framework == "BCD"
        self.padding_brain = not args.no_padding_brain

