
class Config:
    def __init__(
        self,
        # general constants
        IMAGE_SIZE=64,
        CHANNELS=3,
        BATCH_SIZE=32,
        EPOCHS=50,
        CLASSES=None,
        MODEL = "mobilenet_v3_small",  #| "mobilenet_v3_large" | "mobilenet_v3_small"
        LEARNING_RATE=1e-3,
        SCHEDULER = False, # True, | False
        DROPOUT_RATE = 0.5,
        WEIGHT_DECAY = 1e-4,
        AUGMENTATION_PROB=0.3,
        NORM=None,  # None|"mean"

        # scheduler config
        MODE = "min",
        FACTOR = 0.5,
        PATIENCE = 5,
        MIN_LR = 1e-6,
        VERBOSE = True,

        #debug
        SMALL_DATASET = False,
        # paths
        SAVE_PATH="sweep_test",
        LOAD_MODEL=  r".\experiments\no_aug_no_mean_norm_run_20250423_150921/no_aug_no_mean_norm_run_20250423_150921_acc_0.4339_loss_1.7053.pt",
        RUN_DIR_BASE="./experiments",
        CHECKPOINT_PATH = None , # r".\experiments\no_aug_no_mean_norm_run_20250423_150921/no_aug_no_mean_norm_run_20250423_150921_acc_0.4339_loss_1.7053.pt",

        # data split
        SPLIT_SEED=42,
        VAL_RATIO=0.8, #ratio of validation set out of full test set

        # sweep flag
        SWEEP_MODE = False

    ):
        # general constants
        self.IMAGE_SIZE = IMAGE_SIZE
        self.CHANNELS = CHANNELS
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.CLASSES = CLASSES or [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        self.MODEL = MODEL
        self.LEARNING_RATE = LEARNING_RATE
        self.SCHEDULER = SCHEDULER
        self.DROPOUT_RATE = DROPOUT_RATE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.AUGMENTATION_PROB = AUGMENTATION_PROB
        self.NORM = NORM

        # scheduler config
        self.MODE = MODE,
        self.FACTOR = FACTOR,
        self.PATIENCE = PATIENCE,
        self.MIN_LR = MIN_LR,
        self.VERBOSE = VERBOSE

        # debug
        self.SMALL_DATASET = SMALL_DATASET

        # paths
        self.SAVE_PATH = SAVE_PATH
        self.LOAD_MODEL = LOAD_MODEL
        self.RUN_DIR_BASE = RUN_DIR_BASE
        self.CHECKPOINT_PATH = CHECKPOINT_PATH

        # data split
        self.SPLIT_SEED = SPLIT_SEED
        self.VAL_RATIO = VAL_RATIO

        # sweep mode
        self.SWEEP_MODE = SWEEP_MODE

    def update_from_dict(self, config_dict: dict, verbose=True):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if verbose:
                    print(f"✅ Set {key} = {value}")
            else:
                if verbose:
                    print(f"⚠️ Skipped unknown config key: {key}")

    def to_dict(self):
        return self.__dict__.copy()


# Default config
CONSTS = Config()
