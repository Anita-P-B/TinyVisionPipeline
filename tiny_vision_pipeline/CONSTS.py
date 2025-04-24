
class Config:
    def __init__(
        self,
        # general constants
        IMAGE_SIZE=32,
        CHANNELS=3,
        BATCH_SIZE=32,
        EPOCHS=50,
        CLASSES=None,
        LEARNING_RATE=1e-3,
        AUGMENTATION_PROB=0,
        NORM=None,  # None|"mean"

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
        self.LEARNING_RATE = LEARNING_RATE
        self.AUGMENTATION_PROB = AUGMENTATION_PROB
        self.NORM = NORM

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
