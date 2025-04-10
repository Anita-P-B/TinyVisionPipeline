
class Config:
    def __init__(
        self,
        IMAGE_SIZE=32,
        CHANNELS=3,
        BATCH_SIZE=32,
        EPOCHS=30,
        CLASSES=None,
        SAVE_PATH="test_dragon",
        LOAD_MODEL="marvelous_dragon_100_epochs_acc_0.8010_loss_0.5814.pt",
        LOAD_PATH="../experiments/marvelous_dragon_100_epochs_run_20250408_164505",
        RUN_DIR_BASE="../experiments",
        SPLIT_SEED=42,
        VAL_RATIO=0.8,
        LEARNING_RATE=1e-3,
        AUGMENTATION_PROB=0.5,
    ):
        self.IMAGE_SIZE = IMAGE_SIZE
        self.CHANNELS = CHANNELS
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.CLASSES = CLASSES or [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        self.SAVE_PATH = SAVE_PATH
        self.LOAD_MODEL = LOAD_MODEL
        self.LOAD_PATH = LOAD_PATH
        self.RUN_DIR_BASE = RUN_DIR_BASE
        self.SPLIT_SEED = SPLIT_SEED
        self.VAL_RATIO = VAL_RATIO
        self.LEARNING_RATE = LEARNING_RATE
        self.AUGMENTATION_PROB = AUGMENTATION_PROB

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
