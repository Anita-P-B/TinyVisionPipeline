


# Constants
IMAGE_SIZE = 32
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 1
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SAVE_PATH = "test_dragon"
LOAD_MODEL = "marvelous_dragon_100_epochs_acc_0.8010_loss_0.5814.pt"
LOAD_PATH = r"../experiments/marvelous_dragon_100_epochs_run_20250408_164505"
RUN_DIR_BASE = "../experiments"

SPLIT_SEED = 42
VAL_RATIO = 0.8 # ratio of val/test

LEARNING_RATE = 1e-3
# Augmentations
AUGMENTATION_PROB = 0.5  # like your AUG_CHANCE


