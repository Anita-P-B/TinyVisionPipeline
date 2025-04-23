
from torchvision import transforms


def get_transform(config, is_training = True):
    # ImageNet normalization values
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]

    train_transform_list = [
        # image resize
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),

        # augmentation
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=1.0)
        ], p=config.AUGMENTATION_PROB),

        transforms.RandomApply([
            transforms.RandomRotation(15)
        ], p=config.AUGMENTATION_PROB),

        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ], p=config.AUGMENTATION_PROB),

        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ], p=config.AUGMENTATION_PROB),

        #normalization
        transforms.ToTensor()
    ]
    test_transform_lsit = [
        transforms.ToPILImage(),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor()
    ]


    if is_training:
        transform_list = train_transform_list
    else:
        transform_list = test_transform_lsit

    if config.NORM == "mean":
        transform_list.append(
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        )
    data_pipeline = transforms.Compose(transform_list)
    return data_pipeline