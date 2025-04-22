from torchvision import transforms
from tiny_vision_pipeline.CONSTS import CONSTS

def build_transform(train = True):


    # ImageNet normalization values
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]

    if not train:
        # ✨ Clean, sacred transform for validation/test sets
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CONSTS.IMAGE_SIZE, CONSTS.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])

    aug_type = getattr(CONSTS, "AUGMENTATION_TYPE", None)
    aug_prob = CONSTS.AUGMENTATION_PROB

    augmentations = []

    if aug_type is None:
        # Use all augmentations (default combo)
        augmentations += [
            transforms.RandomHorizontalFlip(p=aug_prob),
            transforms.RandomApply([transforms.RandomRotation(15)], p=aug_prob),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=aug_prob),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=aug_prob),
        ]
    else:
        if aug_type == "flip":
            augmentations.append(transforms.RandomHorizontalFlip(p=aug_prob))
        elif aug_type == "rotate":
            augmentations.append(transforms.RandomApply([transforms.RandomRotation(15)], p=aug_prob))
        elif aug_type == "jitter":
            augmentations.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=aug_prob))
        elif aug_type == "affine":
            augmentations.append(transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=aug_prob))

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONSTS.IMAGE_SIZE, CONSTS.IMAGE_SIZE)),
        *augmentations,
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
