
from torchvision import transforms
from tiny_vision_pipeline.CONSTS import CONSTS

# ImageNet normalization values
cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]

train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.RandomApply([
        transforms.RandomHorizontalFlip(p=1.0)
    ], p=CONSTS.AUGMENTATION_PROB),

    transforms.RandomApply([
        transforms.RandomRotation(15)
    ], p=CONSTS.AUGMENTATION_PROB),

    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ], p=CONSTS.AUGMENTATION_PROB),

    transforms.RandomApply([
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ], p=CONSTS.AUGMENTATION_PROB),

    transforms.Resize((CONSTS.IMAGE_SIZE, CONSTS.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
])


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CONSTS.IMAGE_SIZE, CONSTS.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
])