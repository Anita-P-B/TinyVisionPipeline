from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np
# Custom dataset wrapper to apply transform to numpy dataset
class CIFAR10Wrapped(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None, debug = False):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.debug = debug

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]

        if self.transform:
            #img = Image.fromarray(img)  # Needed because your data is NumPy, HWC
            orig_img = img.copy()
            img = self.transform(img)  # Now becomes Tensor in CHW format
            if self.debug:
                plt.figure(figsize=(6, 3))
                plt.subplot(1, 2, 1)
                plt.imshow(orig_img)
                plt.title(f"Original_index_{index}_label_{label}", fontsize=8)
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
                plt.title(f"Transformed_index_{index}_label_{label}", fontsize=8)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(f"./samples/debug_image_{index}.png")  # Saves instead of showing
                plt.close()
        return img, label

    def __len__(self):
        return len(self.data)
