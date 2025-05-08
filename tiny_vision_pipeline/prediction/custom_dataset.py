from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, configs):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((configs.IMAGE_SIZE, configs.IMAGE_SIZE)),
            transforms.ToTensor()
        ])
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, self.image_files[idx]  # returning file name to help with visualization
