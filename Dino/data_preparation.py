import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms


class FingerprintDataset(Dataset):
    def __init__(self, image_dir, image_paths, labels, transform=None):
        self.image_dir = os.path.abspath(image_dir)
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Split the image into two fingerprints
        width, height = image.size
        left_fingerprint = image.crop((0, 0, width // 2, height))
        right_fingerprint = image.crop((width // 2, 0, width, height))

        if self.transform:
            left_fingerprint = self.transform(left_fingerprint)
            right_fingerprint = self.transform(right_fingerprint)

        label = self.labels[idx]
        return (left_fingerprint, right_fingerprint), label


def prepare_data(image_dir, test_size=0.2, random_state=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_paths = []
    labels = []
    classes = ['class_0', 'class_1']

    for label, class_dir in enumerate(classes):
        class_path = os.path.join(image_dir, class_dir)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"The directory {class_path} does not exist.")
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))
            labels.append(label)

    train_indices, test_indices = train_test_split(list(range(len(image_paths))), test_size=test_size,
                                                   random_state=random_state)

    train_data_info = (train_indices, image_paths, labels)
    test_data_info = (test_indices, image_paths, labels)

    return train_data_info, test_data_info


if __name__ == "__main__":
    image_dir = "C:/Users/FinalProject/Desktop/LoRA/Dino/OurData"  # Replace with your dataset directory
    print(f"Using dataset directory: {os.path.abspath(image_dir)}")
    train_data_info, test_data_info = prepare_data(image_dir)
    torch.save(train_data_info, 'train_data.pth')
    torch.save(test_data_info, 'test_data.pth')
    print("Data prepared and saved.")
