import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch


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
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def prepare_data(image_dir, test_size=0.2, random_state=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_paths = []
    labels = []

    # Assuming 'matching' and 'non_matching' subdirectories exist in the image_dir
    matching_dir = os.path.join(image_dir, 'class_0')
    non_matching_dir = os.path.join(image_dir, 'class_1')
    
    if not os.path.exists(matching_dir) or not os.path.exists(non_matching_dir):
        raise FileNotFoundError(f"The directories for matching or non_matching images do not exist.\nChecked paths:\n{matching_dir}\n{non_matching_dir}")

    # Process matching fingerprints
    for img_name in os.listdir(matching_dir):
        image_paths.append(os.path.join(matching_dir, img_name))
        labels.append(1)  # 1 for matching fingerprints

    # Process non-matching fingerprints
    for img_name in os.listdir(non_matching_dir):
        image_paths.append(os.path.join(non_matching_dir, img_name))
        labels.append(0)  # 0 for non-matching fingerprints

    train_indices, test_indices = train_test_split(list(range(len(image_paths))), test_size=test_size,
                                                   random_state=random_state)

    train_data_info = (train_indices, image_paths, labels)
    test_data_info = (test_indices, image_paths, labels)

    return train_data_info, test_data_info


if __name__ == "__main__":
    image_dir = "C:/Users/FinalProject/Desktop/LoRA/Dino/OurData"  # Replace with your dataset directory
    save_dir = "C:/Users/FinalProject/Desktop/LoRA/Dino/changes"  # Directory to save .pth files

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using dataset directory: {os.path.abspath(image_dir)}")
    train_data_info, test_data_info = prepare_data(image_dir)

    train_data_path = os.path.join(save_dir, 'train_data.pth')
    test_data_path = os.path.join(save_dir, 'test_data.pth')

    torch.save(train_data_info, train_data_path)
    torch.save(test_data_info, test_data_path)
    print(f"Data prepared and saved to {save_dir}.")
