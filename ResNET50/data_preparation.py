# data_preparation.py
import os
import random
from itertools import combinations
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms

# Path to the extracted SOCOFing dataset
data_path = './SOCOFing/Real/'


# Function to prepare pairs
def prepare_pairs(data_path):
    fingerprints = [f for f in os.listdir(data_path) if f.endswith('.BMP')]

    # Group images by individual
    individuals = {}
    for fp in fingerprints:
        person_id = fp.split('__')[0]
        if person_id not in individuals:
            individuals[person_id] = []
        individuals[person_id].append(fp)

    # Create pairs
    same_pairs = []
    different_pairs = []

    for person_id, images in individuals.items():
        # Create matching pairs (same person, different fingers)
        same_pairs.extend([(img1, img2, 1) for img1, img2 in combinations(images, 2)])

        # Create non-matching pairs (different persons, random fingers)
        other_ids = list(individuals.keys())
        other_ids.remove(person_id)
        for other_id in random.sample(other_ids,
                                      min(5, len(other_ids))):  # limit to 5 different persons to keep it balanced
            different_pairs.extend([(img1, img2, 0) for img1 in images for img2 in individuals[other_id]])

    # Print the number of same pairs and different pairs
    print(f"Number of same pairs: {len(same_pairs)}")
    print(f"Number of different pairs: {len(different_pairs)}")

    # Balance the pairs by limiting the number of different pairs
    balanced_different_pairs = random.sample(different_pairs, len(same_pairs))
    print(f"Number of balanced different pairs: {len(balanced_different_pairs)}")

    # Combine and shuffle pairs
    pairs = same_pairs + balanced_different_pairs
    random.shuffle(pairs)

    # Split into training and testing sets
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    return train_pairs, test_pairs


# Custom Dataset Class
class FingerprintDataset(Dataset):
    def __init__(self, pairs, root_dir, transform=None):
        self.pairs = pairs
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(os.path.join(self.root_dir, img1_path)).convert('RGB')
        img2 = Image.open(os.path.join(self.root_dir, img2_path)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), torch.tensor(label, dtype=torch.float)


# Transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(batch_size=32):
    train_pairs, test_pairs = prepare_pairs(data_path)

    # Create datasets
    trainset = FingerprintDataset(pairs=train_pairs, root_dir=data_path, transform=transform)
    testset = FingerprintDataset(pairs=test_pairs, root_dir=data_path, transform=transform)

    # DataLoader
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
