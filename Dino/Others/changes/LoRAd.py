# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset, Subset
# from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms as transforms
# from PIL import Image
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from peft import LoraConfig, get_peft_model
# from sklearn.model_selection import train_test_split
#
#
# class FingerprintDataset(Dataset):
#     def __init__(self, image_dir, image_paths, labels, transform=None):
#         self.image_dir = os.path.abspath(image_dir)
#         self.transform = transform
#         self.image_paths = image_paths
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path).convert("RGB")
#         label = self.labels[idx]
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image, label
#
#
# def prepare_data(image_dir, test_size=0.2, random_state=42):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     image_paths = []
#     labels = []
#
#     # Assuming 'matching' and 'non_matching' subdirectories exist in the image_dir
#     matching_dir = os.path.join(image_dir, 'class_0')
#     non_matching_dir = os.path.join(image_dir, 'class_1')
#
#     print(f"Checking directory: {matching_dir}")
#     print(f"Checking directory: {non_matching_dir}")
#
#     if not os.path.exists(matching_dir) or not os.path.exists(non_matching_dir):
#         raise FileNotFoundError(
#             f"The directories for matching or non_matching images do not exist.\nChecked paths:\n{matching_dir}\n{non_matching_dir}")
#
#     # Process matching fingerprints
#     for img_name in os.listdir(matching_dir):
#         image_paths.append(os.path.join(matching_dir, img_name))
#         labels.append(1)  # 1 for matching fingerprints
#
#     # Process non-matching fingerprints
#     for img_name in os.listdir(non_matching_dir):
#         image_paths.append(os.path.join(non_matching_dir, img_name))
#         labels.append(0)  # 0 for non-matching fingerprints
#
#     train_indices, test_indices = train_test_split(list(range(len(image_paths))), test_size=test_size,
#                                                    random_state=random_state)
#
#     train_data_info = (train_indices, image_paths, labels)
#     test_data_info = (test_indices, image_paths, labels)
#
#     return train_data_info, test_data_info
#
#
# def load_data(train_path, test_path, image_dir):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     train_indices, image_paths, labels = torch.load(train_path)
#     test_indices, _, _ = torch.load(test_path)
#
#     train_data = Subset(FingerprintDataset(image_dir, image_paths, labels, transform), train_indices)
#     test_data = Subset(FingerprintDataset(image_dir, image_paths, labels, transform), test_indices)
#
#     return train_data, test_data
#
#
# if __name__ == "__main__":
#     image_dir = "C:/Users/FinalProject/Desktop/LoRA/Dino/OurData"
#     save_dir = "C:/Users/FinalProject/Desktop/LoRA/Dino/changes"  # Directory to save .pth files
#
#     # Ensure the save directory exists
#     os.makedirs(save_dir, exist_ok=True)
#
#     print(f"Using dataset directory: {os.path.abspath(image_dir)}")
#     train_data_info, test_data_info = prepare_data(image_dir)
#
#     train_data_path = os.path.join(save_dir, 'train_data.pth')
#     test_data_path = os.path.join(save_dir, 'test_data.pth')
#
#     torch.save(train_data_info, train_data_path)
#     torch.save(test_data_info, test_data_path)
#     print(f"Data prepared and saved to {save_dir}.")
#
#     # Load Training & Test DataSets
#     train_data, test_data = load_data(train_data_path, test_data_path, image_dir)
#
#     # Set Device
#     os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Hyperparameters
#     batch_size = 64
#     epochs = 5
#     learning_rate = 0.001
#
#     # Data Loader
#     trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
#     testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     # Load a pre-trained model and feature extractor
#     processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
#     model = AutoModelForImageClassification.from_pretrained('facebook/dinov2-base')
#
#     # Freeze all the parameters in the model
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # Modify the model for embedding extraction
#     model.classifier = torch.nn.Identity()
#
#     # Apply LoRA using the PEFT library
#     config = LoraConfig(
#         r=16,
#         lora_alpha=16,
#         target_modules=["query", "value"],
#         lora_dropout=0.1,
#         bias="none",
#         modules_to_save=["classifier"],
#     )
#     lora_model = get_peft_model(model, config)
#     lora_model.print_trainable_parameters()
#     _ = lora_model.to(device)
#
#
#     # Siamese Network Architecture
#     class SiameseNetwork(nn.Module):
#         def __init__(self, base_model):
#             super(SiameseNetwork, self).__init__()
#             self.base_model = base_model
#
#         def forward(self, x1, x2):
#             output1 = self.base_model(x1).logits  # Assuming logits is the final output of the model
#             output2 = self.base_model(x2).logits
#             return output1, output2
#
#
#     # Instantiate the Siamese Network
#     siamese_network = SiameseNetwork(lora_model).to(device)
#
#
#     # Contrastive Loss Function
#     class ContrastiveLoss(nn.Module):
#         def __init__(self, margin=1.0):
#             super(ContrastiveLoss, self).__init__()
#             self.margin = margin
#
#         def forward(self, output1, output2, label):
#             euclidean_distance = nn.functional.pairwise_distance(output1, output2)
#             loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
#                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#             return loss
#
#
#     # Loss Function and Optimizer
#     criterion = ContrastiveLoss()
#     optimizer = optim.SGD(siamese_network.parameters(), lr=learning_rate, momentum=0.9)
#     torch.cuda.empty_cache()
#
#     # Initialize TensorBoard
#     writer = SummaryWriter(os.path.join(save_dir, 'runs/LoRAd'))
#
#     # Training The Model
#     for epoch in range(epochs):
#         siamese_network.train()  # Set the model to training mode
#         running_loss = 0.0
#
#         for i, data in enumerate(trainloader, 0):
#             img1, labels = data[0].to(device), data[1].to(device)
#             optimizer.zero_grad()
#             output1, output2 = siamese_network(img1, img1)
#             loss = criterion(output1, output2, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         epoch_loss = running_loss / len(trainloader)
#
#         siamese_network.eval()  # Set the model to evaluation mode
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for data in testloader:
#                 img1, labels = data[0].to(device), data[1].to(device)
#                 output1, output2 = siamese_network(img1, img1)
#                 euclidean_distance = nn.functional.pairwise_distance(output1, output2)
#                 predicted = (euclidean_distance < 0.5).float()
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)
#             epoch_accuracy = 100 * correct / total
#
#             writer.add_scalar('Accuracy/test', epoch_accuracy, epoch)
#             writer.add_scalar('Loss/train', epoch_loss, epoch)
#
#             print(f'Accuracy: {epoch_accuracy:.2f}%  Loss: {epoch_loss:.3f}')
#
#     print('Finished Training')
#     writer.close()


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split


class FingerprintPairDataset(Dataset):
    def __init__(self, image_dir, pairs, labels, transform=None):
        self.image_dir = os.path.abspath(image_dir)
        self.transform = transform
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path1, image_path2 = self.pairs[idx]
        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label


def prepare_data(image_dir, test_size=0.2, random_state=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    matching_dir = os.path.join(image_dir, 'class_0')
    non_matching_dir = os.path.join(image_dir, 'class_1')

    if not os.path.exists(matching_dir) or not os.path.exists(non_matching_dir):
        raise FileNotFoundError("The directories for matching or non_matching images do not exist.")

    pairs = []
    labels = []

    for img_name in os.listdir(matching_dir):
        pairs.append((os.path.join(matching_dir, img_name), os.path.join(matching_dir, img_name)))
        labels.append(1)

    for img_name in os.listdir(non_matching_dir):
        pairs.append((os.path.join(non_matching_dir, img_name), os.path.join(non_matching_dir, img_name)))
        labels.append(0)

    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, test_size=test_size, random_state=random_state
    )

    torch.save((train_pairs, train_labels), 'C:/Users/FinalProject/Desktop/LoRA/Dino/changes/train_pairs.pth')
    torch.save((test_pairs, test_labels), 'C:/Users/FinalProject/Desktop/LoRA/Dino/changes/test_pairs.pth')


def load_data(train_path, test_path, image_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_pairs, train_labels = torch.load(train_path)
    test_pairs, test_labels = torch.load(test_path)

    train_data = FingerprintPairDataset(image_dir, train_pairs, train_labels, transform)
    test_data = FingerprintPairDataset(image_dir, test_pairs, test_labels, transform)

    return train_data, test_data


class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model

    def forward(self, x1, x2):
        output1 = self.base_model(x1).logits
        output2 = self.base_model(x2).logits
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


if __name__ == '__main__':
    # Prepare the data
    image_dir = "C:/Users/FinalProject/Desktop/LoRA/Dino/OurData"
    print(f"Using dataset directory: {os.path.abspath(image_dir)}")
    prepare_data(image_dir)
    print("Data prepared and saved.")

    # Load the data
    train_data, test_data = load_data('C:/Users/FinalProject/Desktop/LoRA/Dino/changes/train_pairs.pth',
                                      'C:/Users/FinalProject/Desktop/LoRA/Dino/changes/test_pairs.pth', image_dir)

    # Set Device
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 64
    epochs = 5
    learning_rate = 0.001

    # Data Loader
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load a pre-trained model and feature extractor
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModelForImageClassification.from_pretrained('facebook/dinov2-base')

    # Freeze all the parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Modify the model for embedding extraction
    model.classifier = torch.nn.Identity()

    # Apply LoRA using the PEFT library
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    _ = lora_model.to(device)

    # Instantiate the Siamese Network
    siamese_network = SiameseNetwork(lora_model).to(device)

    # Loss Function and Optimizer
    criterion = ContrastiveLoss()
    optimizer = optim.SGD(siamese_network.parameters(), lr=learning_rate, momentum=0.9)
    torch.cuda.empty_cache()

    # Initialize TensorBoard
    writer = SummaryWriter('C:/Users/FinalProject/Desktop/LoRA/Dino/changes/runs/LoRAd')

    # Training The Model
    for epoch in range(epochs):
        siamese_network.train()  # Set the model to training mode
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            img1, img2, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            output1, output2 = siamese_network(img1, img2)
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)

        siamese_network.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                img1, img2, labels = data[0].to(device), data[1].to(device), data[2].to(device)
                output1, output2 = siamese_network(img1, img2)
                euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                predicted = (euclidean_distance < 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            epoch_accuracy = 100 * correct / total

            writer.add_scalar('Accuracy/test', epoch_accuracy, epoch)
            writer.add_scalar('Loss/train', epoch_loss, epoch)

            print(f'Accuracy: {epoch_accuracy:.2f}%  Loss: {epoch_loss:.3f}')

    print('Finished Training')
    writer.close()

