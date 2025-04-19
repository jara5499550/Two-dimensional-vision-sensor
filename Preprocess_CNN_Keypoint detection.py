import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from skimage.io import imread
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# === CONFIGURATION ===
NUM_EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 6e-4
TRAIN_SPLIT = 0.75
SAVE_MODEL_PATH = "./model_saves/"

# === Loss Function & Optimizer ===
alpha = torch.tensor([0.0015, 0.3085, 0.69], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=alpha)

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)

# === DIRECTORIES ===
IMAGE_DIR_1 = "./Laplacian_Kernel/"
IMAGE_DIR_2 = "./Average_Kernel/"
IMAGE_DIR_3 = "./0416_dataset/Laplacian_Kernel_simu/"
LABEL_DIR = "./Labels/"

# === CNN Model Definition ===
class MinutiaeCNN(nn.Module):
    def __init__(self):
        super(MinutiaeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.conv3(x)
        return self.softmax(x)


# === Data Augmentation ===
transform = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])


# === Dataset Loader ===
class FingerprintDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = imread(self.image_paths[idx])
        label = np.loadtxt(self.label_paths[idx], delimiter=",")

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
        valid_mask = (label != -1)
        label[label == -1] = 0
        label_one_hot = np.zeros((3, 7, 7))
        for i in range(7):
            for j in range(7):
                label_one_hot[int(label[i, j]), i, j] = 1

        label_tensor = torch.tensor(label_one_hot, dtype=torch.float32).to(device)
        valid_mask = torch.tensor(valid_mask, dtype=torch.float32).to(device)
        valid_mask = valid_mask.unsqueeze(0)  # add one more axis
        if self.transform:
            torch.manual_seed(idx)
            image = self.transform(image)
            label_tensor = self.transform(label_tensor)
            valid_mask = self.transform(valid_mask)
        return image, label_tensor, valid_mask


class FingerprintDatasetBin(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths  # 0: No Minutiae, 1: Minutiae
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = torch.tensor(imread(self.image_paths[idx]), dtype=torch.float32).unsqueeze(0) / 255.0
        label = np.loadtxt(self.label_paths[idx], delimiter=",")
        if np.any(label > 0):
            label = torch.tensor(1, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


# === Load and Split Data ===
def prepare_data(image_dir):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".tif")])
    label_paths = sorted([os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith(".csv")])

    train_images, test_images, train_labels, test_labels = train_test_split(
        image_paths, label_paths, train_size=TRAIN_SPLIT, random_state=42, shuffle=True
    )

    return train_images, test_images, train_labels, test_labels


train_images_1, test_images_1, train_labels_1, test_labels_1 = prepare_data(IMAGE_DIR_1)
train_images_2, test_images_2, train_labels_2, test_labels_2 = prepare_data(IMAGE_DIR_2)
train_images_3, test_images_3, train_labels_3, test_labels_3 = prepare_data(IMAGE_DIR_3)


# === Define DataLoaders ===
def create_dataloaders(train_images, train_labels, test_images, test_labels):
    train_dataset = FingerprintDataset(train_images, train_labels, transform=transform)
    test_dataset = FingerprintDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False)

    return train_loader, test_loader


train_loader_1, test_loader_1 = create_dataloaders(train_images_1, train_labels_1, test_images_1, test_labels_1)
train_loader_2, test_loader_2 = create_dataloaders(train_images_2, train_labels_2, test_images_2, test_labels_2)
train_loader_3, test_loader_3 = create_dataloaders(train_images_3, train_labels_3, test_images_3, test_labels_3)


def plot_weighted_accuracy(history, kernel_name):
    """
    Plot the weighted training and testing accuracy after training.
    Args:
        history (dict): Dictionary containing 'train_weighted_acc' and 'test_weighted_acc'.
        kernel_name (str): Name of the kernel (for plot title).
    """
    epochs = history['epoch']
    train_acc = history['train_weighted_acc']
    test_acc = history['test_weighted_acc']
    train_loss = history['train_loss']
    test_loss = history['test_loss']

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_acc, label='Training Weighted Accuracy')
    plt.plot(epochs, test_acc, label='Testing Weighted Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Weighted Accuracy')
    plt.title(f'Weighted Accuracy During Training ({kernel_name})')
    plt.legend(loc='best')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, test_loss, label='Testing Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss During Training ({kernel_name})')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./model_saves/weighted_accuracy_plot_{kernel_name}.png')


# === Training & Evaluation Function ===
def train_and_evaluate(model, train_loader, test_loader, kernel_name, save_result_name):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    history = {"epoch": [], "train_loss": [], "test_loss": [],
               "train_acc": [], "test_acc": [],
               "train_weighted_acc": [], "test_weighted_acc": [],
               "train_acc_class_0": [], "train_acc_class_1": [], "train_acc_class_2": [],
               "test_acc_class_0": [], "test_acc_class_1": [],
               "test_acc_class_2": []}
    max_train_weight_acc = 0
    max_recall = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 1e-8
        train_preds = []
        train_targets = []

        for images, labels, valid_mask in train_loader:
            images, labels = images.to(device), labels.to(device)
            valid_mask = valid_mask.to(device)
            valid_mask = torch.squeeze(valid_mask, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            ground_truth = torch.argmax(labels, dim=1)

            batch_weights = alpha[ground_truth]
            temp_correct_train = (predicted == ground_truth).float() * batch_weights
            temp_correct_train *= valid_mask
            temp_correct_train = temp_correct_train.sum()
            correct_train += temp_correct_train

            total_train += (batch_weights * valid_mask).sum()

            train_preds.extend(predicted.cpu().numpy().flatten())
            train_targets.extend(ground_truth.cpu().numpy().flatten())

        avg_train_loss = total_loss / len(train_loader)
        weighted_train_acc = correct_train.item() / total_train.item()
        if weighted_train_acc > max_train_weight_acc:
            max_train_weight_acc = weighted_train_acc
            torch.save(model, SAVE_MODEL_PATH + save_result_name + '_model.pth')
        conf_matrix_train = confusion_matrix(train_targets, train_preds, labels=[0, 1, 2])
        per_class_train_acc = conf_matrix_train.diagonal() / conf_matrix_train.sum(axis=1)
        overall_train_acc = np.mean(np.array(train_preds) == np.array(train_targets))
        # Testing
        model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []
        correct_test = 0
        total_test = 1e-8

        with torch.no_grad():
            for images, labels, valid_mask in test_loader:
                images, labels = images.to(device), labels.to(device)
                valid_mask = valid_mask.to(device)
                valid_mask = torch.squeeze(valid_mask, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                ground_truth = torch.argmax(labels, dim=1)

                all_preds.extend(predicted.cpu().numpy().flatten())
                all_targets.extend(ground_truth.cpu().numpy().flatten())
                batch_weights = alpha[ground_truth]

                temp_correct_test = (predicted == ground_truth).float() * batch_weights
                temp_correct_test *= valid_mask
                temp_correct_test = temp_correct_test.sum()
                correct_test += temp_correct_test

                total_test += (batch_weights * valid_mask).sum()

        avg_test_loss = test_loss / len(test_loader)
        weighted_test_acc = correct_test.item() / total_test.item()
        overall_test_acc = np.mean(np.array(all_preds) == np.array(all_targets))

        # Per-Class Accuracy
        conf_matrix = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        recall_train = np.mean(per_class_train_acc)
        recall_test = np.mean(per_class_acc)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)
        history["train_acc"].append(overall_train_acc)
        history["test_acc"].append(overall_test_acc)
        history["train_weighted_acc"].append(recall_train)
        history["test_weighted_acc"].append(recall_test)
        history["train_acc_class_0"].append(per_class_train_acc[0])
        history["train_acc_class_1"].append(per_class_train_acc[1])
        history["train_acc_class_2"].append(per_class_train_acc[2])
        history["test_acc_class_0"].append(per_class_acc[0])
        history["test_acc_class_1"].append(per_class_acc[1])
        history["test_acc_class_2"].append(per_class_acc[2])

        # Print Results

        print(f"{kernel_name} | Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Test Loss: {avg_test_loss:.6f}")
        print(f"  Weighted Train Accuracy: {weighted_train_acc:.4f}")
        print('Average recall training: ', recall_train)
        print('Average recall testing: ', recall_test)
        print(f"  Weighted Test Accuracy: {weighted_test_acc:.4f}")
        print(
            f"  Train Class 0 Accuracy: {per_class_train_acc[0]:.4f}, Class 1: {per_class_train_acc[1]:.4f}, Class 2: {per_class_train_acc[2]:.4f}")
        print(
            f"  Test Class 0 Accuracy: {per_class_acc[0]:.4f}, Class 1: {per_class_acc[1]:.4f}, Class 2: {per_class_acc[2]:.4f}")
        print()
        if recall_train + recall_test > max_recall:
            max_recall = recall_train + recall_test
            print("current best: ", recall_train, recall_test)
            print()
            torch.save(model, SAVE_MODEL_PATH + save_result_name + 'best_recall_model.pth')

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(SAVE_MODEL_PATH, f"training_results_{save_result_name}.csv"), index=False)
    print('max training accuracy: ', max_train_weight_acc)
    plot_weighted_accuracy(history, kernel_name)

start_t = time.time()
# Train both models
train_and_evaluate(MinutiaeCNN(), train_loader_1, test_loader_1, "Laplacian_Kernel",
                   "LK1_" + str(LEARNING_RATE) + '0416_whole_frame_no drop')

train_and_evaluate(MinutiaeCNN(), train_loader_2, test_loader_2, "Average_Kernel",
                   "AV1_" + str(LEARNING_RATE) + '0416_whole_frame_no drop')
train_and_evaluate(MinutiaeCNN(), train_loader_3, test_loader_3, "Simulate Laplacian_Kernel",
                   "sLK1_" + str(LEARNING_RATE) + '0416_whole_frame_no drop')
end_t = time.time()
print('total time: ', end_t - start_t)
