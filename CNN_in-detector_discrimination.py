import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
from skimage.io import imread
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

TRAIN_IMAGE_DIR = "./train/images/"
TRAIN_LABEL_DIR = ".train/labels/"
TEST_IMAGE_DIR = ".test/images/"
TEST_LABEL_DIR = ".test/labels/"
SAVE_MODEL_PATH = "./model_saves/"


def metric_compute(y_prob, y_true, loss, threshold=0.5):
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:

        auc_roc = float('nan')
        print("Warning: Only one class present, ROC AUC is undefined")
    else:
        auc_roc = roc_auc_score(y_true, y_prob)

    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)

    return {
        "Loss": loss,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC_ROC": auc_roc
    }


class ResultLogger:
    def __init__(self, prefix="experiment"):
        # Timestamped file paths for logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"{SAVE_MODEL_PATH}{prefix}_{self.timestamp}.csv"
        self.plot_path = f"{SAVE_MODEL_PATH}{prefix}_curve_{self.timestamp}.png"
        self.results = []
        self.fieldnames = ['epoch']  # to maintain CSV header fields dynamically

    def log_epoch(self, epoch, train_result, test_result):
        """Record metrics for a single epoch (train & test) and save to CSV."""
        # Combine train and test metrics with prefixes
        combined = {'epoch': epoch}
        combined.update({f'train_{k}': v for k, v in train_result.items()})
        combined.update({f'test_{k}': v for k, v in test_result.items()})
        # Update fieldnames for any new metrics
        for key in combined.keys():
            if key not in self.fieldnames:
                self.fieldnames.append(key)
        # Append results and save the latest to CSV
        self.results.append(combined)
        self._save_to_csv()
        # Print metrics for this epoch
        print(f"\nEpoch {epoch} Results:")
        for metric in train_result.keys():
            train_val = train_result[metric]
            test_val = test_result.get(metric, None)
            if test_val is not None:
                print(f"  Train {metric}: {train_val:.4f} | Test {metric}: {test_val:.4f}")
            else:
                print(f"  Train {metric}: {train_val:.4f}")

    def plot_metrics(self):
        """Plot training curves for each metric and save the figure."""
        if not self.results:
            print("No data to plot!")
            return
        df = pd.DataFrame(self.results)
        # Identify metric names (without train/test prefix)
        metrics = [col.replace('train_', '') for col in df.columns if col.startswith('train_')]
        plt.figure(figsize=(11, 6))
        for i, metric in enumerate(metrics, start=1):
            plt.subplot(3, 2, i)
            if f"train_{metric}" in df.columns:
                plt.plot(df['epoch'], df[f'train_{metric}'], label=f'Train {metric}')
            if f"test_{metric}" in df.columns:
                plt.plot(df['epoch'], df[f'test_{metric}'], label=f'Test {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric} vs. Epoch')
            plt.legend()

        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.show()
        print(self.plot_path)

    def _save_to_csv(self):
        """Append the latest epoch results to the CSV file."""
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()  # write header once
            # Write only the newest result to avoid duplicates
            if self.results:
                writer.writerow(self.results[-1])


def weight_layer(x):
    weight = torch.ones(1, 64) / 64
    weight = weight.to(device)
    return x @ weight.T 

class MinutiaePresenceCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
           nn.Conv2d(1, 64, kernel_size=3, stride=3, padding=1),

        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze()



# === CONFIGURATION ===
NUM_EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 2e-3
TRAIN_SPLIT = 0.8
SAVE_MODEL_PATH = "./model_saves/"
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(device))

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)



# === Data Augmentation ===
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)), 
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.GaussianBlur(3)
])


# transform = None


class FingerprintDatasetBin(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = sorted([
            os.path.join(image_paths, f)
            for f in os.listdir(image_paths)
            if f.endswith(".tif")
        ])
        self.label_paths = sorted([
            os.path.join(label_paths, f)
            for f in os.listdir(label_paths)
            if f.endswith(".csv")
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = torch.tensor(imread(self.image_paths[idx]), dtype=torch.float32).unsqueeze(0) / 255.0
        label = np.loadtxt(self.label_paths[idx], delimiter=",")
        label = 1 if np.any(label > 1) else 0 
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# === Define DataLoaders ===
def create_dataloaders(train_images, train_labels, test_images, test_labels):
    train_dataset = FingerprintDatasetBin(train_images, train_labels, transform=transform)
    test_dataset = FingerprintDatasetBin(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False)

    return train_loader, test_loader


train_loader_1, test_loader_1 = create_dataloaders(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, TEST_IMAGE_DIR, TEST_LABEL_DIR)

# === Training & Evaluation Function ===


def train_and_evaluate_bin(model, train_loader, test_loader):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    logger = ResultLogger(prefix="fingerprint")
    patience = NUM_EPOCHS
    best_f1 = 0
    counter = 0
    true_threshold = 0.42
    best_ave_f1 = 0
    best_ave_epoch = 0
    best_epoch = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        all_train_probs = []
        all_train_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            # labels = labels.unsqueeze(1)
            outputs = torch.flatten(outputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # probs = torch.sigmoid(outputs).detach().cpu().numpy()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()  # predicted probability maps
            # Flatten and collect probabilities and true labels
            all_train_probs.append(probs.flatten())
            all_train_labels.append(labels.cpu().numpy().flatten())
        all_train_probs = np.concatenate(all_train_probs)
        all_train_labels = np.concatenate(all_train_labels)
        train_metrics = metric_compute(all_train_probs, all_train_labels, train_loss, threshold=true_threshold)
        # Testing
        model.eval()
        test_loss = 0
        all_test_probs = []
        all_test_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                # labels = labels.unsqueeze(1)
                outputs = model(images)
                outputs = torch.flatten(outputs)
                loss = criterion(outputs, labels.float())
                test_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_test_probs.append(probs.flatten())
                all_test_labels.append(labels.cpu().numpy().flatten())

        all_test_probs = np.concatenate(all_test_probs)
        all_test_labels = np.concatenate(all_test_labels)
        test_metrics = metric_compute(all_test_probs, all_test_labels, test_loss, threshold=true_threshold)
        if test_metrics['F1'] > 0.75 and train_metrics['F1'] > 0.75:
            torch.save(model.state_dict(), f'on_detector_75_f1_{epoch}_model_64.pth')
        if test_metrics['F1'] > best_f1:
            best_f1 = test_metrics['F1']
            best_epoch = epoch
            torch.save(model.state_dict(), 'on_detector_best_f1_testing_model_64.pth')
            counter = 0
        if (test_metrics['F1'] + train_metrics['F1']) / 2 > best_ave_f1:
            best_ave_f1 = (test_metrics['F1'] + train_metrics['F1']) / 2
            best_ave_epoch = epoch
            torch.save(model.state_dict(), 'on_detector_best_ave_f1_testing_model_64.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                print(best_f1)
                break
        logger.log_epoch(epoch, train_metrics, test_metrics)
    print('Best Test F1: ', best_f1, ' at ', best_epoch)
    print('Best Aver Test F1: ', best_ave_f1, ' at ', best_ave_epoch)
    logger.plot_metrics()

start_t = time.time()
train_and_evaluate_bin(MinutiaePresenceCNN(), train_loader_1, test_loader_1)
end_t = time.time()
print('total time: ', end_t - start_t)
