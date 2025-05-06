import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from itertools import combinations
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm




def load_and_preprocess_data(file_path, normalize=True, standardize=False):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    if normalize:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def generate_bba_simple_improved(x, class_label):

    bba_list = []
    for feature_val in x:
        bba = {}

        belief_class = torch.tensor(0.9 * feature_val, dtype=torch.float32)

        belief_H = torch.tensor(max(0.0, 1.0 - belief_class.item() - 0.1), dtype=torch.float32)

        total_belief = belief_class + belief_H
        if total_belief > 1.0:
            belief_class = belief_class / total_belief
            belief_H = belief_H / total_belief


        focal_element_class = str(class_label)
        focal_element_H = "H"

        bba[focal_element_class] = belief_class
        bba[focal_element_H] = belief_H
        bba_list.append(bba)
    return bba_list

def convert_bba_to_tensor(bba_list, num_classes):

    all_focal_elements = [str(c) for c in range(num_classes)] + ["H"]
    num_focal_elements = len(all_focal_elements)
    num_features = len(bba_list)
    bba_tensor = torch.zeros((num_features, num_focal_elements), dtype=torch.float32)

    for i, bba in enumerate(bba_list):
        for j, focal_element in enumerate(all_focal_elements):
            if focal_element in bba:
                bba_tensor[i, j] = bba[focal_element]
    return bba_tensor


def dynamic_fractal_probability_transform(bba_tensor_in, lmbda):
    num_features, num_classes = bba_tensor_in.shape
    current_masses = {}
    final_result = torch.zeros_like(bba_tensor_in)
    for j in range(num_features):
        bba_tensor = bba_tensor_in[j]
        for i in range(num_classes-1):
            current_masses[frozenset([f'a{i + 1}'])] = bba_tensor[i]
        current_masses[frozenset(['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'])] = bba_tensor[-1]
        all_results = []

        all_results.append(current_masses.copy())

        for t in range(1, int(lmbda) + 1):
            new_masses = {}
            total_mass = 0.0

            # 遍历所有子集
            for Ai in current_masses:
                new_mass = 0.0
                for Aj in current_masses:
                    if Ai.issubset(Aj):
                        tau_power = lmbda ** (len(Aj) - len(Ai))
                        D = (len(Ai) / len(Aj)) ** (1 / lmbda)
                        denominator = (t + 1) ** len(Aj) - t ** len(Aj)
                        new_mass += tau_power * D / denominator * current_masses[Aj]
                new_masses[Ai] = new_mass
                total_mass += new_mass

            normalized_masses = {key: value / total_mass for key, value in new_masses.items()}
            all_results.append(normalized_masses.copy())
            current_masses = normalized_masses

        mid_result = torch.zeros_like(bba_tensor)
        for i in range(num_classes-1):
            mid_result[i] = current_masses[frozenset([f'a{i + 1}'])]
        mid_result[-1] = current_masses[frozenset(['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'])]
        final_result[j] = mid_result
    return final_result


def calculate_discount_factors(bba_tensor, labels, sigma):

    num_features = bba_tensor.shape[0]
    num_samples = labels.shape[0]


    bba_probs = torch.softmax(bba_tensor, dim=1)

    # 将标签转换为 one-hot 编码
    labels_onehot = F.one_hot(labels, num_classes=bba_probs.shape[1]).float()

    discount_factors = torch.ones(num_features, dtype=torch.float32)

    for i in range(num_features):
        x = bba_probs[i].unsqueeze(0).repeat(num_samples, 1)
        y = labels_onehot
        mmd_loss = compute_mmd(x, y, sigma, kernel='rbf')
        discount_factors[i] = 1.0 / (1 + mmd_loss)

    discount_factors = torch.where(discount_factors < 0, torch.tensor(1e-6, dtype=torch.float32), discount_factors)

    return discount_factors


def compute_mmd(x, y, sigma, kernel='rbf'):

    xx = torch.cdist(x, x, p=2)
    yy = torch.cdist(y, y, p=2)
    xy = torch.cdist(x, y, p=2)

    if kernel == 'rbf':
        kxx = torch.exp(-xx / (2 * sigma ** 2))
        kyy = torch.exp(-yy / (2 * sigma ** 2))
        kxy = torch.exp(-xy / (2 * sigma ** 2))
    elif kernel == 'linear':
        kxx = xx
        kyy = yy
        kxy = xy
    elif kernel == 'polynomial':
        kxx = (xx + 1) ** 2
        kyy = (yy + 1) ** 2
        kxy = (xy + 1) ** 2
    else:
        raise ValueError("Invalid kernel type.")

    m = x.shape[0]
    n = y.shape[0]

    # 计算mmd
    if m > 1:
        kxx_sum = kxx.sum() / (m * (m - 1))
    else:
        kxx_sum = 0.0

    if n > 1:
        kyy_sum = kyy.sum() / (n * (n - 1))
    else:
        kyy_sum = 0.0

    kxy_sum = kxy.sum() / (m * n)

    mmd = kxx_sum - 2 * kxy_sum + kyy_sum
    return mmd


def discount_bba(bba_tensor, discount_factors):
    discounted_bba_tensor = bba_tensor * discount_factors.unsqueeze(1)
    return discounted_bba_tensor


class DS3_Dempster(torch.nn.Module):


    def __init__(self, n_prototypes, num_class):
        super(DS3_Dempster, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        m1 = inputs[..., 0, :]
        omega1 = torch.unsqueeze(inputs[..., 0, -1], -1)
        for i in range(self.n_prototypes - 1):
            m2 = inputs[..., (i + 1), :]
            omega2 = torch.unsqueeze(inputs[..., (i + 1), -1], -1)
            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = combine1 + combine2
            combine2_3 = combine1_2 + combine3
            combine2_3 = combine2_3 / torch.sum(combine2_3, dim=-1, keepdim=True)
            m1 = combine2_3
            omega1 = torch.unsqueeze(combine2_3[..., -1], -1)
        return m1


def fuse_bbas(discounted_bba_tensor):
    inputs = discounted_bba_tensor.unsqueeze(0)

    num_classes = 0
    for i in discounted_bba_tensor[0]:
        if i != 0:
            num_classes += 1

    dempster_fusion = DS3_Dempster(n_prototypes=discounted_bba_tensor.shape[0], num_class=num_classes)

    # 融合
    fused_bba = dempster_fusion(inputs)
    return fused_bba.squeeze(0)



def make_decision(fused_bba):

    predicted_class = torch.argmax(fused_bba).item()
    return predicted_class


class DSTClassifier(nn.Module):
    def __init__(self, num_class, tau, sigma):
        super(DSTClassifier, self).__init__()
        self.model = FitNet()
        self.fc = nn.Linear(128, num_class)


    def forward(self, data, target):
        features = self.model(data)
        a = self.fc(features)
        return features


class FitNet(nn.Module):
    def __init__(self, pretrained=False):
        super(FitNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv1_4 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)

        self.conv2_1 = nn.Conv2d(48, 80, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(80, 80, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(80, 80, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv2d(80, 80, kernel_size=3, padding=1)
        self.conv2_5 = nn.Conv2d(80, 80, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(80)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.5)

        self.conv3_1 = nn.Conv2d(80, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((8, 8))
        self.dropout3 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def forward_features(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = F.relu(self.conv1_4(x))
        x = F.relu(self.conv1_5(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.relu(self.conv2_4(x))
        x = F.relu(self.conv2_5(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.relu(self.conv3_4(x))
        x = F.relu(self.conv3_5(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        return x

# --- 主函数 ---
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./cifar10', train=False, download=True, transform=transform)

    train_loader_full = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    test_loader_full = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    model = DSTClassifier(10, 1, 0.5)
    weight_path = r'.\checkpoints\CNN_CIFAR10_best.weights.pth'
    loaded_weights = torch.load(weight_path)
    model.load_state_dict(loaded_weights)
    model.cuda()
    model.eval()


    all_train_data, all_train_targets = next(iter(train_loader_full))
    all_train_data = all_train_data.cpu().numpy()
    all_train_targets = all_train_targets.cpu().numpy()


    all_test_data, all_test_targets = next(iter(test_loader_full))
    all_test_data = all_test_data.cpu().numpy()
    all_test_targets = all_test_targets.cpu().numpy()

    num_classes = 10
    lmbda = 2.0
    sigma = 0.5

    train_batch_size = 50
    test_batch_size = 10

    all_fold_accuracies = []
    num_train_batches = len(all_train_data) // train_batch_size + (1 if len(all_train_data) % train_batch_size != 0 else 0)
    num_test_batches = len(all_test_data) // test_batch_size + (1 if len(all_test_data) % test_batch_size != 0 else 0)


    if num_train_batches < num_test_batches:
        raise ValueError("Number of training batches must be greater than or equal to the number of testing batches under this configuration.")

    for k in range(num_test_batches):
        start_test = k * test_batch_size
        end_test = min((k + 1) * test_batch_size, len(all_test_data))
        X_test_batch = all_test_data.astype(np.float32)[start_test:end_test]
        y_test_batch = all_test_targets.astype(np.int64)[start_test:end_test]
        y_test_batch_tensor = torch.tensor(y_test_batch).cuda()
        X_test_batch_tensor = torch.tensor(X_test_batch).cuda()

        print(f"Processing test batch {k + 1}/{num_test_batches}")

        # 找到对应的训练数据批次
        train_batch_index = k % num_train_batches # 使用取余来循环训练批次，如果测试批次更多
        start_train = train_batch_index * train_batch_size
        end_train = min((train_batch_index + 1) * train_batch_size, len(all_train_data))
        X_train_batch = all_train_data.astype(np.float32)[start_train:end_train]
        y_train_batch = all_train_targets.astype(np.int64)[start_train:end_train]
        y_train_batch_tensor = torch.tensor(y_train_batch).cuda()
        X_train_batch_tensor = torch.tensor(X_train_batch).cuda()

        print(f"  Using training batch {train_batch_index + 1}/{num_train_batches} to calculate discount factors.")

        bba_list_train_batch = []
        with torch.no_grad():
            output_train = model(X_train_batch_tensor, y_train_batch_tensor)
            X_train_batch_features = output_train.cpu().detach().numpy()
            X_train_batch_features = np.array(X_train_batch_features, dtype=np.float32)

            for j in range(len(X_train_batch_features)):
                bba_list_train_batch.extend(
                    generate_bba_simple_improved(X_train_batch_features[j], y_train_batch[j]))

        bba_tensor_train_batch = convert_bba_to_tensor(bba_list_train_batch, num_classes)

        transformed_bba_tensor_train_batch = dynamic_fractal_probability_transform(bba_tensor_train_batch, lmbda)

        discount_factors_train_batch = calculate_discount_factors(transformed_bba_tensor_train_batch,
                                                                  torch.tensor(y_train_batch), sigma)

        num_features = X_train_batch_features.shape[-1]
        discount_factors_train_reshape = torch.mean(discount_factors_train_batch.reshape(-1, num_features), dim=0)

        batch_fold_accuracies = []

        with torch.no_grad():
            output_test = model(X_test_batch_tensor, y_test_batch_tensor)
            X_test_batch_features = output_test.cpu().detach().numpy()
            X_test_batch_features = np.array(X_test_batch_features, dtype=np.float64)

            for l in range(len(X_test_batch_features)):
                x_test_sample = X_test_batch_features[l]
                y_true = y_test_batch[l].item()

                bba_list_test = generate_bba_simple_improved(x_test_sample, y_true)

                bba_tensor_test = convert_bba_to_tensor(bba_list_test, num_classes)

                transformed_bba_tensor_test = dynamic_fractal_probability_transform(bba_tensor_test, lmbda)

                discounted_bba_tensor_test = discount_bba(transformed_bba_tensor_test,
                                                          discount_factors_train_reshape) # 确保折扣因子在 GPU 上

                fused_bba_test = fuse_bbas(discounted_bba_tensor_test)

                predicted_classes = make_decision(fused_bba_test)

                if predicted_classes == y_true:
                    batch_fold_accuracies.append(1)
                else:
                    batch_fold_accuracies.append(0)

        avg_accuracy = np.mean(batch_fold_accuracies) if batch_fold_accuracies else 0.0
        all_fold_accuracies.append(avg_accuracy)
        print(f"  Accuracy for this test batch: {avg_accuracy:.4f}")

    overall_accuracy = np.mean(all_fold_accuracies) if all_fold_accuracies else 0.0
    print(f"\nFinal Overall Average Accuracy: {overall_accuracy:.4f}")
    print("Accuracies for each test batch (corresponding to a training batch):", all_fold_accuracies)


if __name__ == "__main__":
    main()