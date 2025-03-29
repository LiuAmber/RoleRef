# import sys
# sys.path.append('/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/roleplay_refuse_answer/code')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from tqdm import tqdm
import random
import pickle as pkl

import pandas as pd
# from prompt import PROFILE
import os
import logging
import argparse

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Define model
class SimpleFCNet(nn.Module):
    def __init__(self,input_dim = 4096):
        super(SimpleFCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(512, 1)  # Change output to 1 for regression
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
        return x

# Step 2: Data define
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item['data'], dtype=torch.float), torch.tensor(item['label'], dtype=torch.float)  # Change label to float

def load_dataset(data_path,layer,keys):
    train_dataset = []
    # 
    all_keys = ["answer","out_series","context_conflict","absent","fake"]
    for key in keys:
        with open(f"{data_path}/rep_{key}.pkl","rb") as f:
            dataset = pkl.load(f)
        for data in dataset:
            train_dataset.append(
                {
                "data": data[layer],
                "label": 1 if key == "answer" else 0
                }
            )
    if "test" in data_path:
        test_dataset = {}
        for key in all_keys:
            test_dataset[key] = []
            with open(f"{data_path}/rep_{key}.pkl","rb") as f:
                dataset = pkl.load(f)
            for data in dataset:
                test_dataset[key].append(
                    {
                    "data": data[layer],
                    "label": 1 if key == "answer" else 0
                    }
                )
    else:
        test_dataset = {}
    return train_dataset,test_dataset

def train(layer, role,data_path,keys):
    train_data_path = f"{data_path}/train/{role}"
    validation_data_path = f"{data_path}/test/{role}"
    train_data,_ = load_dataset(train_data_path,layer,keys)
    validation_data,test_data = load_dataset(validation_data_path,layer,keys)

    train_dataset = CustomDataset(train_data)
    validation_dataset = CustomDataset(validation_data)
    

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=512)
    test_loader = {}
    for key in test_data:
        test_dataset = CustomDataset(test_data[key])
        test_loader[key] = DataLoader(test_dataset, batch_size=512)
        

    # Step 3: Model training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "qwen" in data_path:
        model = SimpleFCNet(input_dim = 3584).to(device)
    else:
        model = SimpleFCNet(input_dim = 4096).to(device)
    criterion = nn.MSELoss()  # Change to MSELoss
    
    # lr = 0.00001
    lr = 0.00005
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0  # Initialize best validation accuracy

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels.unsqueeze(1))  # Adjust shape for MSELoss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validate model
        model.eval()
        train_total = 0
        train_correct = 0
        with torch.no_grad():
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted.squeeze() == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total

        total = 0
        correct = 0
        with torch.no_grad():
            for data, labels in validation_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

        val_accuracy = 100 * correct / total

        if val_accuracy > best_val_acc:
            # test_part
            with torch.no_grad():
                result = {}
                for key in test_loader:
                    test_total = 0
                    test_correct = 0
                    
                    for data, labels in test_loader[key]:
                        data, labels = data.to(device), labels.to(device)
                        outputs = model(data)
                        predicted = (outputs > 0.5).float()
                        test_total += labels.size(0)
                        test_correct += (predicted.squeeze() == labels).sum().item() 
                    try:
                        test_accuracy = 100 * test_correct / test_total
                    except:
                        test_accuracy = 0
                    result[key] = test_accuracy
                    
            best_val_acc = val_accuracy
            best_train_acc = train_accuracy
            best_epoch = epoch
            # torch.save(model.state_dict(), f"/apdcephfs_cq10/share_2992827/siyuan/leoleoliu/research/code/reject_answer/code/one_stage/representation/model/classify/Hermione Granger/mse_loss/{layer}.pth")
            # print(f'Epoch {best_epoch+1}, Train Accuracy: {best_train_acc}%, Validation Accuracy: {best_val_acc}%')
    result_dict = {
        'role': role,
        'layer': layer,
        'train_acc': best_train_acc,
        'validation_acc': best_val_acc,
            }
    for key in result:
        dict_name = f"{key}_acc"
        result_dict[dict_name] = result[key]
    return result_dict

# import json
def run(data_path,seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    all_keys = ["answer","out_series","context_conflict","absent","fake"]
    # keys_list = [["answer","out_series"],["answer","fake"],["answer","fake","out_series"]]
    keys_list = [["answer","fake","out_series"]]
    for keys in keys_list:
        if "qwen" in data_path:
            end = 28
        else:
            end = 32
        for i in tqdm(range(0, end)):
            # print(keys,i)
            results = []
            roles = ['Harry Potter', 'Hermione Granger', 'Ronald Weasley', 'Aragorn', 'Frodo Baggins', 'Legolas', 'Samwise Gamgee', 'Gandalf', 'Jacob Black', 'Bella Swan', 'Edward Cullen', 'Gale Hawthorne', 'Katniss Everdeen', 'Peeta Mellark']
            for role in roles:
                # print(PROFILE.keys())
                result = train(i, role,data_path,keys)
                results.append(result)
                # print()
                logger.info(f"role:{role}, layer:{i}, result:{result}")
            df = pd.DataFrame(results)
            average_train_acc = df['train_acc'].mean()
            average_validation_acc = df['validation_acc'].mean()

            # 将平均值添加到DataFrame
            average_row = {
                'role': 'Average',
                'layer': i,
                'train_acc': average_train_acc,
                'validation_acc': average_validation_acc
            }
            # average_test_acc = {}
            for key in all_keys:
                dict_name = f"{key}_acc"
                # average_test_acc[dict_name] = df[dict_name].mean()
                average_row[dict_name] = df[dict_name].mean()
            df = df.append(average_row, ignore_index=True)

            role_save_path = f"{data_path}/classify/{str(keys)}"
            if not os.path.exists(role_save_path):
                os.makedirs(role_save_path)
            df.to_excel(f"{role_save_path}/{i}_{seed}.xlsx", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run script with parameters.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed.')
    
    args = parser.parse_args()
    
    run(data_path=args.data_path, seed=args.seed)


