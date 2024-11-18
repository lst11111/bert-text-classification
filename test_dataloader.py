from dataprocess import create_data_loader
from torch.utils.data import DataLoader
import torch

train_data ,test_data = create_data_loader()
for i in train_data:
    print(i["input_ids"], i["attention_mask"], i["label"])
