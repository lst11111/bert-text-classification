#首先处理英文数据集，里面包含了4个label。
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer


# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }
def create_data_loader():
    dataset = load_dataset("csv", data_files={"train": "data/ag_news/train.csv", "test": "data/ag_news/test.csv"})
    train_texts, train_labels = dataset['train']['text'], dataset['train']['label']
    test_texts, test_labels = dataset['test']['text'], dataset['test']['label']
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader,test_loader