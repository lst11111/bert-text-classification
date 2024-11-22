import torch
from numpy import dtype
from pyarrow import output_stream
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, data_file, tokenizer_file, max_length=128, with_labels=True):
        self.texts, self.labels, _ = self.read_data(data_file)  # 一次性获取 texts 和 labels
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
        self.max_length = max_length
        self.with_labels = with_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

    def read_data(self, file):
        texts, labels, max_len = [], [], []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:  # 检查行是否为空或是否包含逗号
                    continue
                try:
                    text, label = line.rsplit(',', 1)  # 从右侧分割
                    texts.append(text.strip('"'))  # 去掉可能的引号
                    labels.append(int(label))  # 转为整数
                    max_len.append(len(text))
                except ValueError as e:
                    print(f"Error parsing line: {line}, Error: {e}")
        return texts, labels, max(max_len) if max_len else 0
    def my_collate_fn(self, batch):#自己的想法就是这个函数将数据进行分批处理，这一个batch就是一批，对他进行tokenizer的操作以及返回我们想要的
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        labels=torch.tensor(labels,dtype=torch.long)
        encodings =self.tokenizer(texts,padding=True,max_len=self.max_length,truncation=True,return_tensors='pt')

        outputs={
            'input_ids':encodings['input_ids'],
            'attention_mask':encodings['attention_mask'],
            'labels':labels
        }
        return outputs


# 测试
if __name__ == '__main__':
    train_dataset = TextDataset("data/ag_news/train.csv", "bert-base-uncased", with_labels=True)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False ,collate_fn=train_dataset.my_collate_fn)
    for i, batch in enumerate(train_dataloader):
        print(batch)
        break
