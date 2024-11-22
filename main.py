import torch
import torch.nn as nn
from dataprocess import create_data_loader
import time
import logging
from tqdm import tqdm
from model import Model3
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 训练函数
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", ncols=100)

    loss_fn = nn.CrossEntropyLoss()  # 多分类损失函数

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # [batch_size], 直接分类标签

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, model_type="bert_bilstm")  # [batch_size, num_labels]

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        predictions = torch.argmax(logits, dim=1)  # [batch_size]
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_samples)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples

    logger.info(f"Train Loss (Epoch {epoch+1}): {avg_loss:.4f}")
    logger.info(f"Train Accuracy (Epoch {epoch+1}): {accuracy * 100:.2f}%")

def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(test_loader, desc="Testing", unit="batch", ncols=100)
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [batch_size]

            logits = model(input_ids, attention_mask, model_type="bert_bilstm")  # [batch_size, num_labels]

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)  # [batch_size]
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_samples)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model3(bert_model_name='bert-base-uncased', num_labels=4, model_type="bert_bilstm").to(device)

    train_loader, dev_loader, test_loader = create_data_loader()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    epochs = 3

    for epoch in range(epochs):
        start_time = time.time()

        train(model, train_loader, optimizer, epoch, device)
        test(model, test_loader, device)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Epoch {epoch+1} completed in {elapsed_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
