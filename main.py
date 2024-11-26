import torch
import torch.nn as nn
from dataprocess import create_data_loader
import time
import logging
from tqdm import tqdm
from model import Model3
import  datetime

# 配置日志，输出到log.txt文件中
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("log//train_log.txt"), logging.StreamHandler()])
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

        optimizer.zero_grad()  # 清除模型参数的梯度
        logits = model(input_ids, attention_mask, model_type="bert_attention")  # [batch_size, num_labels]

        loss = loss_fn(logits, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据上面的梯度来用于更新模型参数的函数
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
    # 获取当前日期，格式为 YYYY-MM-DD
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    # 每个epoch结束时保存模型的权重
    torch.save(model.state_dict(), f"log//{current_date}model_epoch_{epoch+1}.pth")  # 保存模型权重


def dev(model, dev_loader, device):
    model.eval()  # 切换为评估模式，关闭梯度计算
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    pbar = tqdm(dev_loader, desc="Evaluating", unit="batch", ncols=100)  # 将desc改为Evaluating，表示验证过程
    loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    with torch.no_grad():  # 在评估阶段，不需要计算梯度
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [batch_size]

            logits = model(input_ids, attention_mask, model_type="bert_attention")  # [batch_size, num_labels]

            loss = loss_fn(logits, labels)  # 计算损失
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)  # 获取预测结果 [batch_size]
            correct_predictions += (predictions == labels).sum().item()  # 计算正确预测的数量
            total_samples += labels.size(0)  # 统计样本总数

            pbar.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_samples)  # 更新进度条

    avg_loss = total_loss / len(dev_loader)  # 计算平均损失
    accuracy = correct_predictions / total_samples  # 计算准确率

    logger.info(f"Validation Loss: {avg_loss:.4f}")  # 输出验证损失
    logger.info(f"Validation Accuracy: {accuracy * 100:.2f}%")  # 输出验证准确率


def test(model, test_loader, device):
    model.eval()  # 切换为评估模式，关闭梯度计算
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

            logits = model(input_ids, attention_mask, model_type="bert_attention")  # [batch_size, num_labels]

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

    model = Model3(bert_model_name='bert-base-uncased', num_labels=4, model_type="bert_attention").to(device)

    train_loader, dev_loader, test_loader = create_data_loader()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    epochs = 3

    for epoch in range(epochs):
        start_time = time.time()
        # 训练
        train(model, train_loader, optimizer, epoch, device)
        # 验证（调用dev）
        dev(model, dev_loader, device)
        # 测试
        test(model, test_loader, device)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Epoch {epoch+1} completed in {elapsed_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
