from model import Model3

import torch
from transformers import BertTokenizer
import time
def load_model(device, model_path):
    myModel = Model3(model_type = 'bert_bilstm').to(device)#不确定这里是否需要写入模型类型
    myModel.load_state_dict(torch.load(model_path, map_location=device))  # 加载到正确设备
    myModel.eval()
    return myModel


def process_text(text, bert_pred, device):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    mask = [1] * len(token_id) + [0] * (128 + 2 - len(token_id))
    token_ids = token_id + [0] * (128 + 2 - len(token_id))

    # 将张量移动到 device 上
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
    mask = torch.tensor(mask).unsqueeze(0).to(device)
    x = torch.stack([token_ids, mask]) #[2*1*130]
    return x


def text_class_name(pred,text):
    result = torch.argmax(pred, dim=1)
    result = result.cpu().numpy().tolist()
    classification = open('data/ag_news/class.txt', "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    print(f"文本：{text}\t预测的类别为：{classification_dict[result[0]]}")

if __name__ == "__main__":
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(device, 'log/best_bert_bilstm2024-11-28-14-02model.pth')

    texts = [
        ("Scientists discover new way to predict volcanic eruptions using AI models"),  # tech
        ("Olympic athlete breaks world record in 100m sprint"),  # sports
        ("Tech giants face pressure to address data privacy concerns in new legislation"),  # tech
        ("Stock market reacts to interest rate hike by Federal Reserve"),  # business
        ("Global leaders discuss climate change at UN summit"),  # world
        ("NBA Finals predictions: Can the Lakers defend their title?"),  # sports
        ("Oil prices surge as OPEC members reach new agreement on output cuts"),  # business
        ("AI-powered apps are revolutionizing healthcare in developing countries"),  # tech
        ("Humanitarian aid arrives in conflict zones amid worsening global crisis"),  # world
        ("Tech startups continue to innovate in AI and blockchain technologies")  # tech
    ]

    print("模型预测结果：")
    for text in texts:
        x = process_text(text, 'bert-base-uncased', device)
        with torch.no_grad():
            pred = model(x[0],x[1],model_type="bert_bilstm")
        text_class_name(pred,text )
    end = time.time()
    print(f"耗时为：{end - start} s")