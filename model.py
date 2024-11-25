import torch
import torch.nn as nn
from transformers import BertModel
###
class Model3(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=4, kernel_size=3, num_filters=256,model_type=None):
        super(Model3, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        if model_type=='bert_cnn':
            self.conv = nn.Conv2d(1, num_filters, kernel_size=(kernel_size,768), padding=0)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(num_filters, num_labels)
        elif model_type == 'bert_bilstm':
            self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=2, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(2 * 768, num_labels)  # 2 * hidden_size for bidirectional LSTM
        else:
            # 只使用 Transformer 编码器
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
            self.fc = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask,model_type=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        #cls_result= sequence_output[:,0,:].unsqueeze(1)
        #max_feature = torch.max(sequence_output,dim=1).values.unsqueeze(1)

        #sequence_output=torch.cat((cls_result,max_feature),dim=-1)
        #sequence_output = sequence_output.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        if model_type == "bert_cnn":
            sequence_output = sequence_output.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size]
            cnn_output = torch.relu(self.conv(sequence_output))  # [batch_size, num_filters, seq_len, 1]
            pooled_output = torch.max(cnn_output.squeeze(3), dim=2).values  # [batch_size, num_filters]
            features = self.dropout(pooled_output)
            logits = self.fc(features)  # [batch_size, num_labels]
            return logits
        elif model_type=="bert_bilstm":
            #sequence_output = sequence_output.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
            lstm_output = self.lstm(sequence_output)  # [batch_size, seq_len, 2 * lstm_hidden_size]
            pooled_output = torch.max(lstm_output[0], dim = 1).values# [batch_size, 2 * lstm_hidden_size]
            logits = self.fc(pooled_output).squeeze()  # [batch_size]
            return logits
        else:
            # Transformer 输入需要转置
            sequence_output = sequence_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]

            transformer_output = self.transformer_encoder(sequence_output)  # [seq_len, batch_size, hidden_size]
            pooled_output = torch.max(transformer_output, dim=0).values  # [batch_size, hidden_size]
            logits = self.fc(pooled_output)  # [batch_size, num_labels]
            return logits

