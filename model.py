import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self,bert_model_name="bert-base-uncased",num_labels=4,kernel_size=3,num_filters=256):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.conv = nn.Conv1d(768, num_filters, kernel_size, padding=1)
        self.fc = nn.Linear(num_filters, num_labels)
    def forward(self, input_ids, attention_mask):
            bert_output = self.bert(input_ids, attention_mask=attention_mask)
            sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
            sequence_output = sequence_output.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]

            cnn_output = self.conv(sequence_output)  # [batch_size, num_filters, seq_len]
            cnn_output = cnn_output.permute(0, 2, 1)  # [batch_size, seq_len, num_filters]
            pooled_output = torch.max(cnn_output, dim=1).values  # [batch_size, num_filters]
            logits = self.fc(pooled_output).squeeze()  # [batch_size]
            return logits