from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")
input_text = ["Hello, my dog is cute", "I love to play football"]
inputs = tokenizer(input_text , return_tensors="pt",padding=True)
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state