import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch
from transformers import DistilBertTokenizer, DistilBertModel

df = pd.read_csv('public_data/train/track_a/eng.csv')
test_df = pd.read_csv('public_data/dev/track_a/eng_a.csv')

import string

def remove_punctuation(text):
    return ''.join(char for char in text if char not in string.punctuation)

df['text'] = df['text'].apply(remove_punctuation)
test_df['text'] = test_df['text'].apply(remove_punctuation)

from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def lol(sentence):
    encoded = tokenizer(sentence, 
                    padding='max_length',
                    max_length=512,
                    truncation=True,
                    return_tensors='pt')
    return encoded.input_ids[0]

def lol2(sentence):
    encoded = tokenizer(sentence, 
                    padding='max_length',
                    max_length=512,
                    truncation=True,
                    return_tensors='pt')
    return encoded.attention_mask[0]
# lol(text)
df['input_ids'] = df['text'].apply(lol)
df['attention_mask'] = df['text'].apply(lol2)

test_df['input_ids'] = test_df['text'].apply(lol)
test_df['attention_mask'] = test_df['text'].apply(lol2)

class TweetsDataset(torch.utils.data.Dataset):
    def __init__(self, df, target_columns, feature_columns):
        self.df = df.copy()
        self.target_columns = target_columns
        self.feature_columns = feature_columns
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row[self.feature_columns].values.tolist(), torch.tensor(row[self.target_columns].values.tolist(), dtype=torch.float32)
    

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

emotions_list = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
input_list = ['input_ids', 'attention_mask']
dataset = TweetsDataset(df, emotions_list, input_list)

train_size = int(.7 * len(dataset))
validation_size = len(dataset) - train_size

# Create splits
train_dataset, validation_dataset = random_split(
    dataset, 
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=8,
)


from copy import deepcopy

# Create normalized coccurance matrix (3.1)
cooccurance_matrix = [[0] * 5 for i in range(5)]

for _, row in df.iterrows():
    for index, emotion in enumerate(emotions_list):
        if row[emotion] == 1:
            for index2, emotion2 in enumerate(emotions_list):
                cooccurance_matrix[index][index2] += row[emotion2]
                
normalized_m = deepcopy(cooccurance_matrix)
for i in range(len(normalized_m)):
    k = sum(df[emotions_list[i]])
    for j in range(len(normalized_m)):
        normalized_m[i][j] /= k

mu = 0.2
binarized_m = deepcopy(normalized_m)
for i in range(5):
    for j in range(5):
        if binarized_m[i][j] > mu:
            binarized_m[i][j] = 1
        else:
            binarized_m[i][j] = 0

w = .35
mitigate_oversmooth_m = deepcopy(binarized_m)
for i in range(5):
    row_sum = sum([i for i in mitigate_oversmooth_m[i]])
    for j in range(5):
        if i != j:
            mitigate_oversmooth_m[i][j] /= row_sum
        else:
            mitigate_oversmooth_m[i][j] -= w

adj = torch.tensor(mitigate_oversmooth_m)

# Calculate degree matrix
degrees = adj.sum(dim=1)

# Calculate D^(-1/2)
# Add small epsilon to prevent division by zero
d_inv_sqrt = torch.pow(degrees + 1e-7, -0.5)

# Convert to diagonal matrix
d_inv_sqrt = torch.diag(d_inv_sqrt)

# Normalized adjacency: D^(-1/2) A D^(-1/2)
q = d_inv_sqrt @ adj @ d_inv_sqrt


import numpy as np

def load_glove_vectors(glove_file):
    embeddings_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    return embeddings_dict

def encode_words(words, embeddings_dict, embedding_dim=300):
    # Initialize zero vector for unknown words
    unknown_vector = np.zeros(embedding_dim)
    
    # Encode each word
    encoded = []
    for word in words:
        word = word.lower()  # GloVe vocab is lowercase
        vector = embeddings_dict.get(word, unknown_vector)
        encoded.append(vector)
    
    return np.array(encoded)

glove_file = "glove.6B.300d.txt"
embeddings = load_glove_vectors(glove_file)

vectors = torch.tensor(encode_words(emotions_list, embeddings))

import pickle

all_results = []

from main import EmoGraph  # if EmoGraph is defined in main.py

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# # print(next(iter(test_df))[['input_ids', 'attention_mask']].values.tolist()[0].shape)
# for idx, row in test_df.iterrows():
#     print(row[['input_ids', 'attention_mask']].values.tolist()[0].unsqueeze(0).shape)
#     break

for idx, row in test_df.iterrows():
    print(idx)
    qj = row[['input_ids', 'attention_mask']].values.tolist()
    input_id = qj[0].unsqueeze(0)
    attention_mask = qj[1].unsqueeze(0)

    pred = loaded_model(input_id, attention_mask, q).tolist()[0]

    res = [1 if kk > .4 else 0 for kk in pred]
    lol = [row['id']]
    lol.extend(res)
    all_results.append(lol)
    # print(all_results)
    # break

output_df = pd.DataFrame(all_results, columns=['id','Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'])

output_df.to_csv('pred_eng_a.csv', index=False)

# import math
# import torch.nn as nn
# from transformers import BertModel

# class EmoGraph(torch.nn.Module):
#     def __init__(self, ee):
#         super().__init__()
#         self.w1 = torch.nn.Parameter(torch.randn(300, 768))
#         nn.init.xavier_uniform_(self.w1)
#         self.pp = [self.w1]
#         # self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
#         self.ee = ee
#     def forward(self, input_ids, attention_mask, matrix):
#         bert_output = self.bert(input_ids, attention_mask=attention_mask)
#         bert_embeddings = bert_output.last_hidden_state[:, 0, :]

#         # h = nn.functional.relu(matrix @ self.ee @ self.w1)
#         h = nn.functional.relu(matrix @ self.ee @ self.w1)
#         logits = bert_embeddings @ h.t()
#         out = nn.functional.sigmoid(logits)
#         return out
    

# model = EmoGraph(vectors)
# loss_fn = torch.nn.BCELoss()

# bert_optimizer = torch.optim.Adam(
#     model.bert.parameters(), 
#     lr=2e-5
# )
# gcn_optimizer = torch.optim.AdamW(
#     model.pp,
#     lr=0.001,
# )

# losses = []
# i_s = []
# for i in range(3):
#     model.train()
#     for idx, data in enumerate(train_loader):
#         inputs, labels = data
#         input_id, attention = inputs
#         bert_optimizer.zero_grad()
#         gcn_optimizer.zero_grad()
#         output = model(input_id, attention, q)
#         # print(output.shape)
#         # break
#         loss = loss_fn(output, labels)
#         loss.backward()

#         bert_optimizer.step()
#         gcn_optimizer.step()
#         print(loss.item(), i)

#     losses.append(loss.item())
#     i_s.append(i)

# import pickle
# with open('model.pkl', 'wb') as file:
#     pickle.dump(model, file)