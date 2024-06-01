#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# In[2]:


# load data for summarization
def load_and_sample(path, frac=0.1):
    df = pd.read_csv(path)
    return df.sample(frac=frac, random_state=42)

def preprocess_data(df):
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df = df.dropna(subset=['article', 'highlights'])
    df['article'] = df['article'].str.lower().str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    return df

sum_train_df = load_and_sample("train.csv")
sum_validation_df = load_and_sample("valid.csv")


# In[3]:


# load data for question answering
def load_json(path, frac=0.1):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def preprocess_qa(data):
    questions = []
    answers = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if not qa['is_impossible']:
                    questions.append(qa['question'])
                    answers.append(qa['answers'][0]['text'])
    return questions, answers

qa_train_questions, qa_train_answers = preprocess_qa(load_json("train-v2.0.json"))
qa_validation_questions, qa_validation_answers = preprocess_qa(load_json("dev-v2.0.json"))



# In[4]:


# load data for machine translation
def load_text_file(path, frac=0.1):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

mt_train_de = load_text_file("europarl-v7.de-en.de")
mt_train_en = load_text_file("europarl-v7.de-en.en")


# In[5]:


# initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
special_tokens = ['[SUMMARIZE]', '[QUESTION]', '[ANSWER]', '[TRANSLATE EN DE]']
tokenizer.add_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token


# In[6]:


import torch.nn as nn

# create soft prompt embedding layer
class SoftPromptEmbedding(nn.Module):
    def __init__(self, num_prompts, embedding_size):
        super(SoftPromptEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_prompts, embedding_size)

    def forward(self, input_ids):
        return self.embedding(input_ids)



# In[7]:


# define the model with soft prompt embeddings
class GPT2WithSoftPrompt(nn.Module):
    def __init__(self, model, soft_prompt_embedding):
        super(GPT2WithSoftPrompt, self).__init__()
        self.model = model
        self.soft_prompt_embedding = soft_prompt_embedding


# In[8]:


from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, texts, task_type, tokenizer, max_length=512):
        self.texts = texts
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_prefix = {
            'summarization': '[SUMMARIZE] ',
            'question_answering': '',
            'translation': ''
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.task_type == 'question_answering':
            question, answer = text
            input_text = f"[QUESTION] {question} [ANSWER] {answer}"
        elif self.task_type == 'translation':
            
            src, tgt = text
            input_text = src  
            output_text = tgt
        else:
            input_text = self.task_prefix[self.task_type] + text

        tokenized_input = self.tokenizer(input_text, max_length=self.max_length, truncation=True, return_tensors='pt').input_ids.squeeze(0)
        if self.task_type == 'translation':
            tokenized_output = self.tokenizer(output_text, max_length=self.max_length, truncation=True, return_tensors='pt').input_ids.squeeze(0)
        else:
            tokenized_output = tokenized_input.clone()

        return {'input_ids': tokenized_input, 'labels': tokenized_output}



# In[9]:


# training function
def train(model, train_loader, optimizer, device, epochs=1, clip_grad_norm=1.0):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1} Training Loss: {epoch_loss / len(train_loader)}')



# In[10]:


# evaluation function
def evaluate(model, dataloader, tokenizer, device, task, scorer=None):
    model.eval()
    predictions = []
    references = []
    if task == 'summarization':
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    elif task == 'question_answering':
        def normalize_answer(s):
            s = s.strip()
            s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", ' ', s)
            s = re.sub(r'["\']', '', s)
            s = re.sub(r'[ ]+', ' ', s)
            return s



# In[11]:

#evaluation functions
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

import sacrebleu

def evaluate_model(dataloader, model, tokenizer, device, task):
    predictions = []
    references = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=5, early_stopping=True)
            for output, label in zip(outputs, batch['labels']):
                decoded_pred = tokenizer.decode(output, skip_special_tokens=True)
                decoded_label = tokenizer.decode(label, skip_special_tokens=True)
                predictions.append(decoded_pred)
                references.append(decoded_label)

    if task == 'question_answering':
        f1_scores = []
        exact_matches = []
        for pred, label in zip(predictions, references):
            f1_scores.append(f1_score(pred, label))
            exact_matches.append(int(pred == label))
        avg_f1 = sum(f1_scores) / len(f1_scores)
        exact_match_percentage = sum(exact_matches) / len(exact_matches) * 100
        return {"average_f1": avg_f1, "exact_match": exact_match_percentage}

    elif task == 'summarization':
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        results = {key: scorer.score([ref], [pred]) for key, pred, ref in zip(['rouge1', 'rouge2', 'rougeL'], predictions, references)}
        return results

    elif task == 'translation':
        bleu_score = sacrebleu.corpus_bleu(predictions, [references])  
        return {"bleu": bleu_score.score}



# In[12]:


from multiprocessing import Pool
import gc

def tokenize_text(data):
    text, task_prefix, max_length = data
    tokenized_text = tokenizer.encode(task_prefix + text, add_special_tokens=True, truncation=True, max_length=max_length, return_tensors='pt')
    return tokenized_text

def tokenize_and_prepare_batches(texts, task_prefix, max_length=512):
    
    with Pool(processes=4) as pool:  
        data = [(text, task_prefix, max_length) for text in texts]
        tokenized_batches = pool.map(tokenize_text, data)
    gc.collect()
    return tokenized_batches



# In[ ]:


# Summarization task
sum_train_texts = [article for article in sum_train_df['article']]
sum_train_batches = tokenize_and_prepare_batches(sum_train_texts, '[SUMMARIZE] ')

sum_validation_texts = [article for article in sum_validation_df['article']]
sum_validation_batches = tokenize_and_prepare_batches(sum_validation_texts, '[SUMMARIZE] ')

# Question Answering task
qa_train_texts = ['[QUESTION] ' + q + ' [ANSWER] ' + a for q, a in zip(qa_train_questions, qa_train_answers)]
qa_train_batches = tokenize_and_prepare_batches(qa_train_texts, '')

qa_validation_texts = ['[QUESTION] ' + q + ' [ANSWER] ' + a for q, a in zip(qa_validation_questions, qa_validation_answers)]
qa_validation_batches = tokenize_and_prepare_batches(qa_validation_texts, '')

# Translation task
mt_train_texts = ['[TRANSLATE EN DE] ' + text for text in mt_train_en]
mt_train_batches = tokenize_and_prepare_batches(mt_train_texts, '')

mt_validation_texts = ['[TRANSLATE EN DE] ' + text for text in mt_train_en] 
mt_validation_batches = tokenize_and_prepare_batches(mt_validation_texts, '')


# In[ ]:


# Define the hyperparameters
batch_size = 4
epochs = 10
learning_rate = 1e-5
clip_grad_norm = 1.0


# In[ ]:


#preparing dataset as expected by the model
sum_train_dataset = TextDataset(sum_train_batches, 'summarization', tokenizer)
sum_validation_dataset = TextDataset(sum_validation_batches, 'summarization', tokenizer)
qa_train_dataset = TextDataset(qa_train_batches, 'question_answering', tokenizer)
qa_validation_dataset = TextDataset(qa_validation_batches, 'question_answering', tokenizer)

# DataLoaders
sum_train_loader = DataLoader(sum_train_dataset, batch_size=16, shuffle=True)
sum_validation_loader = DataLoader(sum_validation_dataset, batch_size=16, shuffle=False)
qa_train_loader = DataLoader(qa_train_dataset, batch_size=16, shuffle=True)
qa_validation_loader = DataLoader(qa_validation_dataset, batch_size=16, shuffle=False)




# In[ ]:


# initializing the soft prompt embedding layer
num_prompts = 4
embedding_size = 768
soft_prompt_embedding = SoftPromptEmbedding(num_prompts, embedding_size)

# model with soft prompt embeddings
model_with_soft_prompt = GPT2WithSoftPrompt(model, soft_prompt_embedding)

# push model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_with_soft_prompt.to(device)

# initialize optimizer
optimizer = torch.optim.Adam(model_with_soft_prompt.parameters(), lr=learning_rate)

# train the model 
for task in ['summarization', 'question_answering', 'translation']:
    print(f'Training on {task} task...')
    train(model_with_soft_prompt, train_loaders[task], optimizer, device, epochs=epochs, clip_grad_norm=clip_grad_norm)
    print(f'Evaluating on {task} task...')
    evaluation_results = evaluate(model_with_soft_prompt, validation_loaders[task], tokenizer, device, task)
    print(evaluation_results)

# Inference function 
def summarize(model, tokenizer, text, max_length=512, num_beams=5, early_stopping=True):
    model.eval()
    input_ids = tokenizer(text, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
    input_ids = input_ids.unsqueeze(0)
    attention_mask = tokenizer(text, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')['attention_mask']
    attention_mask = attention_mask.unsqueeze(0)
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

def answer_question(model, tokenizer, question, context, max_length=512, num_beams=5, early_stopping=True):
    model.eval()
    prompt = "[QUESTION] " + question + " [ANSWER] "
    input_text = prompt + context
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length').to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).int()
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def translate_text(model, tokenizer, text, src_lang="en", tgt_lang="de", max_length=512, num_beams=5, early_stopping=True):
    model.eval()
    prompt = f"[TRANSLATE {src_lang.upper()} {tgt_lang.upper()}]"
    input_text = prompt + " " + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length').to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).int()
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text



# In[ ]:


print(type(self.tokenized_batches[idx]))





