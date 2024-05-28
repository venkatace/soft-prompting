# Soft Prompting

The repository consists of a GPT-2 based model designed for several NLP tasks such as text summarization, question answering and machine translation in both English and German. 

Instructions on how to prepare and launch the models are provided below. Moreover, hyperparameters and ids to use are also described.

To meet the objectives, we use a GPT-2 model-controlled generation with additional task-oriented soft prompt embeddings. In practice, it works as follows:

•	Text Summarization: Article condensing to short summaries of their content.

•	Question Answering: Responding to the question on a specific context.

•	Machine Translation: Switching the English language text into German.

## Requirements

•	Python 3.8+

•	PyTorch 1.7+

•	Transformers 4.5+

•	pandas

•	tqdm

•	sacrebleu

Install the necessary packages using pip:

!pip install torch transformers pandas tqdm sacrebleu

## Setup
Clone the repository and install the required packages:

To run the training script:

Download the dataset files in local machine and set path to read dataset

Use the below command to run the training script

**python Softprompting GPT2-final.py **

To perform inference using the trained models, use the following commands in your script or interactive environment:
**from model_module import answer_question, summarize, translate_text**

# Summarization
**summary = summarize(model, tokenizer, "Your text here.")**

# Question Answering
**answer = answer_question(model, tokenizer, "Your question here.", "Relevant context here.")**

# Translation
**translation = translate_text(model, tokenizer, "Your English text here.")**

# Hyperparameters
## Main Hyperparameters:

•	Batch Size: 16

•	Epochs: 10

•	Learning Rate: 0.00001 (1e-5)

•	Clip Grad Norm: 1.0

## Additional Configuration and Model-Specific Parameters:

•	Max Length for Tokenization: 512

•	Number of Prompts in Soft Prompt Embedding: 4

•	Embedding Size for Prompts: 768

•	Special Tokens: ['[SUMMARIZE]', '[QUESTION]', '[ANSWER]', '[TRANSLATE EN DE]']

## Inference and Evaluation Parameters:

•	Number of Beams in Beam Search: 5

•	Early Stopping in Generation: True


# Data
This project utilizes several datasets tailored to specific NLP tasks such as text summarization, question answering, and machine translation. Below are the details and sources for each dataset:

## Text Summarization

•	Dataset: CNN/DailyMail Newspaper Text Summarization

•	Description: This dataset is used for training models on summarizing newspaper articles into concise reports.

•	Access: The dataset can be downloaded from Kaggle at CNN/DailyMail Newspaper Text Summarization Dataset.

## Question Answering

•	Dataset: Stanford Question Answering Dataset (SQuAD) 2.0

•	Description: SQuAD 2.0 combines 100,000 questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding reading passage.

•	Access: Details about the dataset and access instructions are available at SQuAD Homepage.

## Machine Translation

•	Dataset: Europarl v7 German-English

•	Description: This corpus is extracted from the proceedings of the European Parliament and is used for developing statistical machine translation systems.

•	Access: The dataset can be downloaded from the StatMT website at Europarl v7 German-English.

