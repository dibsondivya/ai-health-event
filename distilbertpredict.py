# import requests

# API_URL = "https://api-inference.huggingface.co/models/dibsondivya/distilbert-phm-tweets"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()

# tweet = "Alzheimer's is a terrible, terrible way to live and a horrid way to die. It is the longest goodbye you will ever have in your life."
	
# output = query({
# 	"inputs": tweet,
# })

# print(output)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os
# import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import AutoModel, BertTokenizerFast

from bert import BERT_Arch

DATASET_PATH = './datasets'
MODEL_PATH = './models'

TRAIN_PATH = os.path.join(DATASET_PATH, 'train_textcleaned.csv')
TEST_PATH = os.path.join(DATASET_PATH, 'test_textcleaned.csv')

device = torch.device('cpu')

def predict(tweet):
    ## =====================================
    ## PREPARE THE DATA
    ## =====================================
    df = pd.read_csv(TRAIN_PATH)
    df = df.drop(['Unnamed: 0'], axis=1)

    # train_df = pd.read_csv(TRAIN_PATH)
    # train_df = train_df.drop(['Unnamed: 0'], axis=1)
    # train_df.head()

    # test_df = pd.read_csv(TEST_PATH)
    # test_df = test_df.drop(['Unnamed: 0'], axis=1)
    # print(test_df.head())

    # x_train = np.array(train_df.tweet)
    # y_train = np.array(train_df.label)

    # x_test = np.array(test_df.tweet)
    # y_test = np.array(test_df.label)

    # print(x_test)


    ## =====================================
    ## MODEL DEFINITION
    ## =====================================
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert)

    model = model.to(device)

    max_seq_len = 30

    train_text, temp_text, train_labels, temp_labels = train_test_split(df['tweet'], df['label'], 
                                                                        random_state=2018, 
                                                                        test_size=0.3, 
                                                                        stratify=df['label'])

    # we will use temp_text and temp_labels to create validation and test set
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                    random_state=2018, 
                                                                    test_size=0.5, 
                                                                    stratify=temp_labels)

    # Load the BERT tokenizer
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('dibsondivya/distilbert-phm-tweets')
    
    # # tokenize and encode sequences in the test set
    # tokens_test = tokenizer.batch_encode_plus(
    #     [tweet],
    #     max_length = max_seq_len,
    #     pad_to_max_length=True,
    #     truncation=True,
    #     return_token_type_ids=False
    # )

    # test_seq = torch.tensor(tokens_test['input_ids'])
    # test_mask = torch.tensor(tokens_test['attention_mask'])
    # test_y = torch.tensor(test_labels.tolist())


    ## =====================================
    ## LOAD THE WEIGHTS
    ## =====================================
    # path = './models/saved_weights-500_epochs.pt'
    # path = './models/model-75-torch.pkl'
    # model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


    model = AutoModelForSequenceClassification.from_pretrained('dibsondivya/distilbert-phm-tweets')

    # get predictions for test data
    # with torch.no_grad():
        # preds = model(test_seq.to(device), test_mask.to(device))
        # preds = preds.detach().cpu().numpy()
    
    # Tweet 
    tweet = ["""i had a heart attack today"""]

    # Tokenize inputs
    inputs = tokenizer(tweet, padding=True, truncation=True, return_tensors="pt").to(device) # Move the tensor to the GPU

    # Inference model and get logits
    outputs = model(**inputs)

    # get predictions for test data
    with torch.no_grad():
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = predictions.detach().cpu().numpy()

    print(preds)
    # model's performance
    preds = np.argmax(predictions)
    # print(classification_report(test_y, preds))

    return preds.item()

# pred = predict("Alzheimer's is the worst disease on the planet")
# print(pred)