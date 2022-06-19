from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np
import torch

device = torch.device('cpu')

def predict(tweet):

    tokenizer = AutoTokenizer.from_pretrained('dibsondivya/distilbert-phm-tweets')
    
    model = AutoModelForSequenceClassification.from_pretrained('dibsondivya/distilbert-phm-tweets')

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

    # model's performance
    preds = np.argmax(predictions)
    # print(classification_report(test_y, preds))

    return preds.item()

# pred = predict("Alzheimer's is the worst disease on the planet")
# print(pred)

def convert_label(label):
    label = int(label)

    if label == 0:
        return 'NON-HEALTH'
    elif label == 1:
        return 'AWARENESS'
    elif label == 2:
        return 'OTHER MENTION'
    elif label == 3:
        return 'SELF-MENTION'