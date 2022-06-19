# Tweet Classification for Personal Health Mention
This application classifies user input tweets into their relevant health event groups. The [ERNIE 2.0 model](https://huggingface.co/dibsondivya/ernie-phmtweets-sutd) used for classification has been trained on data found from an [Emory University Study on Detection of Personal Health Mentions in Social Media paper](https://arxiv.org/pdf/1802.09130v2.pdf), and fine-tuned to perform with 88.5% accuracy.

Tweets were labelled with 4 main classes:
/n
0. non-health
1. awareness
2. other-mention
3. self-mention

// add gui images


## App Installation and Usage
```Python
# Install the required packages
pip install requirements.txt
# Run the application
python3 app.py
```

## Model Development
Model was trained on preprocessed tweet data, before finetuning and evaluation against validation and test sets. ERNIE 2.0 was found to be the best performing, reporting an accuracy of 88.5% on the test set. Note that a similarly high performing model based on [DistilBERT](https://huggingface.co/dibsondivya/distilbert-phmtweets-sutd) was also finetuned, performing with 87.7% accuracy. 

## Model Installation and Usage
For ERNIE 2.0 model:
```Python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("dibsondivya/ernie-phmtweets-sutd")
model = AutoModelForSequenceClassification.from_pretrained("dibsondivya/ernie-phmtweets-sutd")
```

For DistilBERT model:
```Python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("dibsondivya/distilbert-phmtweets-sutd")
model = AutoModelForSequenceClassification.from_pretrained("dibsondivya/distilbert-phmtweets-sutd")
```

## Dependencies
* Python 3.7 
* transformers
* datasets


## References for Models Attempted
For ERNIE 2.0 Model:
```bibtex
@article{sun2019ernie20,
  title={ERNIE 2.0: A Continual Pre-training Framework for Language Understanding},
  author={Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:1907.12412},
  year={2019} 
}
```

For DistilBERT model:
```bibtex
@article{Sanh2019DistilBERTAD,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.01108}
}
```
