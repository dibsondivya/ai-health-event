# Tweet Classification
This application classifies user input tweets into their relevant health event groups. The [ERNIE 2.0 model](https://huggingface.co/dibsondivya/ernie-phmtweets-sutd) used for classification has been trained on data found from an [Emory University Study on Detection of Personal Health Mentions in Social Media paper] (https://arxiv.org/pdf/1802.09130v2.pdf), and fine-tuned to perform with 88.5% accuracy.

// add gui images

## Dependencies
* Python 3.7 
* transformers
* datasets

## App Installation and Usage
```Python
# Install the required packages
pip install requirements.txt
# Run the application
python3 app.py
```

## Model Installation and Usage
For DistilBERT based model:
```Python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("dibsondivya/distilbert-phmtweets-sutd")
model = AutoModelForSequenceClassification.from_pretrained("dibsondivya/distilbert-phmtweets-sutd")
```
For ERNIE 2.0 based model:
```Python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("dibsondivya/ernie-phmtweets-sutd")
model = AutoModelForSequenceClassification.from_pretrained("dibsondivya/ernie-phmtweets-sutd")
```

## Work In Progress
* Incorporation of working model to the backend
* Color correcting and styling

## References for Models Attempted
```bibtex
@article{Sanh2019DistilBERTAD,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.01108}
}
```
```bibtex
@article{sun2019ernie20,
  title={ERNIE 2.0: A Continual Pre-training Framework for Language Understanding},
  author={Sun, Yu and Wang, Shuohuan and Li, Yukun and Feng, Shikun and Tian, Hao and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:1907.12412},
  year={2019} 
}
```
