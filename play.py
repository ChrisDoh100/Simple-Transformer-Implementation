import torch
import torch.nn.functional as F
import math
import datasets
from pathlib import Path
import spacy
import pandas as pd


curpath = Path("/Data")

entokener = spacy.load("en_core_web_sm")
frtokener = spacy.load("fr_core_news_sm")

traintest = pd.read_csv('test.csv')

traintestsen = traintest
traintestsfr = traintest['fr'][0]

def tokener(sentence):
    en_tokens,fr_tokens= [word.text for word in entokener.tokenizer(sentence['en'][0])],[word.text for word in entokener.tokenizer(sentence['fr'][0])]
    return {"en_tokens":en_tokens,"fr_tokens":fr_tokens}


traintestsen = traintestsen.map(tokener)
