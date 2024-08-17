import datasets.load
import torch
import torch.nn as nn
import spacy
import datasets
import pathlib as PATH
import os
import pandas as pd


#caching function at the end
entokener = spacy.load("en_core_web_sm")
frtokener = spacy.load("fr_core_news_sm")


data = datasets.load_dataset("wmt/wmt14","fr-en")
train,test,valid = (data['train'][:2000000]['translation'],data['test'][:]['translation'],data['test'][:]['translation'])

def tokener(sentence):
    en_tokens,fr_tokens= [word.text for word in entokener.tokenizer(sentence['en'].replace('\xa0',' '))],[word.text for word in entokener.tokenizer(sentence['fr'].replace('\xa0',' '))]
    return {"en":sentence['en'],"fr":sentence['fr'],"en_tokens":en_tokens,"fr_tokens":fr_tokens}

test = list(map(tokener,test))
print(test)

