import datasets.load
import torch
import torch.nn as nn
import spacy
import datasets
import pathlib as PATH
import os
import pandas as pd
import torchtext
import torchtext.vocab as vocab
from torch.utils.data import Dataset,dataloader

torchtext.disable_torchtext_deprecation_warning()

from Constants import max_length
lower = True
sos_token = "<sos>"
eos_token = "<eos>"


#caching function at the end
entokener = spacy.load("en_core_web_sm")
frtokener = spacy.load("fr_core_news_sm")
fn_kwargs = {
    "en_nlp": entokener,
    "fr_nlp": frtokener,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}
min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

def tokenizer(en_nlp = entokener,fr_nlp=frtokener,max_length=max_length,lower=lower,sos_token=sos_token,eos_token=eos_token,english = True):
    def tokenize_example(example):
            if english:
                tokens = [token.text for token in en_nlp.tokenizer(example)][:max_length]
            else:
                tokens = [token.text for token in fr_nlp.tokenizer(example)][:max_length]
            if lower:
                tokens = [token.lower() for token in tokens]
            tokens = [sos_token] + tokens + [eos_token]
            return tokens
    return tokenize_example

def numericalise_tokens_wrapper(en_vocab = None,fr_vocab=None,pad_token=pad_token,english=True,max_len = max_length):
    def numericalize_tokens(example):
        while len(example)<max_len:
            example.append(pad_token)

        if english:
            ids = en_vocab.lookup_indices(example)
        else:
            ids = fr_vocab.lookup_indices(example)
        return ids
    return numericalize_tokens

#test = test.map(tokenize_example,fn_kwargs=fn_kwargs)
#train = train.map(tokenize_example,fn_kwargs=fn_kwargs)
#valid = valid.map(tokenize_example,fn_kwargs=fn_kwargs)


def get_masks(eng_batch,fr_batch,max_seq_length=max_length):
    size = len(eng_batch)
    basic_attention_mask = torch.full([size,max_seq_length,max_seq_length],True)
    basic_attention_mask.triu_(diagonal=1)
    eng_padding_mask = torch.full([size,max_seq_length,max_seq_length],False)
    fr_padding_mask = torch.full([size,max_seq_length,max_seq_length],False)
    fr_attention_mask = torch.full([size,max_seq_length,max_seq_length],False)

    for idx in range(size):
        eng_len = 0
        fr_len = 0
        for i in range(len(eng_batch[idx])):
            if eng_batch[idx][i]!=2:
                eng_len=i
        for i in range(len(fr_batch[idx])):
            if fr_batch[idx][i]!=2:
                fr_len=i
        eng_padding_rows_columns = torch.arange(eng_len+1,max_seq_length)
        fr_padding_rows_columns = torch.arange(fr_len+1,max_seq_length)

        #For padding tokens that are not included in the original sentence
        eng_padding_mask[idx,eng_padding_rows_columns,:] = True
        eng_padding_mask[idx,:,eng_padding_rows_columns]=True
        fr_padding_mask[idx,:,fr_padding_rows_columns]=True
        fr_padding_mask[idx,fr_padding_rows_columns,:]=True
        
        #Now to add when we dont want to look forward when predicting outputs from our french sentence.
        #Because in the decoder, we have the "output" of the english sentence and the input from the french sentence
        #the dot product is going to be between both different types of lengths, hence why we mix the english masking
        # and the french masking
        fr_attention_mask[idx,:,eng_padding_rows_columns]=True
        fr_attention_mask[idx,fr_padding_rows_columns,:]=True
    
    eng_padding_mask = torch.where(eng_padding_mask,-1e9,0.)
    fr_masked_attention_padding_mask = torch.where(basic_attention_mask+fr_padding_mask,-1e9,0.)
    engfr_attention_mask = torch.where(fr_attention_mask,-1e9,0.)
    
    return eng_padding_mask,fr_masked_attention_padding_mask,engfr_attention_mask






