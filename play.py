import spacy
from functools import partial
import torchtext.vocab as vocab
from Transformer import Transformer
import torch.nn.functional as F
import torch
from datasets import load_dataset_builder

from Dataloading import get_masks



max_length=100
lower = True
sos_token = "<sos>"
eos_token = "<eos>"
pad_token = "<pad>"

data =[line.strip('\n') for line in open('english.txt','r',encoding='utf-8')]
datafr = [line.strip('\n') for line in open('french.txt','r',encoding='utf-8')]
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

def tokenizer(en_nlp = entokener,fr_nlp=frtokener,max_length=max_length,lower=lower,sos_token=sos_token,eos_token=eos_token,pad_token=pad_token,english = True):
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

def numericalise_tokens_wrapper(en_vocab = None,fr_vocab=None,english=True,max_len = max_length):
    def numericalize_tokens(example):
        while len(example)<max_len:
            example.append(pad_token)

        if english:
            ids = en_vocab.lookup_indices(example)
        else:
            ids = fr_vocab.lookup_indices(example)
        return ids
    return numericalize_tokens



my_eng_func = tokenizer(english=True)
my_fr_func = tokenizer(english=False)
tokenized_eng_data = list(map(my_eng_func,data[:1000]))
tokenized_fr_data = list(map(my_fr_func,datafr[:1000]))
min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"
special_tokens = special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]
en_vocab = vocab.build_vocab_from_iterator(
    tokenized_eng_data,
    min_freq=min_freq,
    specials=special_tokens,
)
fr_vocab = vocab.build_vocab_from_iterator(
    tokenized_fr_data,
    min_freq=min_freq,
    specials=special_tokens
)
en_vocab.set_default_index(en_vocab[unk_token])
fr_vocab.set_default_index(fr_vocab[unk_token])
numberspls = numericalise_tokens_wrapper(en_vocab=en_vocab,english=True)
numberspls2 = numericalise_tokens_wrapper(fr_vocab=fr_vocab,english=False)
first=tokenized_eng_data[0]
second=tokenized_fr_data[0]
eng_mask,fr_mask,freng_mask = get_masks([tokenized_eng_data[0]],[tokenized_fr_data[0]],max_length)
ans = list(map(numberspls,tokenized_eng_data))
ans1 = list(map(numberspls2,tokenized_fr_data))
cur_batch = ans[0]
cur_target = ans1[0]
cur_batch = torch.tensor(cur_batch)
cur_trg = torch.tensor(cur_target).to('cuda')
cur_batch = cur_batch[None,:]
cur_trg = cur_trg[None,:]
dummy = Transformer(len(en_vocab),len(fr_vocab),512,1,1,8,100,eng_padding_msk=eng_mask.to('cuda'),fr_attention_msk=fr_mask.to('cuda'),freng_mask=freng_mask.to('cuda')).to('cuda')
for p in dummy.parameters():
    if p.dim()>1:
        torch.nn.init.xavier_normal_(p)
dummy = dummy.forward(cur_batch.to('cuda'),cur_trg.to('cuda'))
##do cross_entorpy on the dummy output vs the actual output
dummt  = dummy.view(-1,len(fr_vocab))
print("FORWARD: ",dummy.shape, cur_trg.shape)
val = torch.nn.CrossEntropyLoss(ignore_index=fr_vocab[pad_token],
                                reduction='none')
cur_target = cur_trg.view(-1)
print(dummy.shape,cur_trg.shape)
cur_target = cur_trg.view(-1)
dummy = dummy.view(dummy.shape[0]*dummy.shape[1],-1)
print(cur_target.shape[0])
cur_target = cur_target.view(-1)
print(dummy.shape,cur_target.shape)
print("LOSS: ", val(dummy,cur_target))
loss= F.cross_entropy(dummy,cur_target,ignore_index=fr_vocab[pad_token])
print("REAL MANLY LOSS: ",loss)
loss.backward()
