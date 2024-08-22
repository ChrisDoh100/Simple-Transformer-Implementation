import copy
import torch.nn as nn
import torchtext.vocab as vocab
import torchtext
from Constants import max_length
torchtext.disable_torchtext_deprecation_warning()


def clones(module,N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def get_data(basetxtfile,transtxtfile):
    firstland = [line.strip('\n')  for line in open(basetxtfile,encoding='utf-8')]
    secondland = [line.strip('\n') for line in open(transtxtfile,encoding='utf-8')]
    firstland = firstland[:1000]
    secondland = secondland[:1000]
    return firstland,secondland

def vocab_builder(data,min_freq,special_tokens,unk_token,set_default=False):
    vocab_builder = vocab.build_vocab_from_iterator(data,min_freq=min_freq,specials=special_tokens,max_tokens=8000)
    if set_default:
        vocab_builder.set_default_index(vocab_builder[unk_token])
    return vocab_builder

