import torch
from Config import config,special_tokens1
from torchtext import vocab



def gettokens(data_iter,source=True):
    """Returns tokenized versions of a single corpus."""
    if source:
        for sentence in data_iter:
            yield [token.text for token in config['entokener'].tokenizer(sentence)]
    else:
        for sentence in data_iter:
            yield [token.text for token in config['frtokener'].tokenizer(sentence)]

def filterlength(first_lang,second_land):
    """Returns a process tuple of both source and target sentences, where
        none are greater than the configured number of tokens."""
    return [filter(lambda x :(len(x[0])<config['max_length'] and len(x[1])<config['max_length']),zip(first_lang,second_land))]


def tokenizer(kwargs,source = True):
    """Wrapper that generates tokenized versions of a given sentence.
        Different from the getTokens function as that returns the full corpus tokenized."""
    def tokenize_example(example):
        if source:
            tokens = [token.text for token in kwargs['entokener'].tokenizer(example)][:kwargs['max_length']]
        else:
            tokens = [token.text for token in kwargs['frtokener'].tokenizer(example)][:kwargs['max_length']]
        if kwargs['lower_case']:
            tokens = [token.lower() for token in tokens]
        tokens = [kwargs['sos_token']] + tokens + [kwargs['eos_token']]
        return tokens
    return tokenize_example


def vocab_builder(data,kwargs,set_default=False):
    """Builds the vocab, based on the corpus of text used,
        with optional choice to set the default token."""
    built_vocab = vocab.build_vocab_from_iterator(data,min_freq=kwargs['min_word_freq'],specials=special_tokens1,max_tokens=kwargs['max_tokens'])
    if set_default:
        built_vocab.set_default_index(built_vocab[kwargs['unk_token']])
    return built_vocab


def numericalise_tokens_wrapper(kwargs,src_vocab = None,trg_vocab=None,source=True):
    """Converts the given tokenized versions of sentences into numberical values that can be passed to the transformer."""
    max_len = kwargs['max_length']
    pad_token = kwargs['pad_token']
    def numericalize_tokens(example):
        while len(example)<max_len:
            example.append(pad_token)

        if source:
            ids = src_vocab.lookup_indices(example)
        else:
            ids = trg_vocab.lookup_indices(example)
        return ids
    return numericalize_tokens


def get_masks(src_batch,trg_batch):
    """Generates the encoder padding mask, the decoder padding mask,
        and the decoder attention mask for any given batch."""
    max_seq_length = config['max_length']
    basic_attention_mask = torch.full([config['batch_size'],max_seq_length,max_seq_length],True)
    basic_attention_mask.triu_(diagonal=1)
    src_padding_mask = torch.full([config['batch_size'],max_seq_length,max_seq_length],False)
    trg_padding_mask = torch.full([config['batch_size'],max_seq_length,max_seq_length],False)
    trg_attention_mask = torch.full([config['batch_size'],max_seq_length,max_seq_length],False)

    for idx in range(config['batch_size']):
        src_len = 0
        trg_len = 0
        #2 is the padding index
        for i in range(len(src_batch[idx])):
            if src_batch[idx][i]!=2:
                src_len=i
        for i in range(len(trg_batch[idx])):
            if trg_batch[idx][i]!=2:
                trg_len=i
        src_padding_rows_columns = torch.arange(src_len+1,max_seq_length)
        trg_padding_rows_columns = torch.arange(trg_len+1,max_seq_length)

        #For padding tokens that are not included in the original sentence
        src_padding_mask[idx,src_padding_rows_columns,:] = True
        src_padding_mask[idx,:,src_padding_rows_columns]=True
        trg_padding_mask[idx,:,trg_padding_rows_columns]=True
        trg_padding_mask[idx,trg_padding_rows_columns,:]=True

        #Masking for the second MHA block in the decoder, between
        #engligh output of encoder and french input
        trg_attention_mask[idx,:,src_padding_rows_columns]=True
        trg_attention_mask[idx,trg_padding_rows_columns,:]=True

    src_padding_mask = torch.where(src_padding_mask,-1e9,0.)
    trg_masked_attention_padding_mask = torch.where(basic_attention_mask+trg_padding_mask,-1e9,0.)
    both_attention_mask = torch.where(trg_attention_mask,-1e9,0.)

    return src_padding_mask,trg_masked_attention_padding_mask,both_attention_mask

def tensorising(sourcelang,targetlang,src_tokenizer,trg_tokenizer,src_converter,trg_converter):
    sourcedata =list(map(src_tokenizer,sourcelang))
    targetdata= list(map(trg_tokenizer,targetlang))
    filtereddata= list(filter(lambda x : (len(x[0])<config['max_length']and len(x[1])<config['max_length']),zip(sourcedata,targetdata)))
    sourcedata = [src[0] for src in filtereddata]
    targetdata = [trg[1] for trg in filtereddata]
    sourcedata = list(map(src_converter,sourcedata))
    targetdata= list(map(trg_converter,targetdata))
    sourcedata = torch.tensor(sourcedata)
    targetdata = torch.tensor(targetdata)
    return targetdata,targetdata
