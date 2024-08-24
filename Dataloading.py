
import torchtext
from Config import config

torchtext.disable_torchtext_deprecation_warning()


def tokenizer(english = True):
    """Wrapper that generates tokenized versions of a given text."""
    def tokenize_example(example):
            if english:
                tokens = [token.text for token in config['entokener'].tokenizer(example)][:config['max_length']]
            else:
                tokens = [token.text for token in config['frtokener'].tokenizer(example)][:config['max_length']]
            if config['lower_case']:
                tokens = [token.lower() for token in tokens]
            tokens = [config['sos_token']] + tokens + [config['eos_token']]
            return tokens
    return tokenize_example

def numericalise_tokens_wrapper(en_vocab = None,fr_vocab=None,english=True):
    """Converts the given tokenized versions of sentences into numberical values that can be passed to the transformer."""
    max_len = config['max_length']
    pad_token = config['pad_token']
    def numericalize_tokens(example):
        while len(example)<max_len:
            example.append(pad_token)

        if english:
            ids = en_vocab.lookup_indices(example)
        else:
            ids = fr_vocab.lookup_indices(example)
        return ids
    return numericalize_tokens


