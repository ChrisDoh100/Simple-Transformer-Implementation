
import torchtext

torchtext.disable_torchtext_deprecation_warning()


def tokenizer(kwargs,english = True):
    """Wrapper that generates tokenized versions of a given text."""
    def tokenize_example(example):
            if english:
                tokens = [token.text for token in kwargs['entokener'].tokenizer(example)][:kwargs['max_length']]
            else:
                tokens = [token.text for token in kwargs['frtokener'].tokenizer(example)][:kwargs['max_length']]
            if kwargs['lower_case']:
                tokens = [token.lower() for token in tokens]
            tokens = [kwargs['sos_token']] + tokens + [kwargs['eos_token']]
            return tokens
    return tokenize_example

def numericalise_tokens_wrapper(kwargs,en_vocab = None,fr_vocab=None,english=True):
    """Converts the given tokenized versions of sentences into numberical values that can be passed to the transformer."""
    max_len = kwargs['max_length']
    pad_token = kwargs['pad_token']
    def numericalize_tokens(example):
        while len(example)<max_len:
            example.append(pad_token)

        if english:
            ids = en_vocab.lookup_indices(example)
        else:
            ids = fr_vocab.lookup_indices(example)
        return ids
    return numericalize_tokens


