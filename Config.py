import spacy
entokener=spacy.load("en_core_web_sm")
frtokener=spacy.load("fr_core_news_sm")

config={
    'eos_token':'<eos>',
    'sos_token':'<sos>',
    'pad_token':'<pad>',
    'unk_token':'<unk>',
    'entokener' : entokener,
    'frtokener' : entokener,
    'max_length':40,
    'model_dimension':512,
    'heads':8,
    'encoderlayers':6,
    'decoderlayers':6,
    'epochs':10,
    'min_word_freq' : 2,
    'lower_case':True,
    'batch_size':5,
    'max_tokens':16000,
    'sentence_amount':20000,
    'inference_mode':True,
    'warmup_steps':4000,
    'eval':True
    }

special_tokens1 = [
    config['eos_token'],
    config['sos_token'],
    config['pad_token'],
    config['unk_token'],
]
