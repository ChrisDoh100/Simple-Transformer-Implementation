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
    'max_length':10,
    'model_dimension':512,
    'heads':8,
    'encoderlayers':1,
    'decoderlayers':1,
    'epochs':100,
    'min_word_freq' : 2,
    'lower_case':True,
    'batch_size':20,
    'max_tokens':1000,
    'sentence_amount':1000
}

special_tokens1 = [
    config['eos_token'],
    config['sos_token'],
    config['pad_token'],
    config['unk_token'],
]
