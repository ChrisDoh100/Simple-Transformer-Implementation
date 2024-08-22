import spacy


entokener = spacy.load("en_core_web_sm")
frtokener = spacy.load("fr_core_news_sm")
eos_token='<eos>'
sos_token='<sos>'
pad_token='<pad>'
unk_token='<unk>'
max_length=10
model_dimension=512
heads=8
encoderlayers=1
decoderlayers=1
epochs=100
special_tokens1 = [
    eos_token,
    sos_token,
    pad_token,
    unk_token,
]
fn_kwargs = {
    "en_nlp": entokener,
    "fr_nlp": frtokener,
    "max_length": max_length,
    "lower": True,
    "sos_token": sos_token,
    "eos_token": eos_token,
}