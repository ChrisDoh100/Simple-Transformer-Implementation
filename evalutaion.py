from torcheval.metrics.functional import bleu_score
from Config import config
from nltk.translate.bleu_score import corpus_bleu


def calculate_bleu(generated_sentences,actual_sentences):
    print(generated_sentences[-5:],actual_sentences[-5:])
    print("SIZES: ",len(generated_sentences),len(actual_sentences))
    score = bleu_score(generated_sentences,actual_sentences)
    return score