import sentencepiece as spm


spm.SentencePieceTrainer.Train(
    input = 'germanenglishtrain.txt',
    model_prefix = 'training',
    model_type = 'bpe',
    vocab_size = 32000,
    pad_id=3,
    shuffle_input_sentence=True,
    bos_id=1,
    eos_id=2,
)



