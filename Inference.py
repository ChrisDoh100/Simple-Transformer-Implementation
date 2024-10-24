#going to get our input sequence
#tokenize it
#numericalize it
#then run it through the transformer until we get the endtoken
#then exit and check sentence.
#generate inference for a lot of sentences and check the bleu score.
#could have two modes, singular sentence translation and evaluation bleu score calculation.

import torch
from Preprocessing import get_trg_mask,get_source_mask
from Config import config
import sentencepiece as spm

sp=spm.SentencePieceProcessor()


@torch.no_grad
def greedy_decoding(model=None, src_sentence_batch=None, max_len=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    sp.Load('training.model')
    src_sentence_batch = src_sentence_batch.to(device)
    batch_size = src_sentence_batch.size(0)

    # Initialize the target sentence with the start token
    trg_sentence_batch = torch.full((batch_size, 1), sp.bos_id(), device=device)  # Shape: [batch_size, 1]
    is_decoded = [False] * src_sentence_batch.shape[0]
    src_mask = get_source_mask(src_sentence_batch, sp.pad_id()).to(device)
    src_sentence_batch = model.encode(src_sentence_batch,src_mask)
    while True:
        # Generate source mask and target mask dynamically
        #src_mask = get_source_mask(src_sentence_batch, pad_token_id).to(device)
        trg_padding_mask = get_trg_mask(trg_sentence_batch, sp.pad_id()).to(device)
        length = len(trg_sentence_batch[0])
        with torch.no_grad():
            # Run the model: feed src_sentence and current target sequence (trg_sentence)
            output1 = model.decode(trg_sentence_batch,src_sentence_batch,trg_padding_mask,src_mask)
        # Get the next token (greedy decoding)
        next_token = torch.argmax(output1[length-1::length], dim=-1).unsqueeze(1)  # Shape: [batch_size, 1]
        for idx in range(len(trg_sentence_batch)):
            if trg_sentence_batch[idx][-1]==sp.eos_id():
                next_token[idx] = sp.eos_id()
                is_decoded[idx] = True
        trg_sentence_batch = torch.cat((trg_sentence_batch,next_token),dim=-1)
        if all(is_decoded) or len(trg_sentence_batch[0])>max_len:
            break

    answers = []
    for x in trg_sentence_batch:
        val = sp.DecodeIds(x.tolist())
        answers.append(val)
    return answers


if __name__=="__main__":
    greedy_decoding()


