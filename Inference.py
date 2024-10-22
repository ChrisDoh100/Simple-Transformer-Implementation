#going to get our input sequence
#tokenize it
#numericalize it
#then run it through the transformer until we get the endtoken
#then exit and check sentence.
#generate inference for a lot of sentences and check the bleu score.
#could have two modes, singular sentence translation and evaluation bleu score calculation.

import torch
from Preprocessing import get_masks_and_count_tokens_trg,get_source_mask
from Config import config
import sentencepiece as spm
from Transformerimp import Transformer

sp = spm.SentencePieceProcessor()
sp.Load('training.model')



inference_model = Transformer(config=config,vocab_size=sp.vocab_size())
try:
    inference_model.load_state_dict(torch.load('model.pth'))
    print("Model Found and Loaded!")
except FileNotFoundError:
    print("No Model Found!")
#input1 = input("enter")
src_sentences = [input("enter")]
inference_model.to('cuda')


@torch.no_grad
def infer(model, src_sentence_batch, pad_token_id=sp.pad_id(), max_len=50, start_token_id=sp.bos_id(), eos_token_id=sp.eos_id()):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    src_sentence_batch = src_sentence_batch.to(device)
    batch_size = src_sentence_batch.size(0)

    # Initialize the target sentence with the start token
    trg_sentence_batch = torch.full((batch_size, 1), start_token_id, device=device)  # Shape: [batch_size, 1]
    is_decoded = [False] * src_sentence_batch.shape[0]
    src_mask = get_source_mask(src_sentence_batch, pad_token_id).to(device)
    src_sentence_batch = model.encode(src_sentence_batch,src_mask)
    while True:
        # Generate source mask and target mask dynamically
        #src_mask = get_source_mask(src_sentence_batch, pad_token_id).to(device)
        trg_padding_mask = get_masks_and_count_tokens_trg(trg_sentence_batch, pad_token_id).to(device)
        length = len(trg_sentence_batch[0])
        with torch.no_grad():
            # Run the model: feed src_sentence and current target sequence (trg_sentence)
            output1 = model.decode(trg_sentence_batch,src_sentence_batch,trg_padding_mask,src_mask)
        # Get the next token (greedy decoding)
        next_token = torch.argmax(output1[length-1::length], dim=-1).unsqueeze(1)  # Shape: [batch_size, 1]
        for idx in range(len(trg_sentence_batch)):
            if trg_sentence_batch[idx][-1]==eos_token_id:
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
#predicted_sequence = infer(inference_model, src_sentence, pad_token_id=src_vocab[config['pad_token']], start_token_id=src_vocab[config['sos_token']], eos_token_id=src_vocab[config['eos_token']])
#print(trg_vocab.lookup_tokens(predicted_sequence[0].tolist()))
#detokenize the sentence etc
src_ids_batch = [sp.EncodeAsIds(src) for src in src_sentences]
max_length = max(len(ids) for ids in src_ids_batch)
src_ids_batch_padded = [ids + [sp.pad_id()] * (max_length - len(ids)) for ids in src_ids_batch]
src_tensor = torch.tensor(src_ids_batch_padded)
print("Input: ",sp.DecodeIds(src_tensor.tolist()))
output = infer(inference_model,src_tensor,sp.pad_id(),max_len=20,start_token_id=sp.bos_id())
print(output)


