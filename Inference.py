#going to get our input sequence
#tokenize it
#numericalize it
#then run it through the transformer until we get the endtoken
#then exit and check sentence.
#generate inference for a lot of sentences and check the bleu score.
#could have two modes, singular sentence translation and evaluation bleu score calculation.

import torch
import math
from Preprocessing import get_trg_mask,get_source_mask
import sentencepiece as spm

sp=spm.SentencePieceProcessor()


@torch.no_grad
def greedy_decoding(model, src_sentence_batch, max_len,config):
    device = config['device']
    model.to(config['device'])
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

def brevity_calculation(generated_sentence,reference_sentence_length):
    """Calculates the brevity penalty for predictions which are shorter than expected."""

    if generated_sentence>=reference_sentence_length:
        return 1.0
    else:
        return math.exp(1-(reference_sentence_length/generated_sentence))


def beam_search(model,tokenized_sentence,config):
    """Implementation of beam search that should improve over a simple greedy strategy."""

    model.to(config['device'])
    sp.Load('training.model')
    actual_sentence = tokenized_sentence.to(config['device'])
    candidate_sentences=[(torch.tensor([[sp.bos_id()]]).to(config['device']),1.0)]
    print(candidate_sentences)
    for x in range(len(actual_sentence[0])):
        new_generations=[]
        #print("CANDIATES: ",len(candidate_sentences))
        for sentence in candidate_sentences:
            trg_sentence=sentence[0]
            if trg_sentence[-1].item()==sp.eos_id():
                new_generations.append(sentence)
                continue
            src_mask = get_source_mask(actual_sentence, sp.pad_id()).to(config['device'])
            trg_padding_mask = get_trg_mask(trg_sentence, sp.pad_id()).to(config['device'])
            length = len(trg_sentence)
            with torch.no_grad():
                # Run the model: feed src_sentence and current target sequence (trg_sentence)
                output1 = model(actual_sentence, trg_sentence, src_mask, trg_padding_mask)
            next_tokens = torch.topk(output1[length-1::length], dim=-1,k=config['beam_width'])
            print("Length: ",length)
            for x in range(config['beam_width']):
                thing = torch.Tensor(next_tokens.indices[0][x]).unsqueeze(0).unsqueeze(0)
                blah=(torch.cat((trg_sentence,thing)),sentence[1]+next_tokens.values[0][x])
                new_generations.append(blah)
        new_generations = sorted(new_generations,key=lambda x:x[1],reverse=True)
        print(new_generations)
        while len(new_generations)>config['beam_width']:
            new_generations.pop()
        candidate_sentences=new_generations
    for x,sentence in enumerate(candidate_sentences):
        gen_length = len(candidate_sentences[x])
        ref_length = len(actual_sentence)
        penalty = 1.0
        candidate_sentences[x] = (candidate_sentences[x][0],penalty*candidate_sentences[x][1])
    sorted(candidate_sentences,key=lambda x:x[1],reverse=True)
    answer = [x.item() for x in candidate_sentences[0][0]]
    print("Answer: ",answer)
    return sp.DecodeIds(answer)

