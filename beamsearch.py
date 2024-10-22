import math
import torch
import sentencepiece as spm
from Config import config
from Transformerimp import Transformer
from Preprocessing import get_masks_and_count_tokens_trg,get_source_mask
sp = spm.SentencePieceProcessor()
sp.Load('training.model')

transformer = Transformer(config,sp.vocab_size()).to('cuda')
try:
    transformer.load_state_dict(torch.load('model.pth'))
    print("Load Sucessful!")
except FileNotFoundError:
    print("No Model Found, Will create one...")

def brevity_calculation(generated_sentence,reference_sentence_length=config['beam_depth']):
    """Calculates the brevity penalty for predictions which are shorter than expected."""

    if generated_sentence>=reference_sentence_length:
        return 1.0
    else:
        return math.exp(1-(reference_sentence_length/generated_sentence))

def beam_search(model,tokenized_sentence,beam_width):
    """Implementation of beam search that should improve over a simple greedy strategy."""

    model.to('cuda')
    actual_sentence = torch.Tensor(tokenized_sentence).to('cuda').unsqueeze(0)
    candidate_sentences=[(torch.tensor([[sp.bos_id()]]).to('cuda'),1.0)]
    for x in range(len(actual_sentence[0])):
        new_generations=[]
        #print("CANDIATES: ",len(candidate_sentences))
        for sentence in candidate_sentences:
            trg_sentence=sentence[0]
            if trg_sentence[-1].item()==sp.eos_id():
                new_generations.append(sentence)
                continue
            src_mask = get_source_mask(actual_sentence, sp.pad_id()).to('cuda')
            trg_padding_mask = get_masks_and_count_tokens_trg(trg_sentence, sp.pad_id()).to('cuda')
            length = len(trg_sentence)
            with torch.no_grad():
                # Run the model: feed src_sentence and current target sequence (trg_sentence)
                output1 = model(actual_sentence, trg_sentence, src_mask, trg_padding_mask)
            next_tokens = torch.topk(output1[length-1::length], dim=-1,k=beam_width)
            for x in range(beam_width):
                thing = torch.Tensor(next_tokens.indices[0][x]).unsqueeze(0).unsqueeze(0)
                blah=(torch.cat((trg_sentence,thing)),sentence[1]+next_tokens.values[0][x])
                new_generations.append(blah)
        new_generations = sorted(new_generations,key=lambda x:x[1],reverse=True)

        while len(new_generations)>beam_width:
            new_generations.pop()
        candidate_sentences=new_generations

    for x,sentence in enumerate(candidate_sentences):
        gen_length = len(candidate_sentences[x])
        ref_length = len(actual_sentence)
        penalty = 1.0
        candidate_sentences[x] = (candidate_sentences[x][0],penalty*candidate_sentences[x][1])
    sorted(candidate_sentences,key=lambda x:x[1],reverse=True)
    answer = [x.item() for x in candidate_sentences[0][0]]
    return sp.DecodeIds(answer)


src = "How are you?"
src_ids_batch = [sp.EncodeAsIds(src)]
