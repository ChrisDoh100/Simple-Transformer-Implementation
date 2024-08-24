#going to get our input sequence
#tokenize it
#numericalize it
#then run it through the transformer until we get the endtoken
#then exit and check sentence.
#generate inference for a lot of sentences and check the bleu score.
#could have two modes, singular sentence translation and evaluation bleu score calculation.
import torch
from Preprocessing import get_masks,tokenizer,numericalise_tokens_wrapper,tensorising
from Config  import config

src_vocab = torch.load('src_vocab.pth')
trg_vocab  = torch.load('trg_vocab.pth')
src_sentence = [' '.join(input("Please Enter your sentence: ").split())]
trg_sentence = [['<sos>']]

src_tokenzier = tokenizer(source=True)
src_converter = numericalise_tokens_wrapper(src_vocab=src_vocab,source=True)
trg_converter = numericalise_tokens_wrapper(trg_vocab=trg_vocab,source=False)

src_sentence = list(map(src_tokenzier,src_sentence))
src_sentence = list(map(src_converter,src_sentence))
trg_sentence = list(map(trg_converter,trg_sentence))
src_sentence,trg_sentence = torch.tensor(src_sentence),torch.tensor(trg_sentence)
print(src_sentence,trg_sentence)

inference_model = torch.load('model.pth')



inference_model.eval()

for x in range(config['max_length']):
    src_mask,trg_attn_msk,trg_pad_mask = get_masks(src_sentence[0],trg_sentence[0])
    src_mask.unsqueeze_(1)
    trg_attn_msk.unsqueeze_(1)
    trg_pad_mask.unsqueeze_(1)
    results = inference_model(src_sentence[0].to('cuda'),trg_sentence[1].to('cuda'),src_mask.to('cuda'),trg_attn_msk.to('cuda'),trg_pad_mask.to('cuda'))
    results = results.view(-1,config['max_tokens'])
    trg_sentence[0][x] = torch.argmax(results[x],dim=-1)
    if torch.argmax(results,dim=-1)==trg_vocab[config['eos_token']]:
        break
trg_sentence[0] = fr_vocab.lookup_tokens(trg_sentence[0])
print("Output Sentence: ",trg_sentence[0])
#detokenize the sentence etc



