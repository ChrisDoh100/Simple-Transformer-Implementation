import argparse
import torch
import os
from Transformer import Transformer
import sentencepiece as spm
from Inference import greedy_decoding,beam_search
def translate(model,sentence_batch, decoding_method):
    """Translates a single sentence."""
    if decoding_method=='greedy':
        print(greedy_decoding(model,sentence_batch,max_len=len(sentence_batch[0]),config=translate_config)[0])
    else:
        print(beam_search(model,sentence_batch,translate_config))


if __name__=="__main__":
    sp = spm.SentencePieceProcessor()
    sp.Load('training.model')
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoderlayers", type=int, default=6)
    parser.add_argument("--decoderlayers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--model_dimension", type=int, default=512)
    parser.add_argument("--vocab_size",type=int, default=32000)
    parser.add_argument("--source_ext",type=str, default='en')
    parser.add_argument("--target_ext",type=str, default='de')
    parser.add_argument("--device",type=str,default='cuda',choices=['cpu','cuda'])
    parser.add_argument("--decoding_type",type=str,default='beam',choices=['greedy','beam'])
    parser.add_argument("--beam_width",type=int,default=1)

    args = parser.parse_args()
    translate_config={}
    for arg in vars(args):
        translate_config[arg] = getattr(args,arg)
    
    assert os.path.exists(f"model-{translate_config['source_ext']}-{translate_config['target_ext']}.pth"),'No Model file Found!, Please run training to Generate a Model File!'
    Translation_model = Transformer(translate_config).to(translate_config['device'])
    Translation_model.load_state_dict(torch.load(f"model-{translate_config['source_ext']}-{translate_config['target_ext']}.pth"))
    sentence = torch.tensor([sp.EncodeAsIds(input("Please Enter your sentence: "))]).to(translate_config['device'])
    translate(Translation_model,sentence,translate_config['decoding_type'])