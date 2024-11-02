import argparse
import torch
from Optimizer import LRscheduler
from Transformer import Transformer
from Sampler import tokenBatchSampler
from LabelSmoothing import LabelSmoothingDistribution
from torch.utils.data import DataLoader
import sentencepiece as spm
import torch.nn as nn
import torch.optim as opt
from Preprocessing import collate_fn
from DataGen import GenerateDatasets
from Trainer import train_loop,val_loop
from Helpers import num_params



def training(training_config):
    """Transformer training setup based on passed config."""
    #1 Get the data.
    if training_config['validation']:
        trainingset,validationset = GenerateDatasets(training_config['validation'],training_config['dataset'],training_config['source_ext'],training_config['target_ext'],training_config['vocab_size'])
    else:
        trainingset = GenerateDatasets(training_config['validation'],training_config['dataset'],training_config['source_ext'],training_config['target_ext'],training_config['vocab_size'])
    #2 Setup training requirements.
    sp = spm.SentencePieceProcessor()
    sp.Load('training.model')
    Language_model = Transformer(config=training_config).to(training_config['device'])
    adam_optimizer = opt.Adam(Language_model.parameters(),betas=(0.9,0.98),eps=1e-9)
    scheduler = LRscheduler(optimizer=adam_optimizer,warmups=training_config['warmup'],model_dimension=training_config['model_dimension'])
    if training_config['validation']:
        validationsampler = tokenBatchSampler(dataset=validationset,max_tokens=1000,shuffle=False)
        validationdataloader = DataLoader(dataset=validationset,collate_fn=collate_fn,batch_sampler=validationsampler)
    
    trainingsampler = tokenBatchSampler(dataset=trainingset,max_tokens=1500,shuffle=True)
    label_smoothing = LabelSmoothingDistribution(smoothing_value=0.1, pad_token_id=3,vocab_size=training_config['vocab_size'],device=training_config['device'])
    trainingdataloder = DataLoader(dataset=trainingset,collate_fn=collate_fn,batch_sampler=trainingsampler)
    KVloss = nn.KLDivLoss(reduction='batchmean')
    
    #3 Execute training.
    print("Num Params: ",num_params(Language_model))
    for epoch in range(training_config['epochs']):
        train_loop(epoch,trainingdataloder,scheduler,Language_model,label_smoothing,KVloss)
        if training_config['validation']:
            val_loop(sp,validationdataloader,Language_model,label_smoothing,KVloss,training_config)
        torch.cuda.empty_cache()

    #4 save resulting model
    torch.save(Language_model.state_dict(),f"model-{training_config['source_ext']}-{training_config['target_ext']}.pth")




if __name__=="__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--encoderlayers", type=int, default=6)
    parser.add_argument("--decoderlayers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--model_dimension", type=int, default=512)
    parser.add_argument("--feedforward_scale",type=int,default=4)
    parser.add_argument("--dropout_prob",type=float,default=0.1)
    parser.add_argument("--vocab_size",type=int, default=32000)
    parser.add_argument("--validation", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset", type=str, default='iwslt')
    parser.add_argument("--source_ext",type=str, default='en')
    parser.add_argument("--target_ext",type=str, default='fr')
    parser.add_argument("--device",type=str,default='cuda')

    args = parser.parse_args()

    trainingconfig = {}
    for arg in vars(args):
        trainingconfig[arg] = getattr(args,arg)
    print("Training Config:")
    for k,v in trainingconfig.items():
        print(k,v)
    training(training_config=trainingconfig)
