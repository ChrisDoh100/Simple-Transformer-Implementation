import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from Dataset import TranslateDataset
from Dataloading import tokenBatchSampler
from helpers import num_params
from Config import config
from Transformerimp import Transformer
from Optimizer import Lscheduler
from Preprocessing import tensorising,collate_fn
from basic import LabelSmoothingDistribution
from Inference import infer
from tqdm.auto import tqdm
import sentencepiece as spm
from evalutaion import calculate_bleu
import pickle as pkl
import time as tm
import gc
from play import load_and_generate_data,Encoding_Data,generate_vocab


sp = spm.SentencePieceProcessor()
sp.Load('training.model')
data =load_and_generate_data('iswlt','en','de')
generate_vocab(data,'en','de')
source,target_inp,target_output = Encoding_Data(data,'train','en','de')
traindataset = TranslateDataset(source,target_inp,target_output)
tokensampler = tokenBatchSampler(dataset=traindataset,max_tokens=2500)
traindataloader = DataLoader(dataset=traindataset,collate_fn=collate_fn,batch_sampler=tokensampler)
transformer = Transformer(config=config,vocab_size=sp.vocab_size()).to('cuda')
label_smoothing = LabelSmoothingDistribution(smoothing_value=0.1,pad_token_id=sp.pad_id(),trg_vocab_size=sp.vocab_size(),device='cuda')
adam = opt.Adam(transformer.parameters(),betas=(0.9,0.98),eps=1e-9)
scheduler = Lscheduler(adam,config['warmup_steps'],config['model_dimension'])
loss = torch.nn.KLDivLoss(reduction='batchmean')
max_bleu=0.00
TOTAL_LOSS=0
for epoch in range(config['epochs']):
    TOTAL_BATCHES=0
    TOTAL_LOSS=0
    transformer.train()
    print(f'Processing epoch {epoch+1}....')
    for batch in tqdm(traindataloader, total= len(traindataloader), desc=f'Processing epoch {epoch+1}...'):
        TOTAL_BATCHES+=1
        (src,src_mask),(trg_input,trg_mask),(trg_output) = batch
        results = transformer(src,
                                trg_input,
                                src_mask=src_mask,
                                trg_mask=trg_mask)  # these are log probabilities
        trg_output = label_smoothing(trg_output)
        cur_loss = loss(results,trg_output)
        TOTAL_LOSS+=cur_loss.item()
        cur_loss.backward()
        scheduler.step()
        scheduler.zero_grad()
    print(f'Finished Epoch {epoch+1} with: {TOTAL_LOSS/TOTAL_BATCHES} average loss with {TOTAL_BATCHES}.')
print(f'Maximum Bleu Found:{max_bleu}')

