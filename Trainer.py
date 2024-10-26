import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from Dataloading import tokenBatchSampler
from helpers import num_params
#from Transformerimp import Transformer
from Optimizer import LRscheduler
from Transformerimp import Transformer
from Preprocessing import collate_fn
from LabelSmoothing import LabelSmoothingDistribution
from tqdm.auto import tqdm
import sentencepiece as spm
from play import GenerateDatasets
from Inference import greedy_decoding
from torcheval.metrics.functional import bleu_score


def train_loop(dataloader=None,scheduler1=None,transformer1=None,label_smoothing1=None,loss1=None):
        Training_loss=0
        Training_batches=0
        for batch in tqdm(dataloader, total= len(dataloader), desc='Processing training...'):
            (src,src_mask),(trg_input,trg_mask),(trg_output) = batch
            results = transformer1(src,
                                    trg_input,
                                    src_mask=src_mask,
                                    trg_mask=trg_mask) 
            trg_output = label_smoothing1(trg_output)
            cur_loss = loss1(results,trg_output)
            Training_loss+=cur_loss.item()
            Training_batches+=1
            cur_loss.backward()
            scheduler1.step()
            scheduler1.zero_grad()
        print(f'Training Loss of:{Training_loss/Training_batches}.')


def val_loop(sp,dataloader=None,transformer=None,label_smoothing=None,loss=None):
    validation_loss=0
    validation_count=0
    sp.Load('training.Model')
    validation_predictions=[]
    validation_actuals=[]
    with torch.no_grad():
        for val_batch in tqdm(dataloader, total=len(dataloader), desc='Processing Validation Loss...'):
            (val_src,val_src_mask),(val_trg_input,val_trg_mask),(val_trg_output) = val_batch
            results = transformer(val_src,val_trg_input,val_src_mask,val_trg_mask)
            val_trg_output = label_smoothing(val_trg_output)
            val_loss = loss(results,val_trg_output)
            validation_loss+=val_loss.item()
            validation_count+=1
        for val_batch in tqdm(dataloader, total=len(dataloader), desc='Processing Bleu Score...'):
            (val_src,val_src_mask),(val_trg_input,val_trg_mask),(val_trg_output) = val_batch
            for sentence in greedy_decoding(transformer,val_src):
                validation_predictions.append(sentence)
            for ground_sentence in val_trg_input:
                validation_actuals.append([sp.DecodeIds(ground_sentence.tolist())])
    b_score = bleu_score(validation_predictions,validation_actuals)
    
    print(f'Validation Loss of:{validation_loss/validation_count} with Bleu Score of {b_score} Bleu.')
    torch.cuda.empty_cache()


    
