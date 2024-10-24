import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from Dataloading import tokenBatchSampler
from helpers import num_params
from Config import config
#from Transformerimp import Transformer
from Optimizer import LRscheduler
from Transformerimp import Transformer
from Preprocessing import collate_fn
from basic import LabelSmoothingDistribution
from tqdm.auto import tqdm
import sentencepiece as spm
from play import GenerateDatasets
from evalutaion import calculate_bleu
from Inference import greedy_decoding

sp = spm.SentencePieceProcessor()
sp.Load('training.model')

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


def val_loop(dataloader=None,transformer=None,label_smoothing=None,loss=None):
    validation_loss=0
    validation_count=0
    sp.Load('training.Model')
    for val_batch in tqdm(dataloader, total=len(dataloader), desc='Processing Validation Loss...'):
        (val_src,val_src_mask),(val_trg_input,val_trg_mask),(val_trg_output) = val_batch
        results = transformer(val_src,val_trg_input,val_src_mask,val_trg_mask)
        val_trg_output = label_smoothing(val_trg_output)
        val_loss = loss(results,val_trg_output)
        validation_loss+=val_loss.item()
        validation_count+=1
    for val_batch in tqdm(dataloader, total=len(dataloader), desc='Processing Bleu Score...'):
        validation_predictions=[]
        validation_actuals=[]
        (val_src,val_src_mask),(val_trg_input,val_trg_mask),(val_trg_output) = val_batch
        sentences = greedy_decoding(transformer,val_src)
        for sentence in sentences:
             validation_predictions.append(sentence)
        for ground_sentence in val_trg_input:
             validation_actuals.append([sp.DecodeIds(ground_sentence.tolist())])
    bleu_score = calculate_bleu(validation_predictions,validation_actuals)
    print(f'Validation Loss of:{validation_loss/validation_count} with Bleu Score of {bleu_score} Bleu.')

traindataset,valdataset = GenerateDatasets(True,'iswlt','en','de')
tokensampler = tokenBatchSampler(dataset=traindataset,max_tokens=2500)
valtokensampler = tokenBatchSampler(dataset=valdataset,max_tokens=2500)
traindataloader = DataLoader(dataset=traindataset,collate_fn=collate_fn,batch_sampler=tokensampler)
valdataloader = DataLoader(dataset=valdataset,collate_fn=collate_fn,batch_sampler=valtokensampler)
transformer = Transformer(config=config,vocab_size=sp.vocab_size()).to('cuda')
label_smoothing = LabelSmoothingDistribution(smoothing_value=0.1,pad_token_id=sp.pad_id(),trg_vocab_size=sp.vocab_size(),device='cuda')
adam = opt.Adam(transformer.parameters(),betas=(0.9,0.98),eps=1e-9)
scheduler = LRscheduler(adam,config['warmup_steps'],config['model_dimension'])
loss = torch.nn.KLDivLoss(reduction='batchmean')
val_loop(dataloader=valdataloader,transformer=transformer,label_smoothing=label_smoothing,loss=loss)

    
