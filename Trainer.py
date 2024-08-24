#import data
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from Dataset import TranslateDataset
from helpers import get_data,num_params
from Config import config
from Transformer import Transformer
from Optimizer import Lscheduler
from Preprocessing import gettokens,numericalise_tokens_wrapper,tokenizer,vocab_builder,get_masks,tensorising

torch.cuda.empty_cache()

sourcedata,targetdata = get_data('englishtrain.txt','frenchtrain.txt',valid=False)
validsource,validtarget = get_data('englishval.txt','frenchval.txt',valid=True)


try:
    src_vocab = torch.load('src_vocab.pth')
    trg_vocab = torch.load('trg_vocab.pth')
except FileNotFoundError:
    src_vocab = vocab_builder(gettokens(sourcedata),config,set_default=True)
    trg_vocab = vocab_builder(gettokens(targetdata,source=False),config,set_default=True)
    torch.save(src_vocab,'src_vocab.pth')
    torch.save(trg_vocab,'trg_vocab.pth')

transformer = Transformer(config).to('cuda')

#Get our tokenizers and our functions to convert our tokens to numbers.
src_tokenzier = tokenizer(source=True)
trg_tokenizer = tokenizer(source=False)
src_converter = numericalise_tokens_wrapper(src_vocab=src_vocab,source=True)
trg_converter = numericalise_tokens_wrapper(trg_vocab=trg_vocab,source=False)

#convert our source data to tensors, throw them into a dataset and dataloader.
sourcedata,targetdata = tensorising(sourcedata,targetdata,src_tokenzier,trg_tokenizer,src_converter,trg_converter)
validsource,validtarget = tensorising(validsource,validtarget,src_tokenzier,trg_tokenizer,src_converter,trg_converter)

#Getting datasets
translatedatasets= TranslateDataset(firstlang=sourcedata,secondlang=targetdata)
validationdataset = TranslateDataset(firstlang=validsource,secondlang=validtarget)

#Creating 
traindataloader = DataLoader(dataset=translatedatasets,batch_size=config['batch_size'],drop_last=True)
validationdataloader = DataLoader(dataset=validationdataset,batch_size=config['batch_size'],drop_last=True)

#setup our optimizer and loss function, and dont spend a few hours wondering why it wont train with a 1e-2 learning rate
#and instead keep it below/around 1e-4/5
adam = opt.Adam(transformer.parameters(),lr=1e-5)
scheduler = Lscheduler(adam,config['warmup_steps'],config['model_dimension'])
loss = torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=src_vocab[config['pad_token']])
print(num_params(transformer))
TOTAL_BATCHES=0
for epoch in range(config['epochs']):
    TOTAL_LOSS =0
    VALIDATION_LOSS=0
    BATCHES=0
    VALIDATION_BATCHES=0
    print(f'Processing epoch {epoch}....')
    for num_batch,batch in enumerate(traindataloader):
        transformer.train()
        BATCHES+=1
        TOTAL_BATCHES+=1
        #Generate the masks for that particular batch
        src_mask,trg_attn_msk,trg_pad_mask = get_masks(batch[0],batch[1])
        #unsqueezing so that the masks can be broadcast over all the heads.
        src_mask.unsqueeze_(1)
        trg_attn_msk.unsqueeze_(1)
        trg_pad_mask.unsqueeze_(1)
        results = transformer(batch[0].to('cuda'),batch[1].to('cuda'),src_mask.to('cuda'),trg_attn_msk.to('cuda'),trg_pad_mask.to('cuda'))
        results = results.view(-1,config['max_tokens'])
        batch[1] = batch[1].view(-1).to('cuda')
        cur_loss = loss(results.view(-1,len(trg_vocab)),batch[1])
        TOTAL_LOSS+=cur_loss.item()
        cur_loss.backward()
        scheduler.step()
        scheduler.zero_grad()
    with torch.no_grad():
        for val_batches,val_batch in enumerate(validationdataloader):
            transformer.eval()
            VALIDATION_BATCHES+=1
            val_src_mask,val_trg_attn_msk,val_trg_pad_msk = get_masks(val_batch[0].to('cuda'),val_batch[1].to('cuda'))
            val_src_mask.unsqueeze_(1)
            val_trg_attn_msk.unsqueeze_(1)
            val_trg_pad_msk.unsqueeze_(1)
            val_result = transformer(val_batch[0].to('cuda'),val_batch[1].to('cuda'),val_src_mask.to('cuda'),val_trg_attn_msk.to('cuda'),val_trg_pad_msk.to('cuda'))
            val_result = val_result.view(-1,config['max_tokens'])
            val_batch[1] = val_batch[1].view(-1).to('cuda')
            val_loss = loss(results,val_batch[1])
            VALIDATION_LOSS+=val_loss

    print(f'Finished {epoch} Epoch with: {TOTAL_LOSS/BATCHES} average loss with {TOTAL_BATCHES} and  {VALIDATION_LOSS/VALIDATION_BATCHES} validation loss.')


