#import data
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from Dataset import TranslateDataset
from helpers import get_data
from Config import config
from Transformer import Transformer
from Preprocessing import gettokens,numericalise_tokens_wrapper,tokenizer,vocab_builder,get_masks,tensorising

torch.cuda.empty_cache()

sourcedata,targetdata = get_data('englishtrain.txt','frenchtrain.txt',config)


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
src_tokenzier = tokenizer(config,source=True)
trg_tokenizer = tokenizer(config,source=False)
src_converter = numericalise_tokens_wrapper(config,src_vocab=src_vocab,source=True)
trg_converter = numericalise_tokens_wrapper(config,trg_vocab=trg_vocab,source=False)

#convert our source data to tensors, throw them into a dataset and dataloader.
sourcedata,targetdata = tensorising(sourcedata,targetdata,src_tokenzier,trg_tokenizer,src_converter,trg_converter)
translatedatasets= TranslateDataset(firstlang=sourcedata,secondlang=targetdata)
dataloaded = DataLoader(dataset=translatedatasets,batch_size=config['batch_size'],drop_last=True)

#setup our optimizer and loss function, and dont spend a few hours wondering why it wont train with a 1e-2 learning rate
#and instead keep it below/around 1e-4/5
adam = opt.Adam(transformer.parameters(),lr=1e-5)
loss = torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=src_vocab[config['pad_token']])

for epoch in range(config['epochs']):
    total_loss =0
    batches=0
    print(f'Processing epoch {epoch}....')
    for num_batch,batch in enumerate(dataloaded):
        transformer.train()
        batches+=1
        #Generate the masks for that particular batch
        eng_mask,fr_attn_msk,fr_pad_mask = get_masks(batch[0],batch[1])
        eng_mask = eng_mask[:,None,:,:]
        fr_attn_msk = fr_attn_msk[:,None,:,:]
        fr_pad_mask = fr_pad_mask[:,None,:,:]
        results = transformer(batch[0].to('cuda'),batch[1].to('cuda'),eng_mask.to('cuda'),fr_attn_msk.to('cuda'),fr_pad_mask.to('cuda'))
        results = results.view(-1,config['max_tokens'])
        batch[1] = batch[1].view(-1).to('cuda')
        cur_loss = loss(results.view(-1,len(trg_vocab)),batch[1])
        total_loss+=cur_loss.item()
        cur_loss.backward()
        adam.step()
        adam.zero_grad()
    print(f'Finished {epoch} Epoch with: {total_loss/batches} average loss')


