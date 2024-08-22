#import data
import torch
from Dataset import TranslateDataset
from torch.utils.data import DataLoader
from helpers import get_data,vocab_builder
from Dataloading import get_masks
from Constants import special_tokens1,unk_token,model_dimension,decoderlayers,encoderlayers,heads,max_length,epochs,pad_token
from Transformer import Transformer
from Dataloading import numericalise_tokens_wrapper,tokenizer
import spacy
import torch.nn.functional as F
import torch.optim as opt

torch.cuda.empty_cache()

entokener = spacy.load("en_core_web_sm")
frtokener = spacy.load("fr_core_news_sm")


def getTokens(data_iter):
    for english in data_iter:
        yield [token.text for token in entokener.tokenizer(text)]

def getTokensfr(data_iter):
    for french in data_iter:
        yield [token.text for token in entokener.tokenizer(text)]

englishdata,frenchdata = get_data('english.txt','french.txt')
#en_vocab = vocab_builder(getTokens(englishdata),min_freq=2,special_tokens=special_tokens1,unk_token=unk_token,set_default=True)
#fr_vocab = vocab_builder(getTokensfr(frenchdata),min_freq=2,special_tokens=special_tokens1,unk_token=unk_token,set_default=True)
#torch.save(en_vocab,'en_vocab.pth')
#torch.save(fr_vocab,'fr_vocab.pth')

en_vocab = torch.load('en_vocab.pth')
fr_vocab =torch.load('fr_vocab.pth')
print(len(en_vocab),len(fr_vocab))
#need to tokenize first


translatedatasets= TranslateDataset(firstlang=englishdata,secondlang=frenchdata)

dataloaded = DataLoader(dataset=translatedatasets,batch_size=500)


transformer = Transformer(len(en_vocab),len(fr_vocab),model_dimension,decodelayers=6,encodelayers=6,heads=8,max_seq_len=max_length).to('cuda')

##move all the this process into one function
en_tokenzier = tokenizer(english=True)
fr_tokenizer = tokenizer(english=False)
en_converter = numericalise_tokens_wrapper(en_vocab=en_vocab,english=True)
fr_converter = numericalise_tokens_wrapper(fr_vocab=fr_vocab,english=False)
#waaaaaay too slow, going to do it once and then just load it forever
englishdata =list(map(en_tokenzier,englishdata))
frenchdata= list(map(fr_tokenizer,frenchdata))
#filtering by max_length
data= list(filter(lambda x :(len(x[0])<max_length and len(x[1])<max_length),zip(englishdata,frenchdata)))
englishdata = [x[0] for x in data]
frenchdata = [x[1] for x in data]
englishdata = list(map(en_converter,frenchdata))
frenchdata= list(map(fr_converter,frenchdata))
englishdata = torch.tensor(englishdata)
frenchdata = torch.tensor(frenchdata)
translatedatasets= TranslateDataset(firstlang=englishdata,secondlang=frenchdata)
dataloaded = DataLoader(dataset=translatedatasets,batch_size=50)
adam = opt.Adam(transformer.parameters(),lr=1e-5)
loss = torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=en_vocab[pad_token])
print(transformer)

for epoch in range(epochs):
    total_loss =0
    batches=0
    print(f'Processing epoch {epoch}....')
    for num_batch,batch in enumerate(dataloaded):
        transformer.train()
        batches+=1
        eng_mask,fr_attn_msk,fr_pad_mask = get_masks(batch[0],batch[1],max_seq_length=max_length)
        #adding a dimension so the masks can be broadcasting over all the heads.
        eng_mask = eng_mask[:,None,:,:]
        fr_attn_msk = fr_attn_msk[:,None,:,:]
        fr_pad_mask = fr_pad_mask[:,None,:,:]
        #convert batches
        results = transformer(batch[0].to('cuda'),batch[1].to('cuda'),eng_mask.to('cuda'),fr_attn_msk.to('cuda'),fr_pad_mask.to('cuda'))
        results = results.view(-1,len(fr_vocab))
        batch[1] = batch[1].view(-1).to('cuda')
        cur_loss = loss(results.view(-1,len(fr_vocab)),batch[1])
        total_loss+=cur_loss.item()
        cur_loss.backward()
        adam.step()
        adam.zero_grad()
    print(f'Finished {epoch} Epoch with: {total_loss/batches} average loss')
#batch in dataloader
#tokenize
#get masks
#numericalize
#model
#loss
#validation
#metrics
