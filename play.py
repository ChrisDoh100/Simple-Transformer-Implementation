from datasets import load_dataset
import sentencepiece as spm
import os
from Dataset import TranslateDataset


def load_and_generate_data(dataset_type,source_ext, target_ext):
    """Gives a choice between a larger,wmt14 that was in the original paper, 
        or the smaller iswlt dataset which gives quicker results. A full list 
        of available languages should be available on the huggingface dataset 
        page for each dataset."""
    
    if dataset_type=='wmt':
        data = load_dataset("wmt/wmt14",source_ext,target_ext)
    else:
        data = load_dataset("IWSLT/iwslt2017",f'iwslt2017-{source_ext}-{target_ext}')

    return data

def generate_vocab(dataset,source_ext,target_ext):
    """We are going to use sentencepiece, to generate a bpe encoded vocab
        this was surprisingly quick and easy."""
    
    with open(f'{source_ext}{target_ext}train.txt', 'w', encoding='utf-8') as file:
        for x in dataset['train'][:]['translation']:
            file.write(f"{x[f'{target_ext}']}\n{x[f'{source_ext}']}\n")
    
    spm.SentencePieceTrainer.Train(
        input = f'{source_ext}{target_ext}train.txt',
        model_prefix = 'training',
        model_type = 'bpe',
        vocab_size = 32000,
        pad_id=3,
        shuffle_input_sentence=True,
        bos_id=1,
        eos_id=2,
    )
    os.remove(f'{source_ext}{target_ext}train.txt')



def Encoding_Data(data_object,data_type,source_ext,target_ext):
    """This is going to use our vocabulary to encode all of our data/sentences into numerical values.
        for each of the training set, validation set and test set."""

    sp = spm.SentencePieceProcessor()
    sp.Load('training.model')

    source = [sp.EncodeAsIds(sentence[f'{source_ext}']) for sentence in data_object[f'{data_type}'][:]['translation']]
    target = [sp.EncodeAsIds(sentence[f'{target_ext}']) for sentence in data_object[f'{data_type}'][:]['translation']]
    
    trg_input = [[sp.bos_id()]+encoded for encoded in target]
    trg_output = [encoded+[sp.eos_id()] for encoded in target]

    return source,trg_input,trg_output

def GenerateDatasets(valset,dataset_type,source_ext,target_ext):
    data = load_and_generate_data(dataset_type,source_ext,target_ext)
    generate_vocab(data,source_ext,target_ext)
    if valset:
        source,target_inp,target_out = Encoding_Data(data,'train',source_ext,target_ext)
        val_source,val_target_inp,val_target_out = Encoding_Data(data,'validation',source_ext,target_ext)
        traindataset = TranslateDataset(source,target_inp,target_out)
        valdataset = TranslateDataset(val_source,val_target_inp,val_target_out)
        return traindataset,valdataset
    
    source,target_inp,target_out = Encoding_Data(data,'train',source_ext,target_ext)
    traindataset = TranslateDataset(source,target_inp,target_out)
    return traindataset
