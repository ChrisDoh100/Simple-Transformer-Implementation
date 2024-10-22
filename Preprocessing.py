import torch
from Config import config
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import torch.nn as nn


sp = spm.SentencePieceProcessor()
sp.Load('training.model')

def get_source_mask(src_seqs,pad_token):
    src_mask = (src_seqs != pad_token).unsqueeze(1).unsqueeze(2)
    return src_mask

def get_trg_mask(trg_seqs, pad_token_id):

    batch_size = trg_seqs.shape[0]
    sequence_length = trg_seqs.shape[1]
    trg_padding_mask = (trg_seqs != pad_token_id).view(batch_size,1,1,-1).to('cuda')
    trg_no_look_forward_mask = torch.triu(torch.ones((batch_size,1,sequence_length, sequence_length)) == 1).transpose(2, 3).to('cuda')
    trg_mask = trg_padding_mask & trg_no_look_forward_mask

    return trg_mask

def collate_fn(batch):
    # Sort batch by sequence length (descending order)
    # Separate the batch into input sequences, target sequences, and additional features
    (actual_input_sequences,decoder_input_sequences,decoder_output_sequences) = zip(*batch)
    # Pad sequences to the maximum length in the batch
    padded_input_sequences = pad_sequence(actual_input_sequences, batch_first=True, padding_value=sp.pad_id())
    padded_decoder_input_sequences = pad_sequence(decoder_input_sequences, batch_first=True, padding_value=sp.pad_id())
    padded_decoder_output_sequences = pad_sequence(decoder_output_sequences, batch_first=True, padding_value=sp.pad_id())

    max_padding_needed = max(padded_input_sequences.size(1),padded_decoder_input_sequences.size(1),padded_decoder_output_sequences.size(1))

    padded_input_sequences = nn.functional.pad(padded_input_sequences, (0, max_padding_needed - padded_input_sequences.size(1)), value=sp.pad_id())
    padded_decoder_input_sequences = nn.functional.pad(padded_decoder_input_sequences, (0, max_padding_needed - padded_decoder_input_sequences.size(1)), value=sp.pad_id())
    padded_decoder_output_sequences = nn.functional.pad(padded_decoder_output_sequences, (0, max_padding_needed - padded_decoder_output_sequences.size(1)), value=sp.pad_id())
    
    # Generate padding masks
    src_mask = get_source_mask(padded_input_sequences,sp.pad_id())
    trg_mask =get_trg_mask(padded_decoder_input_sequences,sp.pad_id())
    return (padded_input_sequences.to('cuda'), src_mask.to('cuda')), \
           (padded_decoder_input_sequences.to('cuda'),trg_mask.to('cuda')), \
           (padded_decoder_output_sequences.view(-1,1).to('cuda'))


