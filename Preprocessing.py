import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn



def get_source_mask(src_seqs,pad_token):
    """Provides the mask for the encoder input."""
    src_mask = (src_seqs != pad_token).unsqueeze(1).unsqueeze(2)
    return src_mask

def get_trg_mask(trg_seqs, pad_token_id):
    """Provides the mask for the decoder input."""
    batch_size = trg_seqs.shape[0]
    sequence_length = trg_seqs.shape[1]
    trg_padding_mask = (trg_seqs != pad_token_id).view(batch_size,1,1,-1).to('cuda')
    trg_no_look_forward_mask = torch.triu(torch.ones((batch_size,1,sequence_length, sequence_length)) == 1).transpose(2, 3).to('cuda')
    #If this is confusing I would suggest looking into bitwise operators or perhaps do a few problems that involve bit operations on leetcode/codeforces.
    #Extremely useful for learning!
    trg_mask = trg_padding_mask & trg_no_look_forward_mask

    return trg_mask

def collate_fn(batch):
    """Generates a batch with padding and global padding to the length of the longest sequence in the batch."""
    (actual_input_sequences,decoder_input_sequences,decoder_output_sequences) = zip(*batch)
    #This adds padding to each of the three sequences to ensure each one is the same length.
    #And ensures they can actually be passed to the transformer.
    padded_input_sequences = pad_sequence(actual_input_sequences, batch_first=True, padding_value=3)
    padded_decoder_input_sequences = pad_sequence(decoder_input_sequences, batch_first=True, padding_value=3)
    padded_decoder_output_sequences = pad_sequence(decoder_output_sequences, batch_first=True, padding_value=3)
    
    src_mask = get_source_mask(padded_input_sequences,3)
    trg_mask =get_trg_mask(padded_decoder_input_sequences,3)
    return (padded_input_sequences.to('cuda'), src_mask.to('cuda')), \
           (padded_decoder_input_sequences.to('cuda'),trg_mask.to('cuda')), \
           (padded_decoder_output_sequences.view(-1,1).to('cuda'))


