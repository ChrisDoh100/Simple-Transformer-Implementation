import torch
import torch.nn as nn
from Encoder import Encoder,LayerEncoder
from Decoder import Decoder,LayerDecoder
from Embeddings import Embeddings
from PositionalEncoding import PositionalEncodings
from DecoderOutputGen import DecoderOutputGenerator

class Transformer(nn.Module):
    #encoder +layers
    #decorder +layers
    #target_embeddings
    #source_embeddings
    #positional_embeddings
    #decoder output layer
    def __init__(self,vocab_size_en,vocab_size_fr,dmodel,decodelayers,encodelayers,heads,max_seq_len):
        super().__init__()
        self.encoder = Encoder(encodelayers,LayerEncoder(heads=heads,dmodel=dmodel))
        self.decoder= Decoder(decodelayers,LayerDecoder(heads=heads,dmodel=dmodel))
        self.en_embed = Embeddings(vocab_size_en,dmodel)
        self.fr_embed = Embeddings(vocab_size_fr,dmodel)
        self.pos_en_embed = PositionalEncodings(dmodel,max_sequence_length=max_seq_len)
        self.pos_fr_embed = PositionalEncodings(dmodel,max_sequence_length=max_seq_len)
        self.decodergenerator = DecoderOutputGenerator(dmodel,vocab_size=vocab_size_fr)

    def encode_preamble(self,x):
        src_encoding_embeddings = self.en_embed(x)
        src_encodingpos_embeddings = self.pos_en_embed(src_encoding_embeddings)
        return src_encodingpos_embeddings
     
    def decode_preamble(self,x):
        trg_encoding_embeddings = self.fr_embed(x)
        trg_encodingpos_embeddings = self.pos_fr_embed(trg_encoding_embeddings)
        return trg_encodingpos_embeddings

    def forward(self,src_tokens,trg_tokens,src_mask,trg_mask):
        src_encoding = self.encode_preamble(src_tokens)
        encoder_output = self.encoder(src_encoding,src_mask)
        trg_encoding = self.decode_preamble(trg_tokens)
        decoder_output = self.decoder(trg_encoding,encoder_output,src_mask,trg_mask)
        answer = self.decodergenerator(decoder_output)
        return answer

        

