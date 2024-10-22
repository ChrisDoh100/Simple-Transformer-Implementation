import torch
import math
import torch.nn as nn
import sentencepiece as spm
import math
from Embeddings import Embeddings
from PositionalEncoding import PositionalEncodings
from Decoder import LayerDecoder,Decoder
from Encoder import LayerEncoder,Encoder
from DecoderOutputGen import DecoderOutputGenerator

sp = spm.SentencePieceProcessor()
sp.Load('training.model')

class Transformer(nn.Module):
    def __init__(self,config,vocab_size):
        super().__init__()
        self.embed = Embeddings(vocab_size,config['model_dimension'])
        self.pos_embed = PositionalEncodings(config['model_dimension'])
        encoderlayer = LayerEncoder(heads=config['heads'],dmodel=config['model_dimension'])
        decoderlayer = LayerDecoder(heads=config['heads'],dmodel=config['model_dimension'])
        self.encoder = Encoder(config['encoderlayers'],encoderlayer)
        self.decoder= Decoder(config['decoderlayers'],decoderlayer)
        self.decodergenerator = DecoderOutputGenerator(config['model_dimension'],vocab_size=vocab_size)
        self.decodergenerator.lin.weight = self.embed.embed.weight
        self.init_params()
    
    def init_params(self):
        """Initialises the Transformer"""
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_normal_(p)

    def forward(self,src_tokens,trg_tokens,src_mask,trg_mask):
        """Forward pass of the transfomer, which returns a matrix with probabilites
            of the output sequence based on the target vocab size."""
        src_encoding = self.encode(src_tokens,src_mask)
        trg_encoding = self.decode(trg_tokens,src_encoding,trg_mask,src_mask)
        return trg_encoding
    
    def encode(self,x,src_mask):
        """Encoding Block of the Tranformer"""
        src_encoding_embeddings = self.embed(x)
        src_encodingpos_embeddings = self.pos_embed(src_encoding_embeddings)
        encoder_output = self.encoder(src_encodingpos_embeddings,src_mask)
        return encoder_output
     
    def decode(self,x,encoder_output,trg_mask,src_mask):
        """Decoding Block of the Tranformer"""
        trg_encoding_embeddings = self.embed(x)
        trg_encodingpos_embeddings = self.pos_embed(trg_encoding_embeddings)
        decoder_output = self.decoder(trg_encodingpos_embeddings,encoder_output,trg_mask,src_mask)
        answer = self.decodergenerator(decoder_output)
        return answer


