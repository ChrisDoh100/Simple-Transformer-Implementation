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
    def __init__(self,config):
        super().__init__()
        self.en_embed = Embeddings(config['max_tokens'],config['model_dimension'])
        self.fr_embed = Embeddings(config['max_tokens'],config['model_dimension'])
        self.pos_en_embed = PositionalEncodings(config['model_dimension'],max_sequence_length=config['max_length'])
        self.pos_fr_embed = PositionalEncodings(config['model_dimension'],max_sequence_length=config['max_length'])
        self.encoder = Encoder(config['encoderlayers'],LayerEncoder(heads=config['heads'],dmodel=config['model_dimension']))
        self.decoder= Decoder(config['decoderlayers'],LayerDecoder(heads=config['heads'],dmodel=config['model_dimension']))
        self.decodergenerator = DecoderOutputGenerator(config['model_dimension'],vocab_size=config['max_tokens'])
        self.init_params()
    
    def init_params(self):
        """Initialises the Transformer"""
        for p in self.parameters():
            if p.dim()>1:
                torch.nn.init.xavier_normal_(p)

    def encode_preamble(self,x,eng_mask):
        """Encoding Block of the Tranformer"""
        src_encoding_embeddings = self.en_embed(x)
        src_encodingpos_embeddings = self.pos_en_embed(src_encoding_embeddings)
        encoder_output = self.encoder(src_encodingpos_embeddings,eng_mask)
        return encoder_output
     
    def decode_preamble(self,x,encoder_output,fr_pad_mask,fr_attn_mask):
        """Decoding Block of the Tranformer"""
        trg_encoding_embeddings = self.fr_embed(x)
        trg_encodingpos_embeddings = self.pos_fr_embed(trg_encoding_embeddings)
        decoder_output = self.decoder(trg_encodingpos_embeddings,encoder_output,fr_pad_mask,fr_attn_mask)
        return decoder_output

    def forward(self,src_tokens,trg_tokens,eng_mask,fr_pad_msk,fr_attn_msk):
        """Forward pass of the transfomer, which returns a matrix with probabilites
            of the output sequence."""
        src_encoding = self.encode_preamble(src_tokens,eng_mask=eng_mask)
        trg_encoding = self.decode_preamble(trg_tokens,src_encoding,fr_pad_mask=fr_pad_msk,fr_attn_mask=fr_attn_msk)
        answer = self.decodergenerator(trg_encoding)
        return answer



        

