import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import fairseq
from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel
from fairseq.models.fconv import Embedding, PositionalEmbedding, FConvDecoder
from torch.distributions import Normal


class NoEncoder(FairseqEncoder):
    """
    The input contain:
        sequence of latent embedding indecies
        class index (positive / negative)
        embed the input and noise the sample embeddings.
    """
    def __init__(self, device, sample_size, padding_index, ntokens=5, embed_dim=512, noise_std=0.1, dropout=0.1):
        """
        number of latent-space tokens is constant.
        """
        super().__init__(None)
        self.device = device
        self.dropout = dropout
        self.dim = embed_dim
        self.ntokens = ntokens

        self.content_embeddings = Embedding(sample_size, embed_dim * ntokens, padding_index) # tokens-encoder, sample-specific

        self.negative_embedding = PositionalEmbedding(num_embeddings=ntokens+1,
                                                      embedding_dim=embed_dim,
                                                      padding_idx=0)

        self.positive_embedding = PositionalEmbedding(num_embeddings=ntokens+1,
                                                      embedding_dim=embed_dim,
                                                      padding_idx=0)

        self.noise = Normal(loc=0.0, scale=noise_std)

    def forward(self, src_tokens, src_lengths):
        """
        src_tokens are two: one for the sentiment (0 or 1),
                            and one for the sample [0.. sample_size]
                            shape is always (batch, 2)
        src_lengths is (batch)-size array full of 2.
        """

        batch_size = src_tokens.size()[0]

        # content embedding and noise
        content = self.content_embeddings(src_tokens[:, 0])
        content = content.view(batch_size, self.ntokens, self.dim)
        content = content + self.noise.sample(sample_shape=content.size()).to(self.device)

        # sentiment positional embedding
        positions = torch.arange(1, self.ntokens+1).unsqueeze(0).to(self.device) # 1 x ntokens
        sentiment = src_tokens[:, 1].unsqueeze(1).unsqueeze(2) # batch x 1 x 1

        sentiment = self.positive_embedding(positions) * sentiment + \
                    self.negative_embedding(positions) * (torch.tensor(1) - sentiment) # batch x ntokens x dim

        x = content + sentiment
        x = F.dropout(x, p=self.dropout, training=self.training)

        return {
            'encoder_out': (x,x),
            'encoder_padding_mask': None
        }


class NoEncoderFConvDecoderModel(FairseqEncoderDecoderModel):
    """
    encoder-decoder that use the no-encoder as encoder and the fconv decoder as decoder.
    inspiration from fconv.py
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)


def create_model(device, nsamples, padding_index, ntokens, dim, noise_std, dropout,
                 dictionary, max_positions, nconv) -> NoEncoderFConvDecoderModel:
    # build model

    encoder = NoEncoder(device=device,
                        sample_size=nsamples,
                        padding_index=padding_index,
                        ntokens=ntokens,
                        embed_dim=dim,
                        noise_std=noise_std,
                        dropout=dropout)

    decoder = FConvDecoder(dictionary,
                           embed_dim=dim,
                           out_embed_dim=dim,
                           max_positions=max_positions,
                           convolutions=((dim, 3),) * nconv)

    model = NoEncoderFConvDecoderModel(encoder, decoder).to(device)

    return model



