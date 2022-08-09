#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from ane_transformers.testing_utils import assert_rank, assert_shape

import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from . import encoder, decoder, multihead_attention, ffn


class AppleNeuralEngineTransformer(nn.Module):
    """ An Apple Neural Engine (ANE) optimized Transformer implementation
    """

    def __init__(
        self,
        *,
        embed_dim=512,
        ffn_dim=2048,
        enc_self_attn_type=multihead_attention.ResidualSelfAttention,
        enc_ffn_type=ffn.ResidualFFN,
        dec_self_attn_type=multihead_attention.ResidualSelfAttention,
        enc_dec_attn_type=multihead_attention.ResidualMultiHeadAttention,
        dec_ffn_type=ffn.ResidualFFN,
        nb_enc_layers=6,
        nb_dec_layers=6,
        nb_attention_heads=8,
        dropout=0.1,
        return_intermediate_decoder_layers=False,
        **kwargs,
    ):
        """
        Args:
            embed_dim:                          Dimensionality of the embedding space attention is computed in
            ffn_dim:                            Number of channels to use in the feed-forward network's hidden layer
            enc_self_attn_type:                 The self-attention module to use in each layer of Transformer encoder
            enc_ffn_type:                       The feed-forward network module to use in each layer of the Transformer encoder
            dec_self_attn_type:                 The self-attention module to use in each layer of Transformer decoder
            enc_dec_attn_type:                  The encoder-decoder attention module to use in each layer of Transformer decoder
            dec_ffn_type:                       The feed-forward network module to use in each layer of the Transformer decoder
            nb_enc_layers:                      Number of identically configured Transformer encoder layers to stack
            nb_dec_layers:                      Number of identically configured Transformer decoder layers to stack
            nb_attention_heads:                 Number of attention heads generate in each and every attention block in the Transformer model
            dropout:                            The dropout probability (`1- keep_probability`) to use on both the attention weights and the
                                                output of the attention block in each and every attention block in the Transformer model
            return_intermediate_decoder_layers: If True, returns the output of all Transformer decoder layers stacked in the 0-th axis.
                                                Example use case: When supervising the Transformer on all intermediate outputs for training stability

        Note: The positional embeddings are passed by the caller of forward() and are not part of this module

        Note: The default configuration reflects the "base" configuration in the original Transformer paper [1] (page 9, table 3, row 1)

        [1] https://arxiv.org/pdf/1706.03762
        """
        super().__init__()
        self.encoder = encoder.TransformerEncoder(
            layer=encoder.TransformerEncoderLayer(
                embed_dim,
                ffn_dim,
                enc_self_attn_type,
                enc_ffn_type,
                nb_attention_heads,
                dropout,
            ),
            num_layers=nb_enc_layers)
        self.decoder = decoder.TransformerDecoder(
            layer=decoder.TransformerDecoderLayer(
                embed_dim,
                ffn_dim,
                dec_self_attn_type,
                enc_dec_attn_type,
                dec_ffn_type,
                nb_attention_heads,
                dropout,
            ),
            num_layers=nb_dec_layers)

        self.return_intermediate_decoder = return_intermediate_decoder_layers
        self.embed_dim = embed_dim

    def forward(
        self,
        encoder_input,
        decoder_input,
        encoder_pos_embed=None,
        decoder_pos_embed=None,
        encoder_k_mask=None,
        decoder_k_mask=None,
        encoder_qk_mask=None,
        decoder_qk_mask=None,
    ):
        """
        Notation:
            src_seq_len:        The sequence length of the source sequence which is the input to the TransformerEncoder
            tgt_seq_len:        The sequence length of the target sequence which is the input to the TransformerDecoder

        Args:
            encoder_input:      Float tensor input to the TransformerEncoder
            decoder_input:      Float tensor input to the TransformerDecoder
            encoder_pos_embed:  Same shape and dtype as `encoder_input`. Serves as additive positional encodings to `encoder_input`
            decoder_pos_embed:  Same shape and dtype as `decoder_input`. Serves as additive positional encodings to `decoder_input`
            encoder_k_mask:     Float tensor similar to the `src_key_padding_mask` in `torch.nn.Transformer.forward`. Example use: masking zero-padded tokens in the source sequence
            decoder_k_mask:     Float tensor similar to the `tgt_key_padding_mask` in `torch.nn.Transformer.forward`. Example use: masking zero-padded tokens in the target sequence
            encoder_qk_mask:    Float tensor similar to `src_mask` in `torch.nn.Transformer.forward`. Example use: masking future tokens in the encoder self-attention
            decoder_qk_mask:    Float tensor similar to `tgt_mask` in `torch.nn.Transformer.forward`. Example use: masking future tokens in the decoder self-attention

        Shapes:
            encoder_input:      (batch_size, embed_dim, 1, src_seq_len)
            decoder_input:      (batch_size, embed_dim, 1, tgt_seq_len)
            encoder_pos_embed:  (batch_size, embed_dim, 1, src_seq_len)
            decoder_pos_embed:  (batch_size, embed_dim, 1, tgt_seq_len)
            encoder_k_mask:     (batch_size, src_seq_len, 1, 1)
            decoder_k_mask:     (batch_size, tgt_seq_len, 1, 1)
            encoder_qk_mask:    (batch_size, src_seq_len, 1, src_seq_len)
            decoder_qk_mask:    (batch_size, tgt_seq_len, 1, tgt_seq_len)

        Returns:
            decoder_output:     Output of the TransformerDecoder
            encoder_output:     Output of the TransformerEncoder


        Note: All arguments ending in "_mask", are applied additively on the intermediate tensor right before softmax in the attention function.
        The recommended float value for preventing attention is -1e4. This allows for composition of multiple masks while staying in the float16-friendly range.
        Use a value of 0 to keep attention unchanged.
        """
        # Verify ranks
        assert_rank(encoder_input, "encoder_input", 4)
        assert_rank(encoder_input, "decoder_input", 4)
        assert_rank(encoder_pos_embed, "encoder_pos_embed", 4)
        assert_rank(decoder_pos_embed, "decoder_pos_embed", 4)
        assert_rank(encoder_k_mask, "encoder_k_mask", 4)
        assert_rank(decoder_k_mask, "decoder_k_mask", 4)
        assert_rank(encoder_k_mask, "encoder_qk_mask", 4)
        assert_rank(decoder_k_mask, "decoder_qk_mask", 4)

        # Verify and prepare encoder inputs
        batch_size, _, _, src_seq_len = encoder_input.shape
        assert_shape(encoder_input, "encoder_input",
                     [batch_size, self.embed_dim, 1, src_seq_len])

        if encoder_pos_embed is not None:
            assert_shape(encoder_pos_embed, "encoder_pos_embed",
                         [batch_size, self.embed_dim, 1, src_seq_len])
        if encoder_k_mask is not None:
            assert_shape(encoder_k_mask, "encoder_k_mask",
                         [batch_size, src_seq_len, 1, 1])
        if encoder_qk_mask is not None:
            assert_shape(encoder_qk_mask, "encoder_qk_mask",
                         [batch_size, src_seq_len, 1, src_seq_len])

        # Verify and prepare decoder inputs
        batch_size, _, _, tgt_seq_len = decoder_input.shape
        assert_shape(decoder_input, "decoder_input",
                     [batch_size, self.embed_dim, 1, tgt_seq_len])

        if decoder_pos_embed is not None:
            assert_shape(decoder_pos_embed, "decoder_pos_embed",
                         [batch_size, self.embed_dim, 1, tgt_seq_len])
        if decoder_k_mask is not None:
            assert_shape(decoder_k_mask, "decoder_k_mask",
                         [batch_size, tgt_seq_len, 1, 1])
        if decoder_qk_mask is not None:
            assert_shape(decoder_qk_mask, "decoder_qk_mask",
                         [batch_size, tgt_seq_len, 1, tgt_seq_len])

        # TransformerEncoder forward pass
        encoder_output = self.encoder(
            encoder_input,
            pos_embed=encoder_pos_embed,
            qk_mask=encoder_qk_mask,
            k_mask=encoder_k_mask,
        )

        # TransformerDecoder forward pass
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            encoder_k_mask=encoder_k_mask,
            decoder_k_mask=decoder_k_mask,
            decoder_qk_mask=decoder_qk_mask,
            decoder_pos_embed=decoder_pos_embed,
            encoder_pos_embed=encoder_pos_embed,
            return_intermediate=self.return_intermediate_decoder,
        )

        return decoder_output, encoder_output
