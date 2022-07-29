#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from ane_transformers.reference.layer_norm import LayerNormANE

import torch
import torch.nn as nn

from transformers.models.distilbert import modeling_distilbert

# Note: Original implementation of distilbert uses an epsilon value of 1e-12
# which is not friendly with the float16 precision that ANE uses by default
EPS = 1e-7

WARN_MSG_FOR_TRAINING_ATTEMPT = \
    "This model is optimized for on-device execution only. " \
    "Please use the original implementation from Hugging Face for training"

WARN_MSG_FOR_DICT_RETURN = \
    "coremltools does not support dict outputs. Please set return_dict=False"


# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix +
               'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix +
                                                                  'weight']
    return state_dict


class LayerNormANE(LayerNormANE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(
            correct_for_bias_scale_order_inversion)


class Embeddings(modeling_distilbert.Embeddings):
    """ Embeddings module optimized for Apple Neural Engine
    """

    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'LayerNorm', LayerNormANE(config.dim, eps=EPS))


class MultiHeadSelfAttention(modeling_distilbert.MultiHeadSelfAttention):
    """ MultiHeadSelfAttention module optimized for Apple Neural Engine
    """

    def __init__(self, config):
        super().__init__(config)

        setattr(
            self, 'q_lin',
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.dim,
                kernel_size=1,
            ))

        setattr(
            self, 'k_lin',
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.dim,
                kernel_size=1,
            ))

        setattr(
            self, 'v_lin',
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.dim,
                kernel_size=1,
            ))

        setattr(
            self, 'out_lin',
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.dim,
                kernel_size=1,
            ))

    def prune_heads(self, heads):
        raise NotImplementedError

    def forward(self,
                query,
                key,
                value,
                mask,
                head_mask=None,
                output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, dim, 1, seq_length)
            key: torch.tensor(bs, dim, 1, seq_length)
            value: torch.tensor(bs, dim, 1, seq_length)
            mask: torch.tensor(bs, seq_length) or torch.tensor(bs, seq_length, 1, 1)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            dim, 1, seq_length) Contextualized layer. Optional: only if `output_attentions=True`
        """
        # Parse tensor shapes for source and target sequences
        assert len(query.size()) == 4 and len(key.size()) == 4 and len(
            value.size()) == 4

        bs, dim, dummy, seqlen = query.size()
        # assert seqlen == key.size(3) and seqlen == value.size(3)
        # assert dim == self.dim
        # assert dummy == 1

        # Project q, k and v
        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)

        # Validate mask
        if mask is not None:
            expected_mask_shape = [bs, seqlen, 1, 1]
            if mask.dtype == torch.bool:
                mask = mask.logical_not().float() * -1e4
            elif mask.dtype == torch.int64:
                mask = (1 - mask).float() * -1e4
            elif mask.dtype != torch.float32:
                raise TypeError(f"Unexpected dtype for mask: {mask.dtype}")

            if len(mask.size()) == 2:
                mask = mask.unsqueeze(2).unsqueeze(2)

            if list(mask.size()) != expected_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `mask` (Expected {expected_mask_shape}, got {list(mask.size())}"
                )

        if head_mask is not None:
            raise NotImplementedError

        # Compute scaled dot-product attention
        dim_per_head = self.dim // self.n_heads
        mh_q = q.split(
            dim_per_head,
            dim=1)  # (bs, dim_per_head, 1, max_seq_length) * n_heads
        mh_k = k.transpose(1, 3).split(
            dim_per_head,
            dim=3)  # (bs, max_seq_length, 1, dim_per_head) * n_heads
        mh_v = v.split(
            dim_per_head,
            dim=1)  # (bs, dim_per_head, 1, max_seq_length) * n_heads

        normalize_factor = float(dim_per_head)**-0.5
        attn_weights = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki]) * normalize_factor
            for qi, ki in zip(mh_q, mh_k)
        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads

        if mask is not None:
            for head_idx in range(self.n_heads):
                attn_weights[head_idx] = attn_weights[head_idx] + mask

        attn_weights = [aw.softmax(dim=1) for aw in attn_weights
                        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads
        attn = [
            torch.einsum('bkhq,bchk->bchq', wi, vi)
            for wi, vi in zip(attn_weights, mh_v)
        ]  # (bs, dim_per_head, 1, max_seq_length) * n_heads

        attn = torch.cat(attn, dim=1)  # (bs, dim, 1, max_seq_length)

        attn = self.out_lin(attn)

        if output_attentions:
            return attn, attn_weights.cat(dim=2)
        else:
            return (attn, )


class FFN(modeling_distilbert.FFN):
    """ FFN module optimized for Apple Neural Engine
    """

    def __init__(self, config):
        super().__init__(config)
        self.seq_len_dim = 3

        setattr(
            self, 'lin1',
            nn.Conv2d(
                in_channels=config.dim,
                out_channels=config.hidden_dim,
                kernel_size=1,
            ))

        setattr(
            self, 'lin2',
            nn.Conv2d(
                in_channels=config.hidden_dim,
                out_channels=config.dim,
                kernel_size=1,
            ))


class TransformerBlock(modeling_distilbert.TransformerBlock):

    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'attention', MultiHeadSelfAttention(config))
        setattr(self, 'sa_layer_norm', LayerNormANE(config.dim, eps=EPS))
        setattr(self, 'ffn', FFN(config))
        setattr(self, 'output_layer_norm', LayerNormANE(config.dim, eps=EPS))


class Transformer(modeling_distilbert.Transformer):

    def __init__(self, config):
        super().__init__(config)
        setattr(
            self, 'layer',
            nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.n_layers)]))


class DistilBertModel(modeling_distilbert.DistilBertModel):

    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'embeddings', Embeddings(config))
        setattr(self, 'transformer', Transformer(config))

        # Register hook for unsqueezing nn.Linear parameters to match nn.Conv2d parameter spec
        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError


class DistilBertForMaskedLM(modeling_distilbert.DistilBertForMaskedLM):

    def __init__(self, config):
        super().__init__(config)
        from transformers.activations import get_activation
        setattr(self, 'activation', get_activation(config.activation))
        setattr(self, 'distilbert', DistilBertModel(config))
        setattr(self, 'vocab_transform', nn.Conv2d(config.dim, config.dim, 1))
        setattr(self, 'vocab_layer_norm', LayerNormANE(config.dim, eps=EPS))
        setattr(self, 'vocab_projector',
                nn.Conv2d(config.dim, config.vocab_size, 1))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if self.training or labels is not None:
            raise ValueError(WARN_MSG_FOR_TRAINING_ATTEMPT)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict:
            raise ValueError(WARN_MSG_FOR_DICT_RETURN)

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        hidden_states = dlbrt_output[0]  # (bs, dim, 1, seq_len)
        prediction_logits = self.vocab_transform(
            hidden_states)  # (bs, dim, 1, seq_len)
        prediction_logits = self.activation(
            prediction_logits)  # (bs, dim, 1, seq_len)
        prediction_logits = self.vocab_layer_norm(
            prediction_logits)  # (bs, dim, 1, seq_len)
        prediction_logits = self.vocab_projector(
            prediction_logits)  # (bs, dim, 1, seq_len)
        prediction_logits = prediction_logits.squeeze(-1).squeeze(
            -1)  # (bs, dim)

        output = (prediction_logits, ) + dlbrt_output[1:]
        mlm_loss = None

        return ((mlm_loss, ) + output) if mlm_loss is not None else output


class DistilBertForSequenceClassification(
        modeling_distilbert.DistilBertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'distilbert', DistilBertModel(config))
        setattr(self, 'pre_classifier', nn.Conv2d(config.dim, config.dim, 1))
        setattr(self, 'classifier', nn.Conv2d(config.dim, config.num_labels,
                                              1))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if labels is not None or self.training:
            raise NotImplementedError(WARN_MSG_FOR_TRAINING_ATTEMPT)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict:
            raise ValueError(WARN_MSG_FOR_DICT_RETURN)

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        hidden_state = distilbert_output[0]  # (bs, dim, 1, seq_len)
        pooled_output = hidden_state[:, :, :, 0:1]  # (bs, dim, 1, 1)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim, 1, 1)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim, 1, 1)
        logits = self.classifier(pooled_output)  # (bs, num_labels, 1, 1)
        logits = logits.squeeze(-1).squeeze(-1)  # (bs, num_labels)

        output = (logits, ) + distilbert_output[1:]
        loss = None

        return ((loss, ) + output) if loss is not None else output


class DistilBertForQuestionAnswering(
        modeling_distilbert.DistilBertForQuestionAnswering):

    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'distilbert', DistilBertModel(config))
        setattr(self, 'qa_outputs', nn.Conv2d(config.dim, config.num_labels,
                                              1))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if self.training or start_positions is not None or end_positions is not None:
            raise ValueError(WARN_MSG_FOR_TRAINING_ATTEMPT)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict:
            raise ValueError(WARN_MSG_FOR_DICT_RETURN)

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        hidden_states = distilbert_output[0]  # (bs, dim, 1, max_query_len)

        hidden_states = self.dropout(
            hidden_states)  # (bs, dim, 1, max_query_len)
        logits = self.qa_outputs(hidden_states)  # (bs, 2, 1, max_query_len)
        start_logits, end_logits = logits.split(
            1, dim=1)  # (bs, 1, 1, max_query_len) * 2
        start_logits = start_logits.squeeze().contiguous(
        )  # (bs, max_query_len)
        end_logits = end_logits.squeeze().contiguous()  # (bs, max_query_len)

        output = (start_logits, end_logits) + distilbert_output[1:]
        total_loss = None

        return ((total_loss, ) + output) if total_loss is not None else output


class DistilBertForTokenClassification(
        modeling_distilbert.DistilBertForTokenClassification):

    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'distilbert', DistilBertModel(config))
        setattr(self, 'classifier',
                nn.Conv2d(config.hidden_size, config.num_labels, 1))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if self.training or labels is not None:
            raise ValueError(WARN_MSG_FOR_TRAINING_ATTEMPT)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict:
            raise ValueError(WARN_MSG_FOR_DICT_RETURN)

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        sequence_output = outputs[0]  # (bs, dim, 1, seq_len)
        logits = self.classifier(
            sequence_output)  # (bs, num_labels, 1, seq_len)
        logits = logits.squeeze(2).transpose(1, 2)  # (bs, seq_len, num_labels)

        output = (logits, ) + outputs[1:]
        loss = None
        return ((loss, ) + output) if loss is not None else output


class DistilBertForMultipleChoice(
        modeling_distilbert.DistilBertForMultipleChoice):

    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'distilbert', DistilBertModel(config))
        setattr(self, 'pre_classifier', nn.Conv2d(config.dim, config.dim, 1))
        setattr(self, 'classifier', nn.Conv2d(config.dim, 1, 1))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if self.training or labels is not None:
            raise ValueError(WARN_MSG_FOR_TRAINING_ATTEMPT)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if return_dict:
            raise ValueError(WARN_MSG_FOR_DICT_RETURN)

        num_choices = input_ids.shape[
            1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(
            -1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(
            -1,
            attention_mask.size(-1)) if attention_mask is not None else None
        inputs_embeds = (inputs_embeds.view(-1, inputs_embeds.size(-2),
                                            inputs_embeds.size(-1))
                         if inputs_embeds is not None else None)

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        hidden_state = outputs[0]  # (bs * num_choices, dim, 1, seq_len)
        pooled_output = hidden_state[:, :, :,
                                     0:1]  # (bs * num_choices, dim, 1, 1)
        pooled_output = self.pre_classifier(
            pooled_output)  # (bs * num_choices, dim, 1, 1)
        pooled_output = nn.ReLU()(
            pooled_output)  # (bs * num_choices, dim, 1, 1)
        logits = self.classifier(pooled_output)  # (bs * num_choices, 1, 1, 1)
        logits = logits.squeeze()  # (bs * num_choices)

        reshaped_logits = logits.view(-1, num_choices)  # (bs, num_choices)

        output = (reshaped_logits, ) + outputs[1:]
        loss = None

        return ((loss, ) + output) if loss is not None else output


def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """ Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
    """
    for k in state_dict:
        is_internal_proj = all(substr in k for substr in ['lin', '.weight'])
        is_output_proj = all(substr in k
                             for substr in ['classifier', '.weight'])
        if is_internal_proj or is_output_proj:
            if len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]
