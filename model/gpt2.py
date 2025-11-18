import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    # CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.modeling_utils import PreTrainedModel, SequenceSummary, ALL_ATTENTION_FUNCTIONS
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from transformers.generation import GenerationMixin

from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    # replace_return_doc,
    )

from transformers import PretrainedConfig
from dataclasses import dataclass

from util.find_root import find_root_smi
import numpy as np
import copy

from util.sampling_strategy import top_k_top_p_sampling


_CHECKPOINT_FOR_DOC = "openai-community/gpt2"
_CONFIG_FOR_DOC = "GPT2Config"

@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    pos_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    # ligand_r_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    bond_ang_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    bond_leng_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    dih_ang_logits: Optional[Tuple[torch.FloatTensor, ...]] = None


class GPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPT2Model`] or a [`TFGPT2Model`]. It is used to
    instantiate a GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPT-2
    [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`GPT2DoubleHeadsModel`].

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import GPT2Config, GPT2Model

    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        # vocab_size=76,
        input_vocab_size=300,
        input_dist_size=300,
        num_resolution=5,
        num_ligand_r=1000,
        num_bond_ang=200,
        num_bond_leng=200,
        num_dih_ang=200,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):  
        self.input_vocab_size = input_vocab_size
        self.input_dist_size = input_dist_size
        self.num_resolution = num_resolution
        self.num_ligand_r=num_ligand_r
        self.num_bond_ang=num_bond_ang
        self.num_bond_leng=num_bond_leng
        self.num_dih_ang=num_dih_ang
        # self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        # logger.error(
        #     "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
        #     "https://www.tensorflow.org/install/ for installation instructions."
        # )
        # raise
        pass
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    # logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        # logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        # logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model

def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights

class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.amp.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None

        is_cross_attention = encoder_hidden_states is not None
        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                # logger.warning_once(
                #     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                #     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                # )
            else:
                # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
                # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
                # not necessarily to eager (if mentionned options are provided).
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            # print(attention_interface, self.config._attn_implementation)
            # assert 0
            # batch_size, num_head, seq_len, _ = query_states.shape
            # attn_mask1 = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool).tril(diagonal=0).to(query_states.device)
            # attn_mask1[:, :, :, :201] = 1
            # attn_mask1 = ~attn_mask1
            # print(attn_mask1.shape)
            # assert 0
            if 0:
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attn_mask1,
                    head_mask=head_mask,
                    dropout=self.attn_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                    **kwargs,
                )
            if 1:
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    head_mask=head_mask,
                    dropout=self.attn_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                    **kwargs,
                )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    
class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

class GPT2Model(GPT2PreTrainedModel):
    _supports_param_buffer_assignment = False

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.input_vocab_size, self.embed_dim)
        self.wte_relative = nn.Embedding(config.input_vocab_size, self.embed_dim)
        self.position_mapping1 = nn.Embedding(config.input_dist_size, self.embed_dim)
        self.position_mapping2 = nn.Embedding(config.input_dist_size, self.embed_dim)
        self.position_mapping3 = nn.Embedding(config.input_dist_size, self.embed_dim)

        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        ####自己加的，一些embedding，包括分辨率，r theta phi
        self.resolution_emb = nn.Embedding(config.num_resolution, self.embed_dim)
        # self.bond_ang_emb = nn.Embedding(config.num_bond_ang, self.embed_dim)
        # self.bond_leng_emb = nn.Embedding(config.num_bond_leng, self.embed_dim)
        # self.dih_ang_emb = nn.Embedding(config.num_dih_ang, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # self.lm_head = nn.Linear(config.input_vocab_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        warnings.warn(
            "`GPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    # @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_position: Optional[torch.LongTensor] = None,
        resolution: Optional[torch.LongTensor] = None,
        ligand_r1_indices: Optional[torch.LongTensor] = None,
        bond_angles: Optional[torch.LongTensor] = None,
        bond_lengths: Optional[torch.LongTensor] = None,
        dihedral_angles: Optional[torch.LongTensor] = None,
        atom_feat: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        shift: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            type_feat = self.wte(input_ids)
            inputs_embeds=type_feat
        #### 特征注入
        position_embeds = self.wpe(position_ids)

        relative_position = self.wte_relative(ligand_r1_indices)
        hidden_states = type_feat# + position_embeds.to(type_feat.device)
        # print(resolution)
        hidden_states = hidden_states + self.resolution_emb(resolution)

        # relative_feat = (self.bond_ang_emb(bond_angles) + self.bond_leng_emb(bond_lengths) + self.dih_ang_emb(dihedral_angles))
        # hidden_states = hidden_states + relative_feat
        relative_feat = None
        feat_input_position = (self.position_mapping1(input_position[:, :,0:1]) + self.position_mapping2(input_position[:, :,1:2]) + self.position_mapping3(input_position[:, :,2:3])) / 3#.mean(dim=-2)

        if hasattr(self, 'atom_feat_mapping'):
            # atom_feat = self.atom_feat_mapping(atom_feat).mean(dim=2)
            atom_feat = self.atom_feat_mapping(atom_feat).flatten(2)
            atom_feat = self.pocket_feat_mapping(atom_feat)
            
            shift_position_ids_scale = torch.where(input_ids == 282)[1] + 1
            pocket_position_ids = torch.clamp(position_ids.repeat(input_ids.shape[0], 1) - shift_position_ids_scale[:, None], 0, 10000)
            position_emebds_pocket = self.pocket_wpe(pocket_position_ids)

            mask = ((input_ids > 260) & (input_ids < 290)).float()
        
            position_embeds = position_embeds * (1 - mask[:, :, None]) + position_emebds_pocket * mask[:, :, None]
            atom_feat = atom_feat * mask[:, :, None]
            hidden_states = hidden_states + atom_feat #* 10
            
            relative_position = relative_position * (1 - mask[:, :, None]) #+ position_emebds_pocket * mask[:, :, None]

            input_position_pocket = (self.pocket_position_mapping1(input_position[:, :,0:1]) + self.pocket_position_mapping2(input_position[:, :,1:2]) + self.pocket_position_mapping3(input_position[:, :,2:3])) / 3
            input_position = feat_input_position * (1 - mask[:, :, None, None]) + input_position_pocket * mask[:, :,None, None]
            # input_position = input_position_pocket + feat_input_position
        else:
            # position_embeds[:, :201] = 0
            if shift is None:
                relative_position[:, :201] = 0
            else:
                relative_position[:, :shift] = 0
                # print(shift)
                # assert 0
            # relative_position[:, :301] = 0
            input_position = feat_input_position

        hidden_states = hidden_states + input_position[:, :, 0]
        hidden_states = hidden_states + position_embeds.to(type_feat.device) + relative_position.to(type_feat.device)

        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,

            type_feat=type_feat,
            relative_feat=relative_feat
        )

class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.input_vocab_size, bias=False)
        # self.pos_head = nn.Linear(config.n_embd, config.input_dist_size * 3, bias=False)
        self.pos_head = nn.Sequential(nn.Linear(config.n_embd, config.n_embd, bias=False) ,
                                    nn.GELU(),
                                    nn.Linear(config.n_embd, config.input_dist_size * 3, bias=False)
                                    )

        self.bond_leng_head = nn.Linear(config.n_embd, config.num_bond_leng, bias=False)
        self.bond_ang_head = nn.Linear(config.n_embd, config.num_bond_ang, bias=False)

        self.dih_ang_head = nn.Linear(config.n_embd, config.num_dih_ang, bias=False)


        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    # @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_position: Optional[torch.LongTensor] = None,
        resolution: Optional[torch.LongTensor] = None,
        ligand_r1_indices: Optional[torch.LongTensor] = None,
        # ligand_r2_indices: Optional[torch.LongTensor] = None, 
        # ligand_r3_indices: Optional[torch.LongTensor] = None,
        bond_angles: Optional[torch.LongTensor] = None,
        bond_lengths: Optional[torch.LongTensor] = None,
        dihedral_angles: Optional[torch.LongTensor] = None,
        atom_feat: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # shift: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            input_position,
            resolution=resolution,
            ligand_r1_indices=ligand_r1_indices,
            # ligand_r2_indices=ligand_r2_indices,
            # ligand_r3_indices=ligand_r3_indices,
            bond_angles=bond_angles,
            bond_lengths=bond_lengths,
            dihedral_angles=dihedral_angles,
            atom_feat=atom_feat,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        type_feat = transformer_outputs.type_feat#.detach()
        # relative_feat = transformer_outputs[-1]#.detach()

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        causal_indices = [i for i in range(1, hidden_states.shape[1])]
        causal_indices.append(0)
        # print(causal_indices)
        # print(type_feat.shape)
        type_feat = type_feat[:, causal_indices]
        # relative_feat = relative_feat[:, causal_indices]
        
        lm_logits = self.lm_head(hidden_states)
        # ligand_r_logits = self.r_head(hidden_states + type_feat).view(lm_logits.shape[0], lm_logits.shape[1], 3, -1)
        bond_ang_logits = self.bond_ang_head(hidden_states + type_feat)
        bond_leng_logits = self.bond_leng_head(hidden_states + type_feat)
        dih_ang_logits = self.dih_ang_head(hidden_states + type_feat)

        pos_logits = self.pos_head(hidden_states + type_feat).view(lm_logits.shape[0], lm_logits.shape[1], 3, -1)


        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            pos_logits=pos_logits,
            # ligand_r_logits=ligand_r_logits,
            bond_ang_logits=bond_ang_logits,
            bond_leng_logits=bond_leng_logits,
            dih_ang_logits=dih_ang_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @torch.no_grad()
    def predict_pointcloud(self, 
                           input_ids: Optional[torch.LongTensor] = None,
                            input_position: Optional[torch.LongTensor] = None,
                            resolution: Optional[torch.LongTensor] = None,
                            ligand_r1_indices: Optional[torch.LongTensor] = None,
                            ligand_r2_indices: Optional[torch.LongTensor] = None, 
                            ligand_r3_indices: Optional[torch.LongTensor] = None,
                            bond_angles: Optional[torch.LongTensor] = None,
                            bond_lengths: Optional[torch.LongTensor] = None,
                            dihedral_angles: Optional[torch.LongTensor] = None,
                            valid_type_max=256,
                            temperature=0.8,
                            fragmentutil=None,
                            max_len=99,
                            shift=199,
                            max_bond_len=21):
        n1, n2, n3 = [], [], []
        seg_index = 0
        prev_index = -1
        prev_prev_index = -1
        char_index = []
        while input_ids.shape[1] != 199:
            transformer_outputs = self.transformer(
                input_ids,
                input_position,
                resolution=resolution,
                ligand_r1_indices=ligand_r1_indices,
                ligand_r2_indices=ligand_r2_indices,
                ligand_r3_indices=ligand_r3_indices,
                bond_angles=bond_angles,
                bond_lengths=bond_lengths,
                dihedral_angles=dihedral_angles,
            )
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)[:, -1, 290:294]
            lm_logits = torch.softmax(lm_logits, dim=-1)
            predicted_token = torch.multinomial(lm_logits, 1) + 290

            type_feat = self.transformer.wte(predicted_token).repeat(1, hidden_states.shape[1], 1)
            
            bond_ang_logits = self.bond_ang_head(hidden_states + type_feat)
            bond_leng_logits = self.bond_leng_head(hidden_states + type_feat)
            dih_ang_logits = self.dih_ang_head(hidden_states + type_feat)

            bond_ang_logits = torch.softmax(bond_ang_logits[:, :, 999:] / temperature, dim=-1)[:, -1]
            predicted_bond_ang_token = torch.multinomial(bond_ang_logits, 1)[None, :, 0] + 999

            bond_leng_logits = torch.softmax(bond_leng_logits[:, :, 999:] / temperature, dim=-1)[:, -1]
            predicted_bond_leng_token = torch.multinomial(bond_leng_logits, 1)[None, :, 0] + 999

            dih_ang_logits = torch.softmax(dih_ang_logits[:, :, 999:] / temperature, dim=-1)[:, -1]
            predicted_dih_ang_token = torch.multinomial(dih_ang_logits, 1)[None, :, 0] + 999

            relative_feat = ((self.transformer.bond_ang_emb(predicted_bond_ang_token) + self.transformer.bond_leng_emb(predicted_bond_leng_token)) / 2 + \
                                            self.transformer.dih_ang_emb(predicted_dih_ang_token))

            pos_logits = self.pos_head(hidden_states + relative_feat + type_feat)[:, -1].view(3, -1)[:, 2:]
            pos_logits = torch.softmax(pos_logits / temperature, dim=-1) 
            predicted_pos = torch.multinomial(pos_logits, 1)[:,0] + 2

            input_ids = torch.cat([input_ids, predicted_token], dim=-1)
            input_position = torch.cat([input_position, predicted_pos[None,None, :].to(predicted_token.device)], dim=1)

            bond_angles = torch.cat([bond_angles, predicted_bond_ang_token], dim=1)
            bond_lengths = torch.cat([bond_lengths, predicted_bond_leng_token], dim=1)
            dihedral_angles = torch.cat([dihedral_angles, predicted_dih_ang_token], dim=1)
        return input_position


    @torch.no_grad()
    def predict(self, 
                input_ids: Optional[torch.LongTensor] = None,
                input_position: Optional[torch.LongTensor] = None,
                resolution: Optional[torch.LongTensor] = None,
                ligand_r1_indices: Optional[torch.LongTensor] = None,
                # ligand_r2_indices: Optional[torch.LongTensor] = None, 
                # ligand_r3_indices: Optional[torch.LongTensor] = None,
                bond_angles: Optional[torch.LongTensor] = None,
                bond_lengths: Optional[torch.LongTensor] = None,
                dihedral_angles: Optional[torch.LongTensor] = None,
                atom_feat: Optional[torch.LongTensor] = None,
                valid_type_max=256,
                temperature=0.7, #原来是1.2
                fragmentutil=None,
                max_len=180,
                shift=201,
                max_bond_len=21,
                begin=0,
                mol_code=None):
        n1, n2, n3 = [], [], []
        seg_index = 0
        prev_index = -1
        prev_prev_index = -1
        char_index = []

        # temp1, temp2, temp3 = 0.1
        # v_atom_cnt = 0
        for _ in range(begin, max_len):
            transformer_outputs = self.transformer(
                input_ids,
                input_position,
                resolution=resolution,
                ligand_r1_indices=ligand_r1_indices,
                # ligand_r2_indices=ligand_r2_indices,
                # ligand_r3_indices=ligand_r3_indices,
                bond_angles=bond_angles,
                bond_lengths=bond_lengths,
                dihedral_angles=dihedral_angles,
                atom_feat=atom_feat,
                shift=shift
            )
            hidden_states = transformer_outputs[0]

            #20 epoch是0.3
            lm_logits = self.lm_head(hidden_states)[:, -1, :valid_type_max]

            lm_logits[:, 155:158] = -1e9
            lm_logits[:, 159:236] = -1e9
            
            # if _ <= 4:
            #     lm_logits = torch.softmax(lm_logits / 1.2, dim=-1) # 原来是0.5
            # else:
            if mol_code is None:
                lm_logits = torch.softmax(lm_logits / 0.7, dim=-1) # 原来是0.5
                predicted_token = torch.multinomial(lm_logits, 1)
            else:
                predicted_token = mol_code[:, _:_+1]
            # if _ <= 4:
            #     predicted_token = top_k_top_p_sampling(lm_logits, top_k=100, top_p=0.9, temperature=1.)
            #     temperature=1.2
            # else:
            #     predicted_token = top_k_top_p_sampling(lm_logits, top_k=25, top_p=0.9, temperature=.7)
            #     temperature=0.7
            if _ == 0:
                predicted_token = torch.tensor([[1]]).to(predicted_token.device)
            # print(torch.max(lm_logits), fragmentutil.fsmiles_vocab_list[int(predicted_token)])
            ###计算祖先节点
            pre_predicted_token = input_ids[0, shift:].cpu().tolist() + predicted_token[0].cpu().tolist()
            valid_pos = copy.deepcopy(input_position[0, shift:])
            fsmiles_pred_token = [fragmentutil.fsmiles_vocab_list[c].split('_')[0] for c in pre_predicted_token]
            codes = [fragmentutil.atom_vocab_c2i[f] for f in fsmiles_pred_token]
            r1_indices, r2_indices, r3_indices = find_root_smi(codes, fragmentutil)
            cur_r1_indices, cur_r2_indices, cur_r3_indices = r1_indices[-1], r2_indices[-1], r3_indices[-1]
            flag1 = (cur_r1_indices == 0)
            flag2 = (cur_r2_indices == 0)
            flag3 = (cur_r3_indices == 0)

            type_feat = self.transformer.wte(predicted_token).repeat(1, hidden_states.shape[1], 1)
            
            bond_ang_logits = self.bond_ang_head(hidden_states + type_feat)
            bond_leng_logits = self.bond_leng_head(hidden_states + type_feat)
            dih_ang_logits = self.dih_ang_head(hidden_states + type_feat)

            if (155 > predicted_token > 3):
                if flag1:
                    predicted_bond_leng_token = torch.tensor([[181]]).to(dih_ang_logits.device)
                else:
                    bond_leng_logits = torch.softmax(bond_leng_logits[:, :, :13] / temperature, dim=-1)[:, -1]
                    predicted_bond_leng_token = torch.multinomial(bond_leng_logits, 1)[None, :, 0]

                if flag1 or flag2:
                    predicted_bond_ang_token = torch.tensor([[181]]).to(dih_ang_logits.device)
                else:
                    bond_ang_logits = torch.softmax(bond_ang_logits[:, :, :19] / temperature, dim=-1)[:, -1]
                    predicted_bond_ang_token = torch.multinomial(bond_ang_logits, 1)[None, :, 0]

                if flag1 or flag2 or flag3:
                    predicted_dih_ang_token = torch.tensor([[181]]).to(dih_ang_logits.device)
                else:
                    dih_ang_logits = torch.softmax(dih_ang_logits[:, :, :19] / temperature, dim=-1)[:, -1]
                    predicted_dih_ang_token = torch.multinomial(dih_ang_logits, 1)[None, :, 0]
            else:
                predicted_bond_ang_token = torch.tensor([[181]]).to(dih_ang_logits.device)
                predicted_bond_leng_token = torch.tensor([[181]]).to(dih_ang_logits.device)
                predicted_dih_ang_token = torch.tensor([[181]]).to(dih_ang_logits.device)

            # relative_feat = (self.transformer.bond_ang_emb(predicted_bond_ang_token) + self.transformer.bond_leng_emb(predicted_bond_leng_token) + self.transformer.dih_ang_emb(predicted_dih_ang_token))

            pos_logits = self.pos_head(hidden_states + type_feat)[:, -1].view(3, -1)
            pos_logits = pos_logits[:, :281]
            # if predicted_token == 3:
            #     seg_index = _
            #     prev_prev_index = prev_index

            # if predicted_token == 255 or predicted_token == 254:
            #     k = _ - 1
            #     while not (155 >  input_ids[0, shift+k]> 3):
            #         k -= 1
            #     prev_index = k
            # print(prev_index, prev_prev_index, predicted_token)
            ####相对坐标修饰
            
            if  _ > 1 and  (155 > predicted_token > 3) and (not flag1):
                # v_atom_cnt += 1
                # pre_predicted_token = input_ids[0, shift:].cpu().tolist() + predicted_token[0].cpu().tolist()
                # valid_pos = copy.deepcopy(input_position[0, shift:])
                # fsmiles_pred_token = [fragmentutil.fsmiles_vocab_list[c].split('_')[0] for c in pre_predicted_token]
                # codes = [fragmentutil.atom_vocab_c2i[f] for f in fsmiles_pred_token]

                # r1_indices, r2_indices, r3_indices = find_root_smi(codes, fragmentutil)
                # cur_r1_indices, cur_r2_indices, cur_r3_indices = r1_indices[-1], r2_indices[-1], r3_indices[-1]
                n1.append(cur_r1_indices)
                n2.append(cur_r2_indices)
                n3.append(cur_r3_indices)
                cur_r1_pos, cur_r2_pos, cur_r3_pos =  valid_pos[cur_r1_indices], valid_pos[cur_r2_indices], valid_pos[cur_r3_indices]
                predicted_bond = fragmentutil.inversed_map[float(predicted_bond_leng_token[0,0])]
                # try:
                basic_index_x = torch.arange(max(cur_r1_pos[0] - predicted_bond - 1, 1), min(cur_r1_pos[0] + predicted_bond + 2, 281), 1).to(cur_r1_pos.device)
                basic_index_y = torch.arange(max(cur_r1_pos[1] - predicted_bond - 1, 1), min(cur_r1_pos[1] + predicted_bond + 2, 281), 1).to(cur_r1_pos.device)
                basic_index_z = torch.arange(max(cur_r1_pos[2] - predicted_bond - 1, 1), min(cur_r1_pos[2] + predicted_bond + 2, 281), 1).to(cur_r1_pos.device)
                # except:
                #     print(cur_r1_pos, cur_r1_indices)
                #     print([fragmentutil.fsmiles_vocab_list[c].split('_')[0] for c in pre_predicted_token])
                #     assert 0
                grid_x, grid_y, grid_z = torch.meshgrid(basic_index_x, basic_index_y, basic_index_z, indexing='ij')
                possible_indices = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

                cur_r1_pos = cur_r1_pos[None].repeat(possible_indices.shape[0], 1)#.cpu().numpy()
                cur_r2_pos = cur_r2_pos[None].repeat(possible_indices.shape[0], 1)#.cpu().numpy()
                cur_r3_pos = cur_r3_pos[None].repeat(possible_indices.shape[0], 1)#.cpu().numpy()
                possible_indices_npy = possible_indices#.cpu().numpy()

                
                ###寻找之前的全部原子坐标
                # valid_pos[char_index] = 999
                # valid_pos = valid_pos[:seg_index+1]
                # if cur_r1_indices != 0:
                #     valid_pos = torch.cat([valid_pos[:cur_r1_indices], valid_pos[cur_r1_indices+1:]], dim=0)
                # dist = torch.sqrt(torch.sum(torch.square(possible_indices[:, None] - valid_pos), dim=-1))
                # dist = torch.min(dist, dim=-1)[0]
                # dist_mask = (dist > 20)

                candidate_bond_lengths = fragmentutil.torch_bond_length(cur_r1_pos, possible_indices_npy)
                # candidate_bond_lengths = np.rint(candidate_bond_lengths).astype("int")
                candidate_bond_lengths = torch.round(candidate_bond_lengths).to(torch.int)

                candidate_bond_angles = fragmentutil.torch_bond_angle(cur_r1_pos, cur_r2_pos, possible_indices_npy)
                # candidate_bond_angles = np.clip(np.rint(candidate_bond_angles).astype("int"), a_min=0, a_max=180)
                candidate_bond_angles = torch.clamp(torch.round(candidate_bond_angles).to(torch.int), 0, 180)

                candidate_dihedral_angles = fragmentutil.torch_dihedral_angle(cur_r1_pos, cur_r2_pos, cur_r3_pos, possible_indices_npy)
                candidate_dihedral_angles = torch.clamp(torch.round(candidate_dihedral_angles).to(torch.int), 0, 180)

                # candidate_bond_lengths, candidate_bond_angles, candidate_dihedral_angles = torch.tensor(candidate_bond_lengths).to(possible_indices.device), torch.tensor(candidate_bond_angles).to(possible_indices.device), torch.tensor(candidate_dihedral_angles).to(possible_indices.device)
                mask = torch.ones_like(candidate_bond_lengths)
                # candidate_bond_lengths, candidate_bond_angles, candidate_dihedral_angles = torch.tensor(candidate_bond_lengths).to(possible_indices.device), torch.tensor(candidate_bond_angles).to(possible_indices.device), torch.tensor(candidate_dihedral_angles).to(possible_indices.device)
                # mask = torch.ones_like(candidate_bond_lengths)
                if not (flag1):
                    mask = mask * ((max(predicted_bond - 1, 0)  <= candidate_bond_lengths) &  (candidate_bond_lengths <= min(predicted_bond + 1, 21))).float()
                
                # if _ > 2 and (cur_r2_indices != cur_r1_indices) and (cur_r2_indices!=0):
                if not ( flag1 or flag2):
                    mask = mask * ((max(predicted_bond_ang_token[0,0] * 10 - 10, 0)  <= candidate_bond_angles) &  (candidate_bond_angles <= min(predicted_bond_ang_token[0,0] * 10 + 20, 180))).float()
                    
                # if _ > 3 and (cur_r2_indices != 0) and (cur_r3_indices != 0):
                if not ( flag1 or flag2 or flag3):
                    mask = mask * ((max(predicted_dih_ang_token[0,0] * 10 - 10, 0)  <= candidate_dihedral_angles) & (candidate_dihedral_angles <= min(predicted_dih_ang_token[0,0] * 10 + 20, 180))).float()
                    # print(torch.sort(candidate_dihedral_angles)[0][10:20])
                    # assert 0
                # add_mask = mask * dist_mask
                # if add_mask.sum() != 0:
                #     mask = add_mask
                if mask.sum() != 0:
                # if False:
                    # print(pos_logits[0, grid_x.flatten()].shape)
                    x_prob = pos_logits[0, grid_x.flatten()]
                    y_prob = pos_logits[1, grid_y.flatten()]
                    z_prob = pos_logits[2, grid_z.flatten()]
                    # print(torch.max(x_prob), torch.max(y_prob), torch.max(z_prob))
                    x_prob[x_prob < 0] = 0
                    y_prob[y_prob < 0] = 0
                    z_prob[z_prob < 0] = 0
                    joint_logits = x_prob * y_prob * z_prob
                    # pos_logits = torch.softmax(pos_logits, dim=-1)

                    # joint_logits = pos_logits[0, grid_x.flatten()] * pos_logits[1, grid_y.flatten()] * pos_logits[2, grid_z.flatten()]
                    joint_logits = joint_logits * mask 
                    # if v_atom_cnt <= 2:
                    #     joint_logits = torch.softmax(joint_logits/5, dim=0)
                    # else:
                    joint_logits = torch.softmax(joint_logits/temperature, dim=0)
                    # joint_logits = joint_logits / torch.sum(joint_logits + 1e-9)
                    if torch.sum(joint_logits) == 0:
                        joint_logits[joint_logits == 0] = 0.1
                        print('use joint logits')
                    predicted_pos_index = torch.multinomial(joint_logits, 1)
                    # except:
                    #     print(torch.sum(joint_logits))
                    #     assert 0
                    predicted_pos = possible_indices[predicted_pos_index][0]
                else:
                    pos_logits = torch.softmax(pos_logits / temperature, dim=-1)
                    predicted_pos = torch.multinomial(pos_logits, 1)[:,0]
                    print(_, predicted_token, predicted_dih_ang_token, predicted_bond_ang_token, predicted_bond)
                

                modify_bond_lengths = fragmentutil.torch_bond_length(cur_r1_pos, predicted_pos[None, :])
                modify_bond_angles = fragmentutil.torch_bond_angle(cur_r1_pos, cur_r2_pos, predicted_pos[None, :])
                modify_dihedral_angles = fragmentutil.torch_dihedral_angle(cur_r1_pos, cur_r2_pos, cur_r3_pos, predicted_pos[None, :])
                    
            else:
                pos_logits = torch.softmax(pos_logits / temperature, dim=-1)
                predicted_pos = torch.multinomial(pos_logits, 1)[:,0]
                
                n1.append(0)
                n2.append(0)
                n3.append(0)


            if  not (155 > predicted_token > 3):
                char_index.append(_)
                predicted_pos = torch.tensor([299, 299, 299]).to(pos_logits.device)
                

            
            input_ids = torch.cat([input_ids, predicted_token], dim=-1)
            input_position = torch.cat([input_position, predicted_pos[None,None, :].to(predicted_token.device)], dim=1)

            bond_angles = torch.cat([bond_angles, predicted_bond_ang_token], dim=1)
            bond_lengths = torch.cat([bond_lengths, predicted_bond_leng_token], dim=1)
            dihedral_angles = torch.cat([dihedral_angles, predicted_dih_ang_token], dim=1)

            if ligand_r1_indices is not None:
                if (155 > predicted_token > 3):
                    # print(ligand_r1_indices.shape, cur_r1_indices.shape)
                    ligand_r1_indices = torch.cat([ligand_r1_indices, torch.tensor([cur_r1_indices])[None, :].to(self.device)], dim=1)
                else:
                    ligand_r1_indices = torch.cat([ligand_r1_indices, torch.tensor([101])[None, :].to(self.device)], dim=1)

            if atom_feat is not None:
                atom_feat =  torch.cat([atom_feat, torch.LongTensor([3, 7, 11, 15, 19, 23, 27, 31, 61])[None, None, :].to(self.device)], dim=1)
                # atom_feat =  torch.cat([atom_feat, torch.LongTensor([-1] * 9)[None, None, :].to(self.device)], dim=1)

            if predicted_token == 2:
                break


        return input_ids, input_position, bond_angles, bond_lengths, dihedral_angles, n1, n2, n3


    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
