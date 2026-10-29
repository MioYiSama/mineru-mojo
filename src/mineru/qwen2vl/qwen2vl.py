"""Qwen2-VL text decoder (Qwen2 with M-RoPE) in ModuleV3.

Mirrors HF ``Qwen2VLTextModel`` + ``Qwen2VLForConditionalGeneration``'s head:

- Decoder blocks are Llama-shaped (pre-RMSNorm, SwiGLU MLP, GQA), so the
  block and MLP come from ``llama3_modulev3``.
- ``Qwen2VLAttention`` projects Q/K/V with bias (``o_proj`` has none) and
  applies multimodal RoPE: frequency ``i`` reads the temporal, height, or
  width position of the token according to ``mrope_section``, with the
  ``rotate_half`` (non-interleaved) layout. Text tokens carry the same
  position on all three axes, which reduces to 1D RoPE.
- Image embeddings from the vision tower replace the image placeholder
  token embeddings before the first block.
- The LM head is tied to ``embed_tokens``.
"""

from __future__ import annotations

from max import tree
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.attention import AttentionWithRope
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import ops
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.llama3_modulev3.layers.rotary_embedding import (
    Llama3RotaryEmbedding,
)
from max.pipelines.architectures.llama3_modulev3.layers.transformer_block import (
    LlamaTransformerBlock,
)
from max.pipelines.architectures.llama3_modulev3.model_config import (
    Llama3Config,
)
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings

from .model_config import Qwen2VLConfig


class Qwen2VLAttention(AttentionWithRope):
    """GQA attention with Q/K/V bias and multimodal RoPE (HF ``Qwen2VLAttention``).

    Reuses ``AttentionWithRope``'s projections; ``forward`` passes the
    per-token 3-axis ``position_ids`` and ``mrope_section`` to the fused
    rope-and-cache-store kernel.
    """

    def __init__(self, *, mrope_section: list[int], **kwargs) -> None:
        super().__init__(has_bias=True, **kwargs)
        self.mrope_section = mrope_section

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        *,
        input_row_offsets: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        qkv = self.qkv_proj(x)
        freqs_cis = F.cast(self.rope.freqs_cis, qkv.dtype).to(qkv.device)
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=self.rope.interleaved,
            position_ids=position_ids,
            mrope_section=self.mrope_section,
        ).rebind_mapping(qkv.mapping)
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        ).rebind_mapping(xq.mapping)
        attn_out = F.reshape(attn_out, shape=[total_seq_len, self.q_weight_dim])
        return self.o_proj(attn_out)


class Qwen2VLTextModel(Module[..., tuple[Tensor, ...]]):
    """Qwen2 decoder stack, final norm, and tied LM head."""

    def __init__(self, config: Llama3Config, mrope_section: list[int]) -> None:
        super().__init__()
        if not config.tie_word_embeddings:
            raise ValueError("Qwen2-VL port expects tie_word_embeddings=True")
        assert config.rms_norm_eps is not None

        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=config.kv_params.head_dim,
            interleaved=config.interleaved_rope_weights,
        )
        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.layers = ModuleList(
            LlamaTransformerBlock(
                attention=Qwen2VLAttention(
                    mrope_section=mrope_section,
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    scale=config.attention_multiplier,
                ),
                mlp=MLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                ),
                input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                post_attention_layernorm=RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ),
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.return_logits = config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        return F.cast(self.norm(h) @ self.embed_tokens.weight.T, DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, ...]:
        h = F_merge_multimodal_embeddings(
            self.embed_tokens(tokens), image_embeddings, image_token_indices
        )
        for idx, layer in enumerate(self.layers):
            layer_idx = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
                position_ids=position_ids,
            )

        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(last_h)

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            last_indices = F.reshape(offsets, shape=(-1,))
            logits = self._compute_logits(F.gather(h, last_indices, axis=0))
            logit_offsets = ops.range(
                0,
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
            return last_logits, logits, logit_offsets
        if self.return_logits == ReturnLogits.ALL:
            return last_logits, self._compute_logits(h), input_row_offsets
        return (last_logits,)


class Qwen2VLLanguageModel(Module[..., tuple[Tensor, ...]]):
    """Compiled language root: unflattens the KV cache, runs the decoder.

    Inputs, in order: ``tokens``, ``return_n_logits``, ``input_row_offsets``,
    ``image_embeddings`` (``[n_image_tokens, hidden]``),
    ``image_token_indices`` (``[n_image_tokens]`` int32 rows into
    ``tokens``), ``position_ids`` (``[3, total_seq_len]`` uint32 M-RoPE
    positions), then the flattened KV cache inputs.
    """

    def __init__(self, config: Qwen2VLConfig, kv_params: KVCacheParamInterface) -> None:
        super().__init__()
        self.language_model = Qwen2VLTextModel(config.llm_config, config.mrope_section)
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        position_ids: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        kv_inputs = iter(x._graph_value for x in variadic_args)
        kv_collections = tree.leaves(
            self.kv_params.unflatten_kv_inputs(kv_inputs),
            leaf=KVCacheInputsPerDevice,
        )
        return self.language_model(
            tokens,
            kv_collections[0],
            return_n_logits,
            input_row_offsets,
            image_embeddings,
            image_token_indices,
            position_ids,
        )
