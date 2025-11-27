import torch
import torch.nn as nn

from einops import rearrange

# from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
# from flash_attn.bert_padding import unpad_input, pad_input
import torch.nn.functional as F



def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    手动实现 scaled_dot_product_attention。

    参数:
        query: [batch_size, seq_len_q, d_k]
        key: [batch_size, seq_len_kv, d_k]
        value: [batch_size, seq_len_kv, d_v]
        attn_mask: [batch_size, seq_len_q, seq_len_kv] 或 None
        dropout_p: dropout 概率
        is_causal: 是否启用因果掩码（用于解码器自注意力）

    返回:
        output: [batch_size, seq_len_q, d_v]
        attention_weights: [batch_size, seq_len_q, seq_len_kv]
    """
    d_k = query.size(-1)  # 获取 d_k 维度

    # 计算缩放点积注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # 如果启用因果掩码，生成一个下三角掩码
    if is_causal:
        seq_len_q, seq_len_kv = query.size(-2), key.size(-2)
        causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.to(query.device)  # 确保掩码在正确的设备上
        scores = scores.masked_fill(causal_mask, float('-inf'))  # 将上三角部分设为负无穷

    # 如果提供了 attn_mask，应用它
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))  # 将掩码为 0 的位置设为负无穷

    # 计算注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    # 应用 dropout
    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)

    # 计算加权值
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        batch_size, seqlen, _, nheads, d = qkv.shape
        q, k, v = qkv.unbind(dim=2)

        if key_padding_mask is not None:
            # Apply key padding mask
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            if causal:
                # Create a causal mask
                causal_mask = torch.ones(seqlen, seqlen, device=qkv.device, dtype=torch.bool).tril()
                mask = mask & causal_mask

            # Apply mask to the attention scores
            attn_mask = torch.zeros_like(mask, dtype=q.dtype)
            attn_mask.masked_fill_(~mask, float('-inf'))
        else:
            attn_mask = None

        # Reshape q, k, v for multi-head attention
        q = rearrange(q, 'b s h d -> (b h) s d')
        k = rearrange(k, 'b s h d -> (b h) s d')
        v = rearrange(v, 'b s h d -> (b h) s d')

        # Apply scaled dot-product attention
        output, attn = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0, is_causal=causal
        )

        # Reshape output back to original shape
        output = rearrange(output, '(b h) s d -> b s h d', b=batch_size, h=nheads)

        return output, None, attn