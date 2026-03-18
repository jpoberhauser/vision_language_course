import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, token_ids):
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=49408, embed_dim=512, num_heads=8,
                 num_layers=12, max_seq_len=77, projection_dim=512):
        super().__init__()
        self.embedding = TextEmbedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, projection_dim, bias=False)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = self.blocks(x)
        x = self.ln_final(x)

        # EOS token as sentence representation
        eos_indices = token_ids.argmax(dim=-1)
        sentence_emb = x[torch.arange(x.shape[0]), eos_indices]

        # Project and L2 normalize
        sentence_emb = self.projection(sentence_emb)
        sentence_emb = sentence_emb / sentence_emb.norm(dim=-1, keepdim=True)
        return sentence_emb
