from typing import Optional, Tuple

import torch
import torch.nn as nn


class SiglipVisionConfig:
    """Config to easily change sizes and configuration for vision encoders."""

    def __init__(
        self,
        hidden_size=768,  # size of embedding vector
        intermediate_size=3072,  # size of linear layer in feed forward
        num_hidden_layers=12,  # num hidden in transformer
        num_attention_heads=12,  # num heads
        num_channels=3,
        image_size=224,
        patch_size=16,  # size of the patches. each patch will be 16x16
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        # we can use a single cls_token of the entire image, or we can use a list of embeddings that represent each patch.
        num_image_tokens: int = None,  # how many output embeddings we expect. (how many image embeddings we have for each image.)
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    ## equivalent to PatchEmbedding in vit.py
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,  # no overlap
            padding='valid',  # indicates no padding necessary
        )

        self.num_patches = int((self.image_size**2) / (self.patch_size**2))
        self.num_positions = self.num_patches
        # In our vit.py we used self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, num_hiddens))
        # these do the same thing, they create a learnable matrix of positional vectors.
        # under the hood, nn.Embedding is a wrapper for nn.Parameter with a lookup operation.
        # nn.Embedding is slightly more flexible,
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # remember in vanilla transformer for text, we use sinusoidal positional embeddings
        # in this vision encoder, we let it learn the positional embedding.
        # this is a vector the size of the patches.
        self.register_buffer(
            'position_ids',
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )  # this pre-creates an index tensor and stored it on the module so it moves to the right device.
        # without it, we'd have to run   position_ids = torch.arange(num_positions, device=x.device) on the forward pass.

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape  # [Batch_Size, Channels, H, W]
        # convolve the patch_size kernel over the image, no overlap
        # the output of the conv will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        patch_emeds = self.patch_embedding(pixel_values)

        embeddings = patch_emeds.flatten(2)  # flatten to turn from grid to a flat vector.
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    # no causal mask like language models.
    # start with a sequence of patches, each represented by a 1x1024 vector.
    # each patch is from a group of pixels.
    # the resulting attention mask now has information about the patches relationship to other patches.
    # in language, we contextualize the token against all the tokens that came _before_ it. slight difference than in vit.
    # we use causal mask for next-token prediction task. transformer lets us do that in parallel. hence the causal mask.
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5  # 1/√d_k divisor as a multiplier for efficiency.
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # parameter matrices that transform input sequences, shape stays the same
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)  # W_o matrix

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # [Batch_Size, Num_Patches, Embed_Dim] you can think of num_patches as the sequence length.
        batch_size, seq_len, _ = hidden_states.size()
        # [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # head splitting here
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # the view splits the last dimension into
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] 1024/8 for example will be 128. So each head receives 128
        # the transpose then changes to [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        # why transpose? [1, 4, 8, 128] -> [1, 8, 4, 128] for example. So think of it as one big matrix, with 8 smaller matrices, one going into each head.
        # once its transposed, we can parallelize better. each head has a sequence of 4 tokens x 128. each head can be treated independently basically.
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        # in our vit.py attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # calculate the attention using Q * K^T /sqrt(d_k). These will now be [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f'Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)} but is'
                f'{attn_weights.size()}'
            )
        # apply the softmax row-wise: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # last part of the formula
        # compute a weighted sum of softmaxed Q*K. Q*K should be 0 above the diagonal. So this is 'causal'
        # so Q * K tells us how much each token will contribute to the final embedding, and by how much.
        # each head is only looking at a part of the embeddings, and it will learn different attention scores
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # Transpose: [Batch, Num_Heads, Num_Patches, Head_Dim] -> [Batch, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # this reshape is the concat operation!
        # Reshape: [Batch, Num_Patches, Num_Heads, Head_Dim] -> [Batch, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        # self.out_proj applies the final linear projection W_O on the concatenated result, matching the standard formula:
        # Output = Concat(head_1, ..., head_h) · W_O.
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    # this is the same as TransformerEncoder in our vit.py
    # sequence to sequence model here
    # input is embeddings of patches flattened. with attention mechanism, we contextualize these embeddings.
    # each layer will have layernorm --> MHA --> LayerNorm --> FFN.
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        residual = hidden_states  # save for later
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        # this prepares the sequence of patches for the next layer too, a tiny transform + a non-linearity
        hidden_states = self.mlp(hidden_states)  # independent transforms now, as opposed to attention.
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)  # equivalent to PatchEmbedding in our vit.py
        self.encoder = SiglipEncoder(config)  # equivalent to TransformerEncoder in our vit.py
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, H, W] -> [Batch_Size, Num_Patches, Embed_Size]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, H, W] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
