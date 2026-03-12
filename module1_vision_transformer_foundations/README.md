## Module 1 Advanced Vision Transformer Foundations

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Deliverables** |
| --- | --- | --- | --- | --- | --- |
| X | Week 1 | Re-implement basic ViT from scratch (no framework) | Code | ViT paper (Dosovitskiy), lucidrains' vit-pytorch [https://arxiv.org/abs/2010.11929] | Working ViT class + blog on attention in vision |
| X | Week 2 | Compare ViT with Swin, CoAtNet, DeiT + deep dive on multi-head self-attention in spatial domain | Theory/Compare | Papers: Swin, DeiT, CoAtNet; timm repo; "Attention Is All You Need", Annotated Transformer | Comparison table + doc on token vs patch attention |
| 🔲 | Week 3 | Implement masked image modeling (MIM) pretraining | Code | MAE (He et al.), SimMIM | Jupyter notebook training MAE on subset of ImageNet |
| 🔲 | Week 4 | Visualize attention maps from ViT/DeiT | Analysis | vit-explain, einops, Captum | Gallery of attention visualizations for varied inputs |
| 🔲 | Week 5 | Fine-tune pretrained DINO and build feature extractor pipeline | Replication/Utility | DINO paper + repo, HuggingFace AutoModel, ViT-B/16 | Fine-tuned DINO + Python module to extract [CLS] or token embeddings |
| 🔲 | Week 6 | Mini project: Vision head classifier using frozen ViT | Project | ViT from HuggingFace, Scikit-learn | Classifier notebook + 2-minute Loom walk-through |

---

## Notebook Overview

### 001_vit_from_scratch.ipynb
Builds a Vision Transformer from first principles, one component at a time: PatchEmbedding (Conv2d-based), positional embeddings (sinusoidal + learned), AttentionHead, MultiHeadAttention, FeedForward, TransformerBlock, TransformerEncoder, and ClassificationHead. Each component includes shape walkthroughs and visualizations (patch grids, attention maps, sinusoidal encoding heatmaps). Trained on FashionMNIST as a sanity check.

**Intuition built:** How images are tokenized into patches, why positional embeddings are needed (permutation invariance), how self-attention routes information between patches, and how residual connections enable deep stacking. All reusable components are exported to `vit.py`.

**Connection to VLMs:** The ViT encoder is the vision backbone in nearly every modern VLM (CLIP, BLIP2, LLaVA, Qwen-VL). Understanding how patches become token embeddings is essential — in VLMs these same patch embeddings get projected into the language model's input space. The CLS token concept also maps directly to how vision features are pooled before being fed to a language model.

### 002_compare_ViTs.ipynb
Trains the vanilla ViT on CIFAR-100 as a baseline, then implements Swin-style windowed attention (window partitioning, CLS token handling, shifted windows with `torch.roll`) as a modification to MultiHeadAttention. Compares vanilla ViT vs windowed (no shift) vs windowed + shifted on accuracy and training time.

**Intuition built:** Why global attention is expensive and how local windowed attention trades receptive field for efficiency. Why isolated windows fail (stuck at random accuracy) and how shifted windows restore cross-window information flow. Demonstrates that architectural changes matter less at small scale (56x56, 64 patches) where global attention is already cheap.

**Connection to VLMs:** Efficient attention is critical when processing high-resolution images or video frames in VLMs. Swin-style windowed attention is used in many vision encoders that feed into language models. Understanding when local vs global attention is appropriate helps when choosing or designing the vision backbone for a VLM pipeline.

### 002_CoAtNet.ipynb
Implements a hybrid CNN + Transformer architecture: a CNN stem (3 conv layers with BatchNorm and GELU) replaces PatchEmbedding, feeding into the same TransformerEncoder. Compares against vanilla ViT on CIFAR-100. Also inspects timm's CoAtNet architecture to understand MbConvBlocks, depthwise convolutions, SE modules, and relative positional biases.

**Intuition built:** CNNs are good at local features with built-in inductive biases (translation equivariance), transformers are good at global reasoning. Stacking CNNs early and attention late gives the best of both worlds. Also clarifies that 1x1 Conv2d and nn.Linear are mathematically identical — just different data layouts.

**Connection to VLMs:** Many VLM vision encoders are hybrids. Understanding the tradeoff between convolutional inductive biases and attention flexibility helps explain why some VLMs use pure ViT encoders (CLIP) while others use hybrid backbones. The staged resolution reduction (high-res CNN stages → low-res attention stages) is also how video VLMs handle temporal downsampling.

### 003_MAE.ipynb
Implements Masked Autoencoder pretraining: random masking of 75% of patches, encoding only visible patches, inserting learnable mask tokens, decoding with a lightweight transformer to reconstruct pixel values. Loss is MSE computed only on masked patches.

**Intuition built:** Self-supervised pretraining without labels — force the model to learn visual structure by reconstructing missing patches. The encoder learns meaningful features because it must understand spatial relationships, object parts, and textures to predict what's hidden. Contrasts with JEPA (predicts representations, not pixels).

**Connection to VLMs:** MAE-style pretraining produces strong vision encoders that can be paired with language models. The masking concept extends directly to video (VideoMAE masks spacetime tubes) and to multimodal settings where either modality can be masked. Understanding reconstruction-based pretraining vs contrastive pretraining (CLIP) is key to choosing how to train or fine-tune the vision component of a VLM.
