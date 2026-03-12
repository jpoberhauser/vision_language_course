## Module 1 Advanced Vision Transformer Foundations

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Deliverables** |
| --- | --- | --- | --- | --- | --- |
| X | Week 1 | Re-implement basic ViT from scratch (no framework) | Code | ViT paper (Dosovitskiy), lucidrains' vit-pytorch [https://arxiv.org/abs/2010.11929] | Working ViT class + blog on attention in vision |
| X | Week 2 | Compare ViT with Swin, CoAtNet, DeiT + deep dive on multi-head self-attention in spatial domain | Theory/Compare | Papers: Swin, DeiT, CoAtNet; timm repo; "Attention Is All You Need", Annotated Transformer | Comparison table + doc on token vs patch attention |
| 🔲 | Week 3 | Implement masked image modeling (MIM) pretraining | Code | MAE (He et al.), SimMIM | Jupyter notebook training MAE on subset of ImageNet |
| 🔲 | Week 4 | Visualize attention maps from ViT/DeiT | Analysis | vit-explain, einops, Captum | Gallery of attention visualizations for varied inputs |
| 🔲 | Week 5 | Fine-tune pretrained DINO and build feature extractor pipeline | Replication/Utility | DINO paper + repo, HuggingFace AutoModel, ViT-B/16 | Fine-tuned DINO + Python module to extract [CLS] or token embeddings |
| 🔲 | Week 6 | Mini project: Vision head classifier using frozen ViT | Project | ViT from HuggingFace, Scikit-learn | Classifier notebook + 2-minute Loom walk-through |