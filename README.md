# Intro to Visual-Language Modeling

I couldn't find a good, hands-on course on this topic. This set of notebooks and exercises are a walkthrough, starting from vanilla ViT, going through image-text contrastive learning, and ending with a research project to bring everything together. 

* Every module has the exercises I found useful to understand the models, along with training scripts on a small dataset that can easily run on a small GPU. 

* Every module has an exercise notebook that walks through steps and has place-holders for your own implementation of the code. My solutions are also included if we need to reference. 

* In the first module I walk through vanilla ViT from scratch, add some tricks like window-shifted attention, and work on self-supervised masked image modeling. Everything from scratch, and then we dig into other implementations from `timm` and `transformers` to see how the pros do it. 





## Module 1 Advanced Vision Transformer Foundations

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Deliverables** |
| --- | --- | --- | --- | --- | --- |
| X | Week 1 | Re-implement basic ViT from scratch (no framework) | Code | ViT paper (Dosovitskiy), lucidrains' vit-pytorch [https://arxiv.org/abs/2010.11929] | Working ViT class + blog on attention in vision |
| 🔲 | Week 2 | Compare ViT with Swin, CoAtNet, DeiT + deep dive on multi-head self-attention in spatial domain | Theory/Compare | Papers: Swin, DeiT, CoAtNet; timm repo; "Attention Is All You Need", Annotated Transformer | Comparison table + doc on token vs patch attention |
| 🔲 | Week 3 | Implement masked image modeling (MIM) pretraining | Code | MAE (He et al.), SimMIM | Jupyter notebook training MAE on subset of ImageNet |
| 🔲 | Week 4 | Visualize attention maps from ViT/DeiT | Analysis | vit-explain, einops, Captum | Gallery of attention visualizations for varied inputs |
| 🔲 | Week 5 | Fine-tune pretrained DINO and build feature extractor pipeline | Replication/Utility | DINO paper + repo, HuggingFace AutoModel, ViT-B/16 | Fine-tuned DINO + Python module to extract [CLS] or token embeddings |
| 🔲 | Week 6 | Mini project: Vision head classifier using frozen ViT | Project | ViT from HuggingFace, Scikit-learn | Classifier notebook + 2-minute Loom walk-through |


## Module 2 Vision + Language Pretraining & Integration

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Deliverables** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 1 | Reproduce CLIP (image-text contrastive training) | Code | OpenCLIP repo, CLIP paper | PyTorch script + training logs on custom pairs |
| 🔲 | Week 2 | Create image-caption dataset (10k+ pairs) + vision prompt QA pairs | Data | LAION viewer, COCO, local annotations, GPT-based auto-captioning | JSONL dataset + 1k+ JSON prompts for VQA, uploaded to HuggingFace Hub |
| 🔲 | Week 3 | Implement visual encoder + LLM head for BLIP-style model | Model Dev | BLIP2 paper, transformers | Working BLIP-style architecture on your data |
| 🔲 | Week 4 | Fine-tune pretrained LLaVA model on niche domain | Fine-tuning | LLaVA repo, visual instruction tuning | Interactive VQA demo on your custom image set |
| 🔲 | Week 5 | Study positional embeddings in vision-language transformers | Theory | "Sinusoidal Encoding Explained", Flamingo paper | Blog: "Why positional encodings matter in V+L" |
| 🔲 | Week 6 | Evaluate model on retrieval/captioning metrics + ablate cross-attention | Eval/Research | pycocoevalcap, Recall@K, Flamingo, BLIP2, LLaVA | Metric summary + report on attention pattern differences |
| 🔲 | Week 7 | Train ViT + Qwen3 model using LoRA or QLoRA | Scaling | HuggingFace PEFT, bitsandbytes, Qwen3 | GPU-efficient model fine-tuned with <10GB |

## Module 3 Research Project & Benchmarking

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Deliverables** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 1 | Identify research niche (e.g. behavioral analysis, medical, robotics) | Scoping | arXiv-sanity, your own data | One-pager problem definition |
| 🔲 | Week 2 | Design data pipeline for multimodal dataset | Infra | FiftyOne, decord, ffmpeg | Git repo: dataloader, formatter, uploader |
| 🔲 | Week 3 | Define baseline (OpenCLIP/BLIP2 inference) | Baseline | HuggingFace Transformers | Benchmark accuracy on custom test set |
| 🔲 | Week 4 | Explore image tokenization: VQ-VAE, visual tokens in modern VLMs | Theory/Code | VQ-VAE paper, Chameleon, Emu | Notebook implementing VQ-VAE + blog on image tokenization in VLMs |
| 🔲 | Week 5 | Add temporal or video-level extension (e.g. VideoMAE + BLIP2) | Research | VideoMAE, Flamingo | Model training logs + comparison with frame-level |
| 🔲 | Week 6 | Perform ablation (frozen vs trained encoder) + build Streamlit dashboard | Experiment/Viz | Torch hooks, wandb, Streamlit, seaborn, t-SNE | Training chart comparison + interactive demo |
| 🔲 | Week 7 | Draft paper-style report or detailed blog post | Writing | LaTeX or markdown | Full write-up with visuals + open-source GitHub repo |


### Open-source Multi-modal models

We can take a look at the fully open-sourced [Molmo2-Models](https://github.com/allenai/molmo2)