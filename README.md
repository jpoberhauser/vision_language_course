
## Module 1 Advanced Vision Transformer Foundations

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Deliverables** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 1 | Re-implement basic ViT from scratch (no framework) | Code | ViT paper (Dosovitskiy), lucidrains’ vit-pytorch [https://arxiv.org/abs/2010.11929] | Working ViT class + blog on attention in vision |
| 🔲 | Week 2 | Compare ViT with Swin, CoAtNet, DeiT | Theory/Compare | Papers: Swin, DeiT, CoAtNet; timm repo | Matrix comparison table + summary blog |
| 🔲 | Week 3 | Train ViT and DeiT on CIFAR100 or Food101 | Practice | timm, HuggingFace Datasets, wandb | Model training logs + report on performance tradeoffs |
| 🔲 | Week 4 | Implement masked image modeling (MIM) pretraining | Code | MAE (He et al.), SimMIM | Jupyter notebook training MAE on subset of ImageNet |
| 🔲 | Week 5 | Visualize attention maps from ViT/DeiT | Analysis | vit-explain, einops, Captum | Gallery of attention visualizations for varied inputs |
| 🔲 | Week 6 | Replicate DINO or iBOT for self-supervised vision | Replication | DINO paper + repo, iBOT | Reproduced pretraining on custom dataset |
| 🔲 | Week 7 | Create feature extractor pipeline for vision heads | Utility | HuggingFace AutoModel, ViT-B/16 | Python module to extract [CLS] or token embeddings |
| 🔲 | Week 8 | Deep dive: Multi-head self-attention in spatial domain | Theory | “Attention Is All You Need”, Annotated Transformer | Markdown doc comparing token vs patch attention |
| 🔲 | Week 9 | Read & explain: SAM (Segment Anything Model) | Research | Meta SAM repo + paper | Slide deck summarizing what makes SAM work |
| 🔲 | Week 10 | Mini project: Vision head classifier using frozen ViT | Project | ViT from HuggingFace, Scikit-learn | Classifier notebook + 2-minute Loom walk-through |


## Module 2 Vision + Language Pretraining & Integration

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Deliverables** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 1 | Reproduce CLIP (image-text contrastive training) | Code | OpenCLIP repo, CLIP paper | PyTorch script + training logs on custom pairs |
| 🔲 | Week 2 | Create a small image-caption dataset (10k+ pairs) | Data | LAION viewer, COCO, local annotations | JSONL dataset + upload to HuggingFace Hub |
| 🔲 | Week 3 | Implement visual encoder + LLM head for BLIP-style model | Model Dev | BLIP2 paper, transformers | Working BLIP-style architecture on your data |
| 🔲 | Week 4 | Fine-tune pretrained LLaVA model on niche domain | Fine-tuning | LLaVA repo, visual instruction tuning | Interactive VQA demo on your custom image set |
| 🔲 | Week 5 | Study positional embeddings in vision-language transformers | Theory | “Sinusoidal Encoding Explained”, Flamingo paper | Blog: “Why positional encodings matter in V+L” |
| 🔲 | Week 6 | Evaluate your model on retrieval and captioning metrics | Eval | pycocoevalcap, Recall@K | Metric summary + qualitative example set |
| 🔲 | Week 7 | Add cross-attention layer and ablate model behavior | Research | Flamingo, BLIP2, LLaVA | Report: attention pattern differences |
| 🔲 | Week 8 | Create a “vision prompt” dataset (images + free-form questions) | Dataset | GPT-based auto-captioning or manual tagging | 1k+ JSON prompts for VQA task |
| 🔲 | Week 9 | Train ViT + LLaMA2 model using LoRA or QLoRA | Scaling | HuggingFace PEFT, bitsandbytes | GPU-efficient model fine-tuned with <10GB |
| 🔲 | Week 10 | Write “How to build CLIP from scratch” tutorial | Writing | Your own repo + notebook | Full markdown + visuals + Colab version |

## Module 3 Research Project & Benchmarking

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Deliverables** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 1 | Identify research niche (e.g. behavioral analysis, medical, robotics) | Scoping | arXiv-sanity, your own data | One-pager problem definition |
| 🔲 | Week 2 | Design data pipeline for multimodal dataset | Infra | FiftyOne, decord, ffmpeg | Git repo: dataloader, formatter, uploader |
| 🔲 | Week 3 | Define baseline (OpenCLIP/BLIP2 inference) | Baseline | HuggingFace Transformers | Benchmark accuracy on custom test set |
| 🔲 | Week 4 | Add temporal or video-level extension (e.g. VideoMAE + BLIP2) | Research | VideoMAE, Flamingo | Model training logs + comparison with frame-level |
| 🔲 | Week 5 | Evaluate model using both text and image prompts | Eval | CLIPScore, BLEU, CIDEr | Mixed-prompt eval report |
| 🔲 | Week 6 | Perform ablation: frozen encoder vs trained | Experiment | Torch hooks, wandb | Training chart comparison |
| 🔲 | Week 7 | Build small dashboard to show results (Streamlit) | Visualization | Streamlit, seaborn, t-SNE | Interactive demo: prompt → image or caption |
| 🔲 | Week 8 | Present work to peer or mentor for critique | Review | Slack, local ML group | Slide deck + talk recording |
| 🔲 | Week 9 | Draft paper-style report or arXiv preprint | Writing | NeurIPS template | Full paper draft in LaTeX |
| 🔲 | Week 10 | Prepare for submission to workshop or NeurIPS track | Publish | OpenReview, CMT | Ready-to-submit PDF and GitHub link |