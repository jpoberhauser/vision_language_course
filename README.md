# Intro to Visual-Language Modeling

I couldn't find a good, hands-on course on this topic. This set of notebooks and exercises are a walkthrough, starting from vanilla ViT, going through image-text contrastive learning, and ending with a research project to bring everything together. 

* Every module has the exercises I found useful to understand the models, along with training scripts on a small dataset that can easily run on a small GPU. 

* Every module has an exercise notebook that walks through steps and has place-holders for your own implementation of the code. My solutions are also included if we need to reference. 

* In the first module I walk through vanilla ViT from scratch, add some tricks like window-shifted attention, and work on self-supervised masked image modeling. Everything from scratch, and then we dig into other implementations from `timm` and `transformers` to see how the pros do it. 





## Module 1 Advanced Vision Transformer Foundations

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| X | Week 1 | Re-implement basic ViT from scratch (no framework) | Code | ViT paper (Dosovitskiy), lucidrains' vit-pytorch [https://arxiv.org/abs/2010.11929] | [001_vit_from_scratch.ipynb](module1_vision_transformer_foundations/001_vit_from_scratch.ipynb) |
| X | Week 2 | Compare ViT with Swin, CoAtNet, DeiT + deep dive on multi-head self-attention in spatial domain | Theory/Compare | Papers: Swin, DeiT, CoAtNet; timm repo; "Attention Is All You Need", Annotated Transformer | [002_compare_ViTs.ipynb](module1_vision_transformer_foundations/002_compare_ViTs.ipynb), [002_CoAtNet.ipynb](module1_vision_transformer_foundations/002_CoAtNet.ipynb) |
| X | Week 3 | Implement masked image modeling (MIM) pretraining | Code | MAE (He et al.), SimMIM | [003_MAE.ipynb](module1_vision_transformer_foundations/003_MAE.ipynb) |
| 🔲 | Week 4 | Visualize attention maps, frozen feature extraction, linear probing. Compare DINOv2 vs CLIP attention. Survey: DINOv1→v2→v3, BEiT, JEPA, V-JEPAv2 | Analysis/Code | DINOv2, DINO, Captum, vit-explain, BEiT, JEPA, V-JEPAv2 papers | |



## Module 2 Vision + Language Pretraining & Integration

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 1 | Build a text encoder from scratch — tokenization, text transformer, sentence embeddings | Code | "Attention Is All You Need", HuggingFace tokenizers, Annotated Transformer | |
| 🔲 | Week 2 | Reproduce CLIP (image-text contrastive training) | Code | OpenCLIP repo, CLIP paper | |
| 🔲 | Week 3 | Create image-caption dataset (10k+ pairs) + vision prompt QA pairs | Data | LAION viewer, COCO, local annotations, GPT-based auto-captioning | |
| 🔲 | Week 4 | Implement visual encoder + LLM head for BLIP-style model | Model Dev | BLIP2 paper, transformers | |
| 🔲 | Week 5 | Fine-tune pretrained LLaVA model on niche domain | Fine-tuning | LLaVA repo, visual instruction tuning | |
| 🔲 | Week 6 | Study positional embeddings in vision-language transformers | Theory | "Sinusoidal Encoding Explained", Flamingo paper | |
| 🔲 | Week 7 | Evaluate model on retrieval/captioning metrics + ablate cross-attention | Eval/Research | pycocoevalcap, Recall@K, Flamingo, BLIP2, LLaVA | |

## Module 3 Modern VLMs & Research Project

* The main idea of Module 3 is that now that we have built everything from scratch to move on to: understand architecture → run inference → fine-tune → image tokenization theory → compare VLMs → ablate → write up

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 1 | Deep dive into modern VLM architectures — how vision encoder + connector + LLM fit together | Theory/Analysis | Raschka's "Understanding Multimodal LLMs", Qwen2.5-VL, Qwen3 technical reports | |
| 🔲 | Week 2 | Run Qwen2.5-VL / Qwen3 inference — experiment with VQA, captioning, grounding, OCR | Code | Qwen2.5-VL, Qwen3, HuggingFace Transformers | |
| 🔲 | Week 3 | Fine-tune Qwen-VL with LoRA/QLoRA on a custom domain | Fine-tuning | HuggingFace PEFT, bitsandbytes, Qwen-VL | |
| 🔲 | Week 4 | Explore image tokenization: VQ-VAE, visual tokens in modern VLMs | Theory/Code | VQ-VAE paper, Chameleon, Emu | |
| 🔲 | Week 5 | Compare VLM architectures: Qwen-VL vs LLaVA vs Molmo — design differences and tradeoffs | Theory/Compare | Qwen-VL, LLaVA, Molmo papers + repos | |
| 🔲 | Week 6 | Ablations on fine-tuned model + build Streamlit dashboard | Experiment/Viz | Torch hooks, wandb, Streamlit, seaborn, t-SNE | |
| 🔲 | Week 7 | Draft paper-style report or detailed blog post | Writing | LaTeX or markdown | |

### Suggested reading/watching


#### [Umar Jamil's Coding a Multimodal Vision Language Model from Scratch](https://www.youtube.com/watch?v=vAmKB7iPkWw)

#### [Understanding MultiModal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)


#### [How AI Taught Itself to See](https://www.youtube.com/watch?v=oGTasd3cliM)


### Open-source Multi-modal models

We can take a look at the fully open-sourced [Molmo2-Models](https://github.com/allenai/molmo2)

Also, a competitive fully open source vision encoder in (Franca)[https://github.com/valeoai/Franca]

## Module 4 (Optional) Building Vision Solutions with Transformers

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 1 | DETR | Scoping | build a Transformer Detector from scratch | |
| 🔲 | Week 2 | Building on top of DETR | LW-DETR, RF-DETR |
| 🔲 | Week 3 | Segmentation and Pose | MaskDino (segmentation) DETR-Pose (pose estimation) |
| 🔲 | Week 4 | MOTracking |MOTR, MOTRv2 , TrackFormer, SAM|