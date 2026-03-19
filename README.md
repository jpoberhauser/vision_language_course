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
| X | Week 4 | Visualize attention maps, frozen feature extraction, linear probing. Compare DINOv2 vs CLIP attention. | Analysis/Code | DINOv2, DINO, CLIP papers | |



## Module 2 Vision + Language Pretraining & Integration

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| X | Week 1 | Build a text encoder from scratch — tokenization, text transformer, sentence embeddings | Code | "Attention Is All You Need", HuggingFace tokenizers, Annotated Transformer | |
| X | Week 2 | Reproduce CLIP (image-text contrastive training) | Code | OpenCLIP repo, CLIP paper | |
| 🔲 | Week 3 | Implement visual encoder + LLM head for BLIP-style model | Model Dev | BLIP2 paper, transformers | |
| 🔲 | Week 4 | Fine-tune pretrained LLaVA model on niche domain | Fine-tuning | LLaVA repo, visual instruction tuning | |
| 🔲 | Week 5 | Study positional embeddings in vision-language transformers | Theory | "Sinusoidal Encoding Explained", Flamingo paper | |
| 🔲 | Week 6 | Evaluate model on retrieval/captioning metrics + ablate cross-attention | Eval/Research | pycocoevalcap, Recall@K, Flamingo, BLIP2, LLaVA | |

## Module 3 Modern VLMs, Alignment & Agentic Systems

* Now that we've built everything from scratch, this module shifts to: understand production VLM architectures → run inference → fine-tune → synthetic data → alignment → agentic systems 

### Phase 1 — Modern VLM Architectures & Inference

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** |
| --- | --- | --- | --- | --- |
| 🔲 | Week 1 | Deep dive into modern VLM architectures — read LLaVA, InternVL2, Qwen2.5-VL, Flamingo. Understand how vision encoder + connector + LLM backbone fit together | Theory | Raschka's "Understanding Multimodal LLMs", LLaVA, InternVL2, Qwen2.5-VL, Flamingo papers |
| 🔲 | Week 2 | Run Qwen2.5-VL inference — experiment with VQA, captioning, grounding, OCR | Code | Qwen2.5-VL, HuggingFace Transformers |
| 🔲 | Week 3 | Compare VLM architectures: Qwen-VL vs LLaVA vs Molmo — unified embedding vs cross-attention, connector design, tradeoffs | Theory/Compare | Qwen-VL, LLaVA, Molmo papers + repos |

### Phase 2 — Fine-Tuning & Image Tokenization

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** |
| --- | --- | --- | --- | --- |
| 🔲 | Week 4 | Fine-tune a VLM (Qwen-VL or LLaVA-1.5) with LoRA/QLoRA on a custom domain | Fine-tuning | LLaMA-Factory, Swift, HuggingFace PEFT, bitsandbytes |
| 🔲 | Week 5 | Explore image tokenization: VQ-VAE, visual tokens in modern VLMs | Theory/Code | VQ-VAE paper, Chameleon, Emu |
| 🔲 | Week 6 | Ablations on fine-tuned model + build Streamlit dashboard | Experiment/Viz | Torch hooks, wandb, Streamlit, seaborn, t-SNE |

### Phase 3 — Data: Curation, Captioning & Synthetic Pipelines

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** |
| --- | --- | --- | --- | --- |
| 🔲 | Week 7 | Create image-caption dataset (10k+ pairs) + vision prompt QA pairs. Study data curation at scale — LAION, DataComp | Data/Theory | LAION viewer, COCO, local annotations, GPT-based auto-captioning, DataComp papers, "Scaling Data-Constrained Language Models" (Muennighoff et al.) |
| 🔲 | Week 8 | Learn data mixture & quality research — optimal ratios, data selection | Theory | DoReMi, "Data Selection for LLMs" papers, data mixing laws |
| 🔲 | Week 9 | Build a synthetic VQA dataset using a VLM as the labeler, fine-tune on it, evaluate | Code/Data | LLM-as-oracle approach, VQA generation pipelines |

### Phase 4 — RL & Alignment for VLMs

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** |
| --- | --- | --- | --- | --- |
| 🔲 | Week 10 | RLHF foundations — InstructGPT, practical PPO/DPO tooling | Theory/Code | InstructGPT paper, TRL library (HuggingFace) |
| 🔲 | Week 11 | Modern alignment methods — DPO, GRPO (DeepSeek-R1), RLCS (GLM-4.1V-Thinking) | Theory | DPO paper, DeepSeek-R1 report, GLM-4.1V-Thinking |
| 🔲 | Week 12 | VLM-specific RL — RLVR (reinforcement learning from verifiable rewards). Hands-on: run a DPO fine-tune with TRL | Code | TRL, RLVR papers, HuggingFace PEFT |

#### Suggested reading

[Reinforcement Learning (RL) Guide from Unsloth](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)

* RL is where and agent learns to make decisions by interacting with an environment and receiving feedback [rewards, penalties].
* **Action:** What the model generates (an answer to a question)
* **Reward:** A signal that indicates how good or bad the model's answer is. Dit it follow instructions, does it handle safety?
* **Environment:** The scenario or task that the model is working on. For example code generation, helpfullness, etc..

* Things to pay attention to: RL, RLVR, PPO, GRPO, RLHF, RF, DPO. 

### Phase 5 — Agentic Systems

* What are agentic systems?
* A 'normal LLM interaction' is basically you ask a question and it gives back a text answer. 
* An **agent** is adding actions into a loop. So instead of answering right away the loop can:
   1. Reason or think about the steps
   2. Call a tool (code, calculator, use an API, read a file for context)
   3. observe a result and decide if it makes sense, or if it needs another tool call or more reasoning
   4. repeat until answer is satisfactory.

* It boils down to think/act/observe/loop

How do agents and VLMS interact? 

* If you give an agent vision capabilities, it now can read markdown and also a pdf to understand a chart for example. It can see a screenshot and understand the layout and feed that into the generation. It can watch a video and understand. 
* So the VLM becomes the perception 'brain' that an agent can use in its loop. 


| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** |
| --- | --- | --- | --- | --- |
| 🔲 | Week 13 | Study agent foundations — ReAct, Toolformer, AgentBench. Learn tool-use patterns (function calling, structured output) | Theory | ReAct, Toolformer, AgentBench papers, Anthropic API docs |
| 🔲 | Week 14 | Build a small agent using smolagents or LangGraph with a VLM as perception module | Code | smolagents, LangGraph, Anthropic Claude agent SDK |

### Phase 6 — Large-Scale Pretraining & JAX 

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** |
| --- | --- | --- | --- | --- |
| 🔲 | Week 15 | JAX/Flax basics — functional paradigm, jit, vmap, pmap | Code | JAX docs, Flax tutorials |
| 🔲 | Week 16 | Scaling & systems papers — Chinchilla, LLM.int8(), FlashAttention | Theory | Chinchilla, LLM.int8(), FlashAttention papers |
| 🔲 | Week 17 | Distributed pretraining concepts — Megatron-LM, NeMo | Theory/Code | Megatron-LM, NVIDIA NeMo |


### Suggested reading/watching


#### [Umar Jamil's Coding a Multimodal Vision Language Model from Scratch](https://www.youtube.com/watch?v=vAmKB7iPkWw)

#### [Understanding MultiModal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)


#### [How AI Taught Itself to See](https://www.youtube.com/watch?v=oGTasd3cliM)


### Open-source Multi-modal models

We can take a look at the fully open-sourced [Molmo2-Models](https://github.com/allenai/molmo2)

Also, a competitive fully open source vision encoder in (Franca)[https://github.com/valeoai/Franca]

## Optional Module: Building Vision Solutions with Transformers

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 1 | DETR | Scoping | build a Transformer Detector from scratch | |
| 🔲 | Week 2 | Building on top of DETR | LW-DETR, RF-DETR |
| 🔲 | Week 3 | Segmentation and Pose | MaskDino (segmentation) DETR-Pose (pose estimation) |
| 🔲 | Week 4 | MOTracking |MOTR, MOTRv2 , TrackFormer, SAM|