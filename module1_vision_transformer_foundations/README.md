
## Notebook Overview


### Week 1 001_vit_from_scratch.ipynb
* Here, we build a ViT from scratch. PatchEmbedding (Conv2d-based), positional embeddings (sinusoidal + learned), AttentionHead, MultiHeadAttention, FeedForward, TransformerBlock, TransformerEncoder, and ClassificationHead.

* We try to build some intuition around shape, visualizations, understanding the patches and how they relate back to the original image. 

* Train a simple FashionMNIST to see if our model can learn.



**Connection to VLMs:** 
The ViT encoder is the backbone for most modern VLMs inlcuding(CLIP, BLIP2, LLaVA, Qwen-VL). Understanding patches and their embeddings is important to see how we build this up. Understanding the cls_token and how we add learnable vectors is important. 



### 002_compare_ViTs.ipynb

Train a vanilla ViT on CIFAR-100 as a baseline. Then add swin-style windowed attention to compare. And finally, we add shifted windows with `torch.roll`. These are trained in local mps, but most of the computational gains would be seen with bigger images, more patches, and more heads on the MHA.


**Connection to VLMs:** Understanding how to make attention faster is critical. Attention when computed globally is very expensive and there is a whole line of research on how to make attention faster without comprising accuracy. Also local vs. global attention understanding. 



### 002_CoAtNet.ipynb
Build a  hybrid CNN + Transformer architecture. Compares against vanilla ViT on CIFAR-100.

Also, we take a look at how `timm` actually would implement this to learn how people are writing these models in bigger porjects. 

CNNs are good at local features with built-in inductive biases (translation equivariance), transformers are good at global reasoning. Stacking CNNs early and attention late gives the best of both worlds. 


**Connection to VLMs:** Many VLM encoders use hybrid CNN and attention. This simple implementations helps build that inuition. 



### 003_MAE.ipynb
Here, we start to build a self-supervised pre-training model. We try to understand how to build a decoder that predicts missing patches in pixel space. We force the model to learn a useful visual representation by making it predict missing patches of an image. 




**Connection to VLMs:**
Some poeple say this is the 'BERT' moment for vision. How can we create models that learn without labels from massive datasets, that can give us useful weights for fine tuning or to use directly as representations. 

Understanding this paradigm is important to understand contrastive pre-training like in CLIP or SigLip. 


