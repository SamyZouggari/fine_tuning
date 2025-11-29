# Parameter Efficient Fine-Tuning of a Large Language Model on a GPU

This repository is for the lab 2 of the course ID2223/HT2025 of KTH.

We want to perform a Fine-Tuning of an LLM, using a pre-trained model (Llama). In particular, we want to do it using Parameter Efficient Fine Tuning with LoRA (Low-Rank Adaptation).

## Different ways of doing Fine-Tuning:

### Full Fine-Tuning

It updates all the model weights. So, the weight matrices are very large. For example, a 7B model hab 7 billion weights.
All the weights are updated repeatedly for multiple "epochs". The problem with this way is that it requires a lot of memory to store and update the weights.

### LoRA

Instead of updating weights directly, we track changes. 
These weights changes are tracked in two separate, smaller matrices that get multiplied together to form a matrix the same size as the model's weight matrix.
They are low rank matrices. 
We sacrify precision, in order to have a better efficiency.

The worse is the rank of the matrices, the worse is the precision, and the better is the efficiency. The problem is that very low rank matrices train only a very few number of parameters. 

But, does Rank actually matter ? The theory is that downstream tasks are intinsically low-rank. Higher rank may help however to teach complex behavior, or to train that falls outside the scope of previous training.

### QLoRA

The idea is that the parameters are coded in 16 bits in memory. By quantisize it, it reduces it until 4 bits. Is like compressing it. At the end, techniques permits to recover the precision of the parameters.

The advantage of it is that it uses less memory with recoverable quantization.

A paper on QLoRA states that the rank of the matrices used in LoRA is unrelated to final performance if LoRA is used on all layers (for a range from 8 to 64)


## The hyperparameters

### Alpha 
Alpha determines the multiplier applied to the weight changes when added to the original weights.

Scale multiplier = Alpha/Rank

Common values : 2x rank for microsoft LoRA
1/4 x rank for QLoRA

It's something like an amplifying factor. Which quantify how the new parameters have an influence on the final matrix

### Dropout

Common values are 0.1 for 7B-13B and 0.05 for 33B-65B.

# LLM

What LLM can we choose ? The FineTome Instruction Dataset gives a list of pre-trained and quantized models LLMs ready to be fine-tuned via QLoRA.

| Model |
|--------|
| Meta-Llama-3.1-8B-bnb-4bit |
| Meta-Llama-3.1-8B-Instruct-bnb-4bit |
| Meta-Llama-3.1-70B-bnb-4bit |
| Meta-Llama-3.1-405B-bnb-4bit |
| Mistral-Small-Instruct-2409 |
| mistral-7b-instruct-v0.3-bnb-4bit |
| Phi-3.5-mini-instruct |
| Phi-3-medium-4k-instruct |
| gemma-2-9b-bnb-4bit |
| gemma-2-27b-bnb-4bit |
| Llama-3.2-1B-bnb-4bit |
| Llama-3.2-1B-Instruct-bnb-4bit |
| Llama-3.2-3B-bnb-4bit |
| Llama-3.2-3B-Instruct-bnb-4bit |
| Llama-3.3-70B-Instruct-bnb-4bit |

We have many LLMs at disposition. There are different size (from 1B to 405B parameters). Also, we have Instruct models that are adapted for chatbots.

# Inference

The inference pipeline is built on Hugging Face Spaces. It can be a chatbot, a text-to-image or anything else. We chose a Chatbot (Why ?).

Inference is done on a CPU, that means that performance will be limited

# Checkpointing the weight periodically

# Saving the LLM

The LLM is saved on HuggingFace, in the samzito12 profile

# App/service

We should use an app/service that uses the fine tuned LLM to make value-added decisions.

We should come to ideas of how we can allow people to use the fine Tuned LLM.

We should also try different open-source LLM to find the one that works best with our UI
