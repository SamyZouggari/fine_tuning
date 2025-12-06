# Parameter Efficient Fine-Tuning of a Large Language Model on a GPU

This repository is for the lab 2 of the course ID2223/HT2025 of KTH.

We want to perform a Fine-Tuning of an LLM, using a pre-trained model. In particular, we want to do it using Parameter Efficient Fine Tuning with LoRA (Low-Rank Adaptation).

# Our Service

Our service provides a specialized service for coding that has been trained on a code-specific dataset (Python, Java, SQL, etc.).

This service is available as a chatbot in the following space [samzito12/iris](https://huggingface.co/spaces/samzito12/iris) in hugging face. You have the liberty to choose the temperature you want from 0 to 1.5 and the max number of tokens the model can output from 32 to 512. Note that more tokens give a better response but it takes more time.

You can also select one of the examples prompt to test the model without waiting.

Also note that the inference takes usually a few minutes because it is done on CPU.
# Technical background
## Different ways of doing Fine-Tuning:

### Full Fine-Tuning

It updates all the model weights. So, the weight matrices are very large. For example, a 7B model has 7 billion weights.
All the weights are updated repeatedly for multiple steps and epochs. The problem with this way is that it requires a lot of memory to store and update the weights and a lot of computational power.

### LoRA

Instead of updating weights directly, we track changes. 
These weights changes are tracked in two separate, smaller matrices that get multiplied together to form a matrix the same size as the model's weight matrix.
They are low rank matrices. 
We sacrifice precision, in order to have a better efficiency.

The worse is the rank of the matrices, the worse is the precision, and the better is the computational efficiency. The problem is that very low rank matrices train only a very few number of parameters. 

But, does rank actually matter ? [A theory](https://youtu.be/t1caDsMzWBk?si=O3qcSzaNAiyIKe-0) states that downstream tasks are intrinsically low-rank. Higher rank may help however to teach complex behavior, or for a training that falls outside the scope of the pre-training.

### QLoRA

The idea is that the weights are coded in 16 bits in memory. By quantizing them, it reduces it until 4 bits. It's like compressing them. At the end, techniques permits to recover the precision of the parameters.

The advantage of it is that it uses less memory with recoverable quantization.

[A paper](https://arxiv.org/pdf/2305.14314) on QLoRA states that the rank of the matrices used in LoRA is unrelated to final performance if LoRA is used on all layers (for a range from 8 to 64)


## The hyperparameters

### Rank
The rank of the matrices A and B that are trained.

### Alpha 
Alpha determines the multiplier applied to the weight changes when AB is merged with the original weights.

Scale multiplier = Alpha/Rank

Common values : 2x rank for microsoft LoRA, 1/4x rank for QLoRA

It's something like an amplifying factor. Which quantify how the new parameters have an influence on the final matrix

### Dropout
Adding dropout of the networks on the training. Common values are 0.1 for 7B-13B and 0.05 for 33B-65B.

## LLM

What LLM can we choose ? The unsloth notebook gives a list of pre-trained and quantized models LLMs ready to be fine-tuned via QLoRA.

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

We have many LLMs at disposition. There are different size (from 1B to 405B parameters). Also, we have Instruct models that are adapted for single instructions.

## Inference

The inference pipeline is built on Hugging Face Spaces. It can be a chatbot, a text-to-image or anything else. We chose a Chatbot because of its simplicity and its fast inference.

Inference is done on a CPU, that means that the performance will be limited.

## Checkpointing the weight periodically
To ensure that we don't lose progress during training and to allow resuming or evaluating intermediate models, we can checkpoint the weights periodically at specified steps of the training. This is particularly useful for long training runs, experiments with unstable training, or when using limited hardware resources.

In Hugging Face's `TrainingArguments`, checkpointing can be configured using the following parameters:

- `save_steps`: Number of steps between each checkpoint. For example, `save_steps=10` will save the model every 10 steps.
- `save_total_limit`: Maximum number of checkpoints to keep. Older checkpoints will be deleted automatically.
- `output_dir`: Directory where the checkpoints will be saved.


# Model used

The service runs with a custom LLM. The original one is a Llama-3.2-3B-Instruct. After the pre-training, a fine-tuning has been done with the [mlabonne/opc-sft-stage2-chat](https://huggingface.co/datasets/mlabonne/opc-sft-stage2-chat/) dataset with the following hyperparameters : (rank = 128, alpha = 128, dropout = 0) for LoRA training.

# Methodology and Evaluation of the models

We chose a code-specific service because its evaluation is easily quantifiable using benchmarks. 

To evaluate the model, we used the human-eval benchmark released by openAI to evaluate LLMs on 164 code tasks. The model is evaluated on only 1 answer using a temperature of 0 (to save some time).

We don't have the data for the performance of the pretrained model on this benchmark (one source says that Llama-3-Instruct score 81.7 for 70B parameters and 62.2 for 8B parameters, our pretrained model has 3B parameters).

We developed **five different models** before selecting the one used in the UI, following both **data-centric** and **model-centric** approaches:

1. **Notebook example model:** trained on the `mlabonne/FineTome-100k` dataset with hyperparameters `rank = 16`, `alpha = 16`, and `dropout = 0`.
2. Same as the previous model, but trained on the `mlabonne/opc-sft-stage2-chat` dataset.
3. Same as the previous, but with `rank` and `alpha` increased from 16 to 128.
4. Same as the previous, but with `alpha` further increased to 256.
5. Same as the third model, but with `dropout` set to 0.1.

## Human-Eval Benchmark Results

| Model | Result |
|-------|--------|
| 1     | 0.366  |
| 2     | 0.433  |
| 3     | 0.457  |
| 4     | 0.427  |
| 5     | 0.445  |

Here are the results of the benchmark of the different models that we've created. That is why we used the third model for the UI.

We would love to try another pre-trained LLM, but we didn't have enough time to do it.