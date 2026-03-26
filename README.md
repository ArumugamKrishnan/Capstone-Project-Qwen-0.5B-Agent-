# Fine-tuning Qwen2-0.5B-Instruct for Aerospace Engineering

This project demonstrates the fine-tuning of the `Qwen2-0.5B-Instruct` Large Language Model (LLM) for specialized aerospace engineering tasks. Utilizing Direct Preference Optimization (DPO) and LoRA (Low-Rank Adaptation), the goal is to enhance the model's ability to respond to aerospace-specific queries with accurate, structured, and helpful information.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Dataset](#dataset)
- [Model Training (DPO with LoRA)](#model-training-dpo-with-lora)
- [Evaluation](#evaluation)
  - [Numerical Results](#numerical-results)
  - [Qualitative Analysis](#qualitative-analysis)
- [Results and Conclusion](#results-and-conclusion)
- [Future Work](#future-work)
- [Contact](#contact)

## Project Overview
This repository contains the code and resources for fine-tuning the Qwen2-0.5B-Instruct model on aerospace engineering-specific data. The primary objective is to create an AI assistant capable of providing precise and structured responses to queries related to aircraft design, maintenance, and operations, with a focus on adhering to instructions like 'Always PLAN steps before answering'.

## Features
*   **Custom Dataset**: Creation of a small, preference-based dataset for DPO fine-tuning.
*   **LoRA Fine-tuning**: Efficient adaptation of the Qwen2 model using Low-Rank Adaptation.
*   **DPO Training**: Application of Direct Preference Optimization to align the model with desired response styles.
*   **Qualitative & Numerical Evaluation**: Comparison of model performance before and after fine-tuning.
*   **Aerospace-Specific Assistant**: An AI agent designed to act as an 'Aerospace Aircraft Engineering Assistant'.

## Getting Started

### Prerequisites
*   Google Colab account (for running the Jupyter notebook).
*   Basic understanding of Python and machine learning concepts.

### Installation
All necessary Python packages can be installed by running the first cell of the provided Colab notebook:

```bash
!pip install -q transformers datasets accelerate peft trl bitsandbytes
```

### Running the Notebook
1.  Open the `fine_tuning_qwen2_dpo.ipynb` notebook in Google Colab (this notebook contains all the steps).
2.  Go to `Runtime` -> `Run all` to execute all cells sequentially.
3.  Alternatively, run cells individually to follow the step-by-step process of model loading, data preparation, training, and evaluation.

## Dataset
The project utilizes a small, custom-created Aerospace Human Preference Dataset (represented within the notebook). This dataset consists of 5 examples, each containing:
*   `prompt`: The user's query (e.g., "Checklist before Aircraft Maiden Flight.")
*   `chosen`: The preferred, high-quality response from the assistant.
*   `rejected`: A less preferred or lower-quality response.

While effective for demonstrating the DPO process, a significantly larger dataset would be required for robust production-ready performance.

## Model Training (DPO with LoRA)
The `Qwen2-0.5B-Instruct` model is fine-tuned using Direct Preference Optimization (DPO) combined with LoRA (Low-Rank Adaptation) for memory efficiency. Key training parameters include:
*   LoRA `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
*   DPO `per_device_train_batch_size=2`, `num_train_epochs=25`, `learning_rate=5e-5`

The `DPOTrainer` from the `trl` library is used for the training loop.

## Evaluation

### Numerical Results
The numerical evaluation on the training dataset yielded the following metrics:

| Metric                        | Value     |
| :---------------------------- | :-------- |
| `eval_loss`                   | 0.0093    |
| `eval_runtime`                | 1.23 seconds |
| `eval_samples_per_second`     | 4.07      |
| `eval_steps_per_second`       | 0.81      |

These results indicate successful optimization of the model towards the DPO objective on the provided dataset.

### Qualitative Analysis
Initial experiments with the very small dataset occasionally led to garbled or non-English output. However, after careful refinement of the dataset preprocessing, tokenizer handling, and generation parameters, the fine-tuned model demonstrated significant qualitative improvements:

*   **Improved Relevance**: The model became better at understanding and responding to specific aerospace concepts (e.g., correctly explaining 'flutter' as an aeroelastic phenomenon, unlike the base model which discussed Google's Flutter framework).
*   **Structured Responses**: The model consistently adhered to the 'Always PLAN steps before answering' instruction, providing well-organized answers.
*   **English Adherence**: Output remained consistently in English.

Below is a comparison table generated during the evaluation:

| Prompt                                                      | Before DPO (Excerpt)                                   | After DPO (Excerpt)                                     |
| :---------------------------------------------------------- | :----------------------------------------------------- | :------------------------------------------------------ |
| What are the primary considerations for aircraft structural design? | ...Material Selection: Selecting appropriate materials... | ...Safety: Ensuring that the system functions as intended... |
| Explain the concept of flutter in aircraft.                 | ...a software engineering framework developed by Airbus... | ...an aeroelastic phenomenon in aircraft...               |
| List common non-destructive testing methods for aerospace components. | ...Surface Mounting Testing (SMT)...                     | ...Plan = [1] # Define list of commonly used techniques... |

*(Full outputs are available within the `fine_tuning_qwen2_dpo.ipynb` notebook and in the `df_results` DataFrame.)*

## Results and Conclusion
This project successfully demonstrated the application of DPO and LoRA to fine-tune `Qwen2-0.5B-Instruct` for aerospace engineering tasks. Despite using a very small dataset, the qualitative improvements, particularly in the model's understanding of specific domain concepts and adherence to instruction following, were notable. The `eval_loss` confirms the DPO training objective was met.

## Future Work
*   **Expand Dataset**: The most critical next step is to curate a significantly larger and more diverse aerospace-specific DPO dataset (hundreds to thousands of examples) to achieve more robust and consistent performance and prevent potential 'catastrophic forgetting' of general knowledge.
*   **Hyperparameter Tuning**: Optimize LoRA and DPO hyperparameters for better performance.
*   **Larger Model**: Experiment with larger Qwen2 models or other base models.
*   **Deployment**: Explore options for deploying the fine-tuned model for inference.

## Contact
Arumugam Krishnan.
