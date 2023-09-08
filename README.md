# S16-Assignment-Solution

This repository contains an implementation of a Encoder-Decoder transformer model for translation task from english to french using Opus dataset implemented using PyTorch Lightning. The model is trained using the One Cycle Policy, Dynamic Padding, Parameter Sharing and FP16 precision training.

## Files Description

The repository consists of the following files:

## Dataset

The Opus Books dataset used in this project contains English and French sentences for training the translation model. In preprocessing the dataset,  `dataset.py` remove sentences that don't meet the token length 150 for english sentences and difference between token length of english and french is not more than 10.

Creating a README.md for your GitHub repository is essential to help users understand your project, how to use it, and the steps to reproduce your results. Here's a template for a README.md file that you can use as a starting point for your repository:

```
markdown
# Transformer Translation Model

This repository contains the code and resources for training a transformer-based English to French translation model using PyTorch Lightning. The goal is to achieve a loss under 1.8 on a custom dataset while handling various constraints like sentence length and token count.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, we train a Transformer model for English-to-French translation using the Opus Books dataset. The model is designed to meet specific requirements:

1. Remove English sentences with more than 150 tokens.
2. Remove French sentences where len(french_sentences) > len(english_sentence) + 10.
3. Achieve a loss under 1.8.

We implement various strategies to meet these requirements, including dynamic padding, parameter sharing, one cycle policy, and FP16 precision training.

## Requirements

- Python >= 3.7
- PyTorch
- PyTorch Lightning
- Other dependencies (listed in `requirements.txt`)

You can install the necessary packages using `pip`:

​```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone this repository:

```
bash
git clone https://github.com/yourusername/transformer-translation-model.git
cd transformer-translation-model
```

1. Download the Opus Books dataset and preprocess it according to the assignment requirements (removing sentences based on token count and length).
2. Create a `config.py` file for model hyperparameters, a `dataset.py` file for dataset handling, and a `model.py` file for the Transformer architecture.
3. Launch the training process by running the provided `train.ipynb` notebook or using the following command:

```
bash
python train.py
```

## Dataset

The Opus Books dataset used in this project contains English and French sentences for training the translation model. To preprocess the dataset, follow the guidelines in `dataset.py` to remove sentences that don't meet the token and length criteria.

## Model

The model architecture is defined in `model.py`, where we implement the Transformer architecture for translation task. Here is the model summary.

```
Max length of source sentence: 150
Max length of target sentence: 159
```

​        

```
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Global seed set to 1
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
```

​        

```
┏━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃   ┃ Name    ┃ Type             ┃ Params ┃
┡━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 0 │ model   │ Transformer      │ 56.3 M │
│ 1 │ loss_fn │ CrossEntropyLoss │      0 │
└───┴─────────┴──────────────────┴────────┘
```

​        

```
Trainable params: 56.3 M                                                                                           
Non-trainable params: 0                                                                                            
Total params: 56.3 M                                                                                               
Total estimated model params size (MB): 225                                                                        
```

We apply various techniques to optimize training like One Cycle Policy, Parameter Sharing and Dynamic Padding to achieve a low loss less than 1.8. 

## Training

Training is managed using PyTorch Lightning, with parameters specified in `config.py`. You can train the model using the provided `train.ipynb` notebook or by running `train.py`. 

## Results

The final model should achieve a loss under 1.8. 