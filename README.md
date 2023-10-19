# Sinkhorn Distance Minimization for Knowledge Distillation
## Installation
To install the environment,

## Download GLUE Data
Download the GLUE data using [this repository](https://github.com/nyu-mll/GLUE-baselines) or from GLUE benchmark website, unpack it to directory datas/glue and rename the folder `CoLA` to `COLA`.

## Download Pre-trained BERT
Download `bert_uncased_L-12_H-768_A-12` (BERT-base) and `bert_uncased_L-6_H-768_A-12` for teacher model and student model, respectively, from [this repository](https://github.com/google-research/bert). and use the [API from Huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py) to transform them to pytorch checkpoint.

## Task-specific BERT Model Distillation

## Task-specific T0 Model Distillation

## Task-specific GPT Model Distillation

## Student Checkpoints
The distilled student model for each task reported in the paper can be downloaded using the following link:
https://drive.google.com/drive/folders/1BsA0VHKSa_-Bp5I7dQ2Ftk2q7cIyPrdC
