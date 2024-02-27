# Sinkhorn Distance Minimization for Knowledge Distillation (COLING 2024)
## Installation
To install the environment, run:

`sh ins.sh`

## Download GLUE and SuperGLUE Data
Download the GLUE data using [this repository](https://github.com/nyu-mll/GLUE-baselines) or from GLUE benchmark website, unpack it to directory datas/glue and rename the folder `CoLA` to `COLA`.

Download the SuperGLUE data from SuperGLUE benchmark website.

## Download Pre-trained BERT
Download `bert_uncased_L-12_H-768_A-12` (BERT-base) and `bert_uncased_L-6_H-768_A-12` for teacher model and student model, respectively, from [this repository](https://github.com/google-research/bert). and use the [API from Huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py) to transform them to pytorch checkpoint.

## Task-specific BERT Model Distillation
The training script for **Task-specific Teacher Model Finetuning** can be found in the `script/teacher/` directory, where **$TEACHER_PATH** denotes the file path of the teacher model.

Similarly, the training script for **Task-specific Student Model Distillation** is located in the `script/student/` directory. In this case, **$STUDENT_PATH** and **$TEACHER_PATH** represent the file paths of the student and teacher models, respectively.

## Task-specific T0 Model Distillation
To install the environment, run:

`sh T0/ins.sh`

To perform **Task-specific Teacher Model Finetuning**, run:

`python3 T0/distillation_t.py     --dataset_name super_glue     --dataset_config_name DATASET_NAME     --template_name "TEMPLATE_NAME"     --model_name_or_path MODEL_DIR     --output_dir ./debug    --parallelize `

To perform **Task-specific Student Model Distillation**, run:

`python3 T0/distillation.py     --dataset_name super_glue     --dataset_config_name DATASET_NAME     --template_name "TEMPLATE_NAME"     --model_name_or_path MODEL_DIR     --output_dir ./debug    --parallelize `

## Task-specific GPT Model Distillation
To install the environment, run:

`sh GPT-Neo/ins.sh`

To perform **Task-specific Teacher Model Finetuning**, run:

`python3 GPT-Neo/distillation_t.py     --dataset_name super_glue     --dataset_config_name DATASET_NAME     --template_name "TEMPLATE_NAME"     --model_name_or_path MODEL_DIR     --output_dir ./debug    --parallelize `

To perform **Task-specific Student Model Distillation**, run:

`python3 GPT-Neo/distillation.py     --dataset_name super_glue     --dataset_config_name DATASET_NAME     --template_name "TEMPLATE_NAME"     --model_name_or_path MODEL_DIR     --output_dir ./debug    --parallelize `

## Student Checkpoints
The distilled student model for each task reported in the paper can be downloaded using the following link:
https://drive.google.com/drive/folders/1BsA0VHKSa_-Bp5I7dQ2Ftk2q7cIyPrdC

## Teacher Checkpoints
The teacher model for each task reported in the paper can be downloaded using the following link:

