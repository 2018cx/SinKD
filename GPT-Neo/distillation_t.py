#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning T0 in PyTorch, optionally few-shot.

This script is adapted from
https://github.com/huggingface/transformers/blob/master/examples/pytorch/multiple-choice/run_swag_no_trainer.py
as well as
https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
"""

import argparse
import logging
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import csv
import math

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler



import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
    AutoModelForCausalLM
)
from transformers.file_utils import PaddingStrategy
from promptsource.templates import DatasetTemplates
import warnings
#from colossalai.nn.parallel import GeminiDDP
warnings.filterwarnings("ignore", category=UserWarning)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning T0 in PyTorch, optionally few-shot.")
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "-s",
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name (usually a subset) of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "-t",
        "--template_name",
        type=str,
        default=None,
        required=True,
        help="The template/prompt name in `promptsource`.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Where to store the results CSV and (TODO) optionally the final model."
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default="/youtu-reid/irvingcui/gpt-neo-125m",
        required=True,
        help=(
            "Path to pretrained model or model identifier from huggingface.co/models. "
            "The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`"
        ),
    )
    parser.add_argument(
        "-pa",
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "-eb",
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader. Will be multiplied by the number of answer choices.",
    )
    parser.add_argument(
        "-tb",
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "-ns",
        "--num_shots",
        type=int,
        help="Number of training examples for few-shot learning. Default is None, which uses the entire train set.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4, #1e-4
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "-ep",
        "--num_train_epochs",
        type=int,
        default=300,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "-ms",
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "-ga",
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "-ie",
        "--input_eos",
        action="store_true",
        help=(
            "T0 was trained without EOS in its input sequences, which is the default in this script."
            "However, T5 was pretrained with EOS in its input sequences. See README for more info."
        ),
    )
    parser.add_argument(
        "-db",
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "-wb",
        "--wandb_proj",
        type=str,
        default=None,
        help="Project name for Weights & Biases. By default, W&B is disabled.",
    )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=32, #42
        help="Especially important for few-shot example sampling.",
    )
    parser.add_argument(
        "-cf",
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "-tk",
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "-il",
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "-tl",
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "-pml",
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "-st",
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the AdamW optimizer."
    )
    parser.add_argument(
        "-ls",
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "-ws",
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    args = parser.parse_args()

    return args


class DataCollatorForGPTNeo:
    def __init__(self, tokenizer, max_length=None, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        input_ids = [example["input_ids"] for example in examples]
        labels = [example["labels"] for example in examples]
        #print(input_ids)
        # Decode the input_ids into strings
        input_texts = [self.tokenizer.decode(ids) for ids in input_ids]

        inputs = self.tokenizer(
            input_texts,
            padding=True, #
            truncation=True, #
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # print(222,self.tokenizer.batch_decode(inputs["input_ids"])[0])
        # print(len(self.tokenizer.batch_decode(inputs["input_ids"])))
        # print(len(self.tokenizer.batch_decode(inputs["input_ids"])[0]))
        # input()


        with self.tokenizer.as_target_tokenizer():
            # Decode the labels into strings
            target_texts = [self.tokenizer.decode(lbl) for lbl in labels]
            targets = self.tokenizer(target_texts, padding=True, truncation=True, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt")

        # Pad the labels tensor to have the same sequence length as the input tensor
        # Calculate padding length
        padding_length = targets["input_ids"].shape[1] - inputs["input_ids"].shape[1]

        if padding_length > 0:
            # Pad the input tensor and attention_mask to have the same sequence length as the labels tensor
            inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.full((inputs["input_ids"].shape[0], padding_length), self.tokenizer.pad_token_id).long()], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros((inputs["attention_mask"].shape[0], padding_length)).long()], dim=1)
            padded_labels = targets["input_ids"]
        else:
            # Pad the labels tensor to have the same sequence length as the input tensor
            padding_length = -padding_length
            padded_inputs = inputs["input_ids"]
            padded_labels = torch.cat([targets["input_ids"], torch.full((targets["input_ids"].shape[0], padding_length), -100).long()], dim=1)

        # Update the inputs dictionary with the padded input_ids tensor
        #inputs["input_ids"] = padded_inputs

        batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": padded_labels,
        }


        return batch



@dataclass
class DataCollatorForMultipleChoice:
    def __init__(self, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [
                {
                    k: v[i]
                    for k, v in feature.items()
                    if k != "targets"
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        max_length = max([len(elem["input_ids"]) for elem in flattened_features] )
        max_length = max( max_length, max([len(elem["labels"]) for elem in flattened_features]))

        # Pad the input sequences to the maximum length
        batch["input_ids"] = [
            seq + [self.tokenizer.pad_token_id] * (max_length - len(seq))
            for seq in [elem["input_ids"] for elem in flattened_features]
        ]

        # Pad the labels and labels_attention_mask sequences to the maximum length
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id] * (max_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0] * (max_length - len(m)) 
            for m in [[1] * len(l) for l in [elem["labels"] for elem in flattened_features]]
        ]
        # # Convert to tensors
        # print(len(batch["input_ids"][0]))
        # print(len(batch["labels"][0]))
        # print(len(batch["labels_attention_mask"][0]))
        # print(len(batch["attention_mask"][0]))
        # input()
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()
    set_seed(args.seed)
    # 在训练循环中
    scaler = GradScaler()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.dataset_name == "anli":
            raw_train_dataset = load_dataset(args.dataset_name, split=f'train_{args.dataset_config_name}')  # dataset_config_name = "r1", "r2", or "r3"
            raw_eval_dataset = load_dataset(args.dataset_name, split=f'dev_{args.dataset_config_name}')
        else:
            raw_train_dataset = load_dataset("", "copa",split="train",cache_dir="",download_mode="reuse_cache_if_exists")
            raw_eval_dataset = load_dataset("", "copa",split="validation",cache_dir="",download_mode="reuse_cache_if_exists")
        

    else:
        raise ValueError('Please specify `args.dataset_name` and `args.dataset_config_name` as appear in `promptsource`.')
    #TODO(Victor): enable loading pre-processed dataset from https://huggingface.co/datasets/bigscience/P3

    # Trim a number of evaluation examples
    if args.debug:
        raw_train_dataset = raw_train_dataset.select(range(min(20, len(raw_train_dataset))))
        raw_eval_dataset = raw_eval_dataset.select(range(min(20, len(raw_eval_dataset))))

    column_names = raw_eval_dataset.column_names

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        #config = AutoConfig.from_pretrained("/youtu-reid/data/t-zero/T0pp")
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "Either `args.config_name` or `args.model_name_or_path` should be provided."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, padding_side='right')
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side='right')
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )


    model_namet = ""
    
    tmodel = AutoModelForCausalLM.from_pretrained(model_namet)
    tokenizer = AutoTokenizer.from_pretrained(model_namet, use_fast=not args.use_slow_tokenizer, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded")

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Get the prompt to apply and the possible targets.
    # TODO(Victor): If pulling from pre-processed data, remove this logic.
    if args.dataset_name == 'anli':
        prompts = DatasetTemplates('anli', None)
    else:
        prompts = DatasetTemplates(
            f"{args.dataset_name}"
            if args.dataset_config_name is None
            else f"{args.dataset_name}/{args.dataset_config_name}"
        )
    template = prompts[args.template_name]

    def generate_with_logits(model, input_ids, attention_mask, max_length):
        logits_output = []
        generated_output = input_ids

        for _ in range(max_length):
            with torch.no_grad():
                outputs = model(input_ids=generated_output, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                logits_output.append(logits)
                next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                generated_output = torch.cat([generated_output, next_token], dim=1)

                # 更新 attention_mask 以匹配新生成的 token
                new_attention_mask = torch.ones_like(next_token, dtype=torch.int64)
                attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)

        logits_output = torch.stack(logits_output, dim=1)
        return generated_output, logits_output

    def calculate_loss(model, input_ids, attention_mask, labels, pad_token_id):
        logits_output = []
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        generated_output = input_ids
        for i in range(0, input_ids.size(1)):
            if labels[0,i]==-100:
                s=i
                break
            outputs = model(input_ids=generated_output, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            logits_output.append(logits)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
            generated_output = torch.cat([generated_output, next_token], dim=1)
            with torch.no_grad():
            # 更新 attention_mask 以匹配新生成的 token
                new_attention_mask = torch.ones_like(next_token, dtype=torch.int64)
                attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)
            
        logits_output = torch.stack(logits_output, dim=1)
        loss = loss_fct(logits_output.view(-1, logits_output.size(-1)), labels[:,:s].view(-1))
        return loss

    def preprocess_train(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }
            input, target = template.apply(ex)
            ex_answer_choices = template.get_answer_choices_list(ex)
            #assert target in ex_answer_choices
            input_texts.append(input)
            #combined_text = input  + target
            #target_texts.append(combined_text)
            target = '\n '+'- '+target
            target_texts.append(target)
           
        model_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=args.input_eos,
        )


        with tokenizer.as_target_tokenizer():
            tokenized_targets = tokenizer(
                target_texts,
                padding=padding,
                max_length=args.target_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            model_inputs['labels'] = [
                [(t if t != tokenizer.pad_token_id else -100) for t in targets]
                for targets in tokenized_targets["input_ids"]
            ]
        return model_inputs

    def preprocess_eval(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        answer_choices_texts = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }

            input, target = template.apply(ex)
            ex_answer_choices = template.get_answer_choices_list(ex)
            assert target in ex_answer_choices
            input_texts.append(input)
            #combined_text = input + target
            #target_texts.append(combined_text)
            target='\n '+'- '+target
            target_texts.append(target)
            input_choices = ['\n '+'- '  + choice for choice in ex_answer_choices]
            answer_choices_texts.append(input_choices)
            #answer_choices_texts.append(ex_answer_choices)

        tokenized_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        tokenized_targets = [
            tokenizer(
                ans_choi,
                padding=True,
                max_length=args.target_max_length,
                truncation=True,
            )
            for ans_choi in answer_choices_texts
        ]

        features = {
            k: [
                [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in tokenized_inputs.items()
        }

        features["labels"] = [
            tokenized_targets[idx]["input_ids"]
            for idx in range(bs)
        ]

        features["labels_attention_mask"] = [
            tokenized_targets[idx]["attention_mask"]
            for idx in range(bs)
        ]
        features["targets"] = [
            answer_choices_texts[idx].index(t)
            for idx, t in enumerate(target_texts)
        ]

        return features

    with accelerator.main_process_first():
        eval_dataset = raw_eval_dataset.map(preprocess_eval, batched=True, remove_columns=column_names)

        if args.num_shots is not None:
            sample_indices = random.sample(range(0, len(raw_train_dataset)), k=args.num_shots)
            raw_train_dataset = raw_train_dataset.select(sample_indices)
        train_dataset = raw_train_dataset.map(preprocess_train, batched=True, remove_columns=column_names)

    # Log a few random examples:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.debug(f"Sample {index} of the training set: {train_dataset[index]}.")
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.debug(f"Sample {index} of the evaluation set: {eval_dataset[index]}.")

    # DataLoaders creation:
    train_collator = DataCollatorForGPTNeo(
        tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_collator,
        batch_size=args.per_device_train_batch_size
    )

    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        eval_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        eval_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in tmodel.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in tmodel.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.parallelize:
        num_gpus = torch.cuda.device_count()
        assert num_gpus > 1, "You need at least 2 GPUs to use `model.parallelize()`."
        tmodel.parallelize()
        optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            optimizer, train_dataloader, eval_dataloader)
    else:
        tmodel, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            tmodel, optimizer, train_dataloader, eval_dataloader)

    # Metrics
    metric = load_metric("/youtu-reid/data/t-zero/dataset_huggingface/accuracy")

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_steps = 0

    if args.wandb_proj and accelerator.is_main_process:
        import wandb
        extra_metadata = {
            'template_jinja': template.jinja,
            'template_answer_choices': template.answer_choices,
            'template_reflects_original_task': template.metadata.original_task,
            'template_choices_in_prompt': template.metadata.choices_in_prompt,
            'template_comment': template.reference,
        }
        run_config = vars(args)
        run_config.update(extra_metadata)
        wandb.init(
            project=args.wandb_proj,
            config=run_config,
            # name=f'S{len(train_set)} {args.template_name} R{args.seed}',  # uncomment to customize each run's name
            # reinit=True,  # uncomment if running multiple runs in one script
        )

    result_table = []
    maxscore=0



    for epoch in range(1, 300):
        tmodel.train()
        sum_loss=0
        for step, batch in enumerate(train_dataloader):

            loss = calculate_loss(
                        tmodel,
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["labels"],
                        tokenizer.pad_token_id
                    )

            loss = loss / args.gradient_accumulation_steps
            #sum_loss=sum_loss+loss
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1
                loss = loss.item()
  

            if step%100==0:
                total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes
            
                tmodel.eval()
                for batch1 in eval_dataloader:
                    model_inputs = {
                        k: batch1[k]
                        for k in ["input_ids", "attention_mask", "labels"]
                    }
                    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"]).cuda()
                    with torch.no_grad():
                        # Generate a sequence of tokens
                        generated_output, logits = generate_with_logits(
                            tmodel,
                            model_inputs["input_ids"],
                            model_inputs["attention_mask"],
                            model_inputs["labels"].shape[1],  # 您期望的输出句子长度
                        )


                    predictions=torch.zeros(4).cuda()
                    for i in range(4):
                        s=0
                        for j in range(2,model_inputs["labels"].shape[1]):
                            if model_inputs["labels"][i*2+batch1["targets"][i],j]!=50256:
                                # if generated_output[i*2,j+model_inputs["labels"].shape[1]+2]==model_inputs["labels"][i*2,j]:
                                #     s=s+1
                                # if generated_output[i*2,j+model_inputs["labels"].shape[1]+2]==model_inputs["labels"][i*2+1,j]:
                                #     s=s-1
                                s=s+logits[i*2,j,model_inputs["labels"][i*2,j]]-logits[i*2+1,j,model_inputs["labels"][i*2+1,j]]
                        if s>=0:
                            predictions[i]=0
                        else:
                            predictions[i]=1


                    metric.add_batch(
                        predictions=accelerator.gather(predictions),
                        references=accelerator.gather(batch1["targets"]),
                    )

                eval_metric = metric.compute()
                score = eval_metric["accuracy"]  # TODO support other metrics; currently hardcoded at load_metric() anyway
                if score>maxscore:
                    maxscore=score
                    tmodel.save_pretrained(os.path.join('/youtu_pedestrian_detection/xiaocui/teachergpt',args.dataset_config_name))
                accelerator.print(f"Accuracy: {score}")
                # result_table.append({
                #     "dataset_name": args.dataset_name,
                #     "dataset_config_name": args.dataset_config_name,
                #     "template_name": args.template_name,
                #     "epoch": epoch,
                #     "step": global_steps,
                #     "metric": 'accuracy',
                #     "score": score,
                # })
                # if args.wandb_proj and accelerator.is_main_process:
                #     wandb.log({"accuracy": score}, step=global_steps)
                tmodel.train()
                #print("sum",sum_loss)
                print("max",maxscore)


        if args.wandb_proj:
            wandb.finish()


if __name__ == "__main__":
    main()