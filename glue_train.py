import os
import timeit
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import GlueDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from loss import *
from utils import matching_alignment
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, glue_compute_metrics, glue_output_modes
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import EvalPrediction, PredictionOutput
import time
import torch.distributed as dist
from sklearn.metrics import matthews_corrcoef

def get_eval_metric(task_name, results):
    if task_name == "cola":
        metric_name = 'mcc'
    if task_name == "mnli":
        metric_name = 'mnli/acc'
    if task_name == "mnli-mm":
        metric_name = 'mnli-mm/acc'
    if task_name == "mrpc":
        metric_name = 'f1'
    if task_name == "qqp":
        metric_name = 'acc'
    if task_name in ["qnli", "rte", "sst-2", "wnli"]:
        metric_name = 'acc'
    if task_name == "sts-b":
        metric_name = 'spearmanr'
    current_eval = results[metric_name]
    return current_eval, metric_name

def build_compute_metrics_fn(task_name: str):
    output_mode = glue_output_modes[task_name]
    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(task_name, preds, p.label_ids)
    return compute_metrics_fn

def train(args, train_dataset, dev_dataset, model, tokenizer, logger):
    # Fine-tnue
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join("./tb_logs/", args.task_name, args.exp_name))

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                sampler=train_sampler, 
                                batch_size=args.train_batch_size,
                                collate_fn=default_data_collator)
    
    args.eval_steps = int(len(train_dataloader) // args.gradient_accumulation_steps)
    t_total = int(args.eval_steps * args.num_train_epochs)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0}]
    
    optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=args.learning_rate, 
                        eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                num_warmup_steps=args.warmup_steps, 
                                num_training_steps=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                    device_ids=[args.local_rank], 
                                    output_device=args.local_rank, 
                                    find_unused_parameters=True)

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # Train!
    logger.info("*****Start Training*****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
    
    global_step = 0
    best_eval = 0.0
    if args.task_name == "mnli":
        best_eval_list = [0.0, 0.0]
    model.zero_grad()

    train_iterator = trange(0, int(args.num_train_epochs), 
                        desc="Epoch", 
                        disable=args.local_rank not in [-1, 0])
    loss_log = {}
    loss_log["ce"] = 0.0

    if args.local_rank in [-1, 0] and ((global_step) % args.eval_steps == 0):
    #if args.eval_steps > 0 and (global_step % args.eval_steps == 0):
        results = evaluate(args, model, dev_dataset, tokenizer, logger)
        if args.task_name == "mnli":
            current_eval_list = []
            metric_name_list = []
            for (i, result) in enumerate(results):
                for key, value in result.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                if i == 0:
                    args.task_name = "mnli"
                else:
                    args.task_name = "mnli-mm"
                eval_temp, metric_temp = get_eval_metric(args.task_name, result)
                current_eval_list.append(eval_temp)
                metric_name_list.append(metric_temp)
            args.task_name = "mnli"

        else:
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            current_eval, metric_name = get_eval_metric(args.task_name, results)
        if args.task_name == "mnli":
            current_eval = sum(current_eval_list)

        if current_eval > best_eval:
            if args.task_name == "mnli":
                best_eval_list[0] = current_eval_list[0]
                best_eval_list[1] = current_eval_list[1]
            best_eval = current_eval
            # best_output_dir = os.path.join(args.checkpoint_dir, "best_checkpoint")
            # model_to_save = (model.module if hasattr(model, "module") else model)
            # model_to_save.save_pretrained(best_output_dir)
            # tokenizer.save_pretrained(best_output_dir)
            # torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
            # torch.save(optimizer.state_dict(), os.path.join(best_output_dir, "optimizer.pt"))
            # torch.save(scheduler.state_dict(), os.path.join(best_output_dir, "scheduler.pt"))
            # logger.info("Best Model Saving optimizer and scheduler states to %s", best_output_dir)
        if args.task_name == "mnli":
            for (j, value) in enumerate(current_eval_list):
                logger.info("[EVAL] Current Results : {} - {}".format(metric_name_list[j], value))                    
                logger.info("[EVAL] Best Results : {} - {}".format(metric_name_list[j], best_eval_list[j]))
        else:
            logger.info("[EVAL] Current Results : {} - {}".format(metric_name, current_eval))                    
            logger.info("[EVAL] Best Results : {} - {}".format(metric_name, best_eval))

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, inputs in enumerate(epoch_iterator):
            model.train()
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)
        
            outputs = model(**inputs)
            loss = outputs[0]

            if args.local_rank == -1:
                loss_log["ce"] += loss.mean().item()
            else:
                loss_tmp = loss.clone()
                dist.all_reduce(loss_tmp, op=dist.ReduceOp.SUM)
                loss_log["ce"] += (loss_tmp / dist.get_world_size()).item()

            if args.n_gpu > 1:
                loss = loss.mean() 
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                len(epoch_iterator)<=args.gradient_accumulation_steps 
                and (step+1)==len(epoch_iterator)):

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1



                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("[TRAIN] Current [Epoch/Iter] : [{}|{}/{}|{}] | LR - {:.7f}"
                                                .format(epoch, args.num_train_epochs, step, len(epoch_iterator),scheduler.get_last_lr()[0]))
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    train_text = "[TRAIN] Train Loss : "
                    for name, value in loss_log.items():
                        tb_writer.add_scalar("loss_{}".format(name), (value / args.logging_steps), global_step)
                        train_text += "{} : {:.8f} |".format(name.upper(), (value / args.logging_steps))
                        loss_log[name] = 0.0
                    logger.info(train_text)
                if global_step>0:
                    args.eval_steps=20
                if args.local_rank in [-1, 0] and ((global_step) % args.eval_steps == 0):
                #if args.eval_steps > 0 and (global_step % args.eval_steps == 0):
                    results = evaluate(args, model, dev_dataset, tokenizer, logger)
                    print(loss)
                    if args.task_name == "mnli":
                        current_eval_list = []
                        metric_name_list = []
                        for (i, result) in enumerate(results):
                            for key, value in result.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            if i == 0:
                                args.task_name = "mnli"
                            else:
                                args.task_name = "mnli-mm"
                            eval_temp, metric_temp = get_eval_metric(args.task_name, result)
                            current_eval_list.append(eval_temp)
                            metric_name_list.append(metric_temp)
                        args.task_name = "mnli"

                    else:
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        current_eval, metric_name = get_eval_metric(args.task_name, results)
                    if args.task_name == "mnli":
                        current_eval = sum(current_eval_list)
            
                    if current_eval > best_eval:
                        if args.task_name == "mnli":
                            best_eval_list[0] = current_eval_list[0]
                            best_eval_list[1] = current_eval_list[1]
                        best_eval = current_eval
                        best_output_dir = os.path.join(args.checkpoint_dir, "best_checkpoint")
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(best_output_dir)
                        tokenizer.save_pretrained(best_output_dir)
                        torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(best_output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(best_output_dir, "scheduler.pt"))
                        logger.info("Best Model Saving optimizer and scheduler states to %s", best_output_dir)
                    if args.task_name == "mnli":
                        for (j, value) in enumerate(current_eval_list):
                            logger.info("[EVAL] Current Results : {} - {}".format(metric_name_list[j], value))                    
                            logger.info("[EVAL] Best Results : {} - {}".format(metric_name_list[j], best_eval_list[j]))
                    else:
                        logger.info("[EVAL] Current Results : {} - {}".format(metric_name, current_eval))                    
                        logger.info("[EVAL] Best Results : {} - {}".format(metric_name, best_eval))

                # Save model checkpoint
                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     output_dir = os.path.join(args.checkpoint_dir, "checkpoint-{}".format(global_step))
                #     model_to_save = model.module if hasattr(model, "module") else model
                #     model_to_save.save_pretrained(output_dir)
                #     tokenizer.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
                #     logger.info("Saving model checkpoint to %s", output_dir)
                #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                #     logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    if args.task_name == "mnli":
        logger.info("Fianl Best Results : {} - {}".format(metric_name_list, best_eval_list))
        return  metric_name_list, best_eval_list
    else:
        logger.info("Fianl Best Results : {} - {}".format(metric_name, best_eval))
        return metric_name, best_eval


def distill_train(args, train_dataset, dev_dataset, student, teacher, tokenizer, logger):
    # Distillation Training
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join("./tb_logs/", args.task_name, args.exp_name))

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                sampler=train_sampler, 
                                batch_size=args.train_batch_size,
                                collate_fn=default_data_collator)

    args.eval_steps = int(len(train_dataloader) // args.gradient_accumulation_steps)
    t_total = int(args.eval_steps * args.num_train_epochs)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay},
        {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0}]
    
    optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=args.learning_rate, 
                        eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                num_warmup_steps=args.warmup_steps, 
                                num_training_steps=t_total)

    if args.n_gpu > 1:
        teacher = nn.DataParallel(teacher)
        student = nn.DataParallel(student)

    if args.local_rank != -1:
        teacher = torch.nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # Train!
    logger.info("*****Start Training*****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
    
    global_step = 0

    loss_log = {}
    loss_log["ce"] = 0.0
    for name in args.distill_loss.split('+'):
        loss_log[name] = 0.0
    
    best_eval = 0.0
    #For MNLI task, MNLI-m and MNLI-mm
    best_eval_list = [0.0, 0.0]

    student.zero_grad()
    teacher.eval()

    train_iterator = trange(0, int(args.num_train_epochs), 
                        desc="Epoch", 
                        disable=args.local_rank not in [-1, 0])

    if "kd" in args.distill_loss.split('+'):
        kd_function = DistillKL(T=args.temperature)

    if "kd_anneal" in args.distill_loss.split('+'):
        kd_function = DistillKL_anneal(T=args.temperature)
    
    if "sinkhorn" in args.distill_loss.split('+'):
        if args.task_name=='mnli':
            saliency_function = Sinkhorn(T=1).to(args.device)
        else:
            saliency_function = Sinkhorn(T=1).to(args.device)
    
    if "integrated_saliency" in args.distill_loss.split('+'):
        if args.task_name=='mnli':
            saliency_function = integrated_gradient_mse_for_multiclass(total_step=args.ig_steps,top_k=args.topk, norm=args.normalization_type, loss_func=args.loss_func).to(args.device)
        else:
            saliency_function = integrated_gradient_mse(total_step=args.ig_steps,top_k=args.topk, norm=args.normalization_type, loss_func=args.loss_func).to(args.device)

    if "mgskd" in args.distill_loss.split('+'):
        mgskd_function = MGSKDLoss().to(args.device)

    train_losses = []

    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        start_time= time.time()
        for step, inputs in enumerate(epoch_iterator):
            student.train()
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)

            inputs["output_attentions"] = True
            inputs["output_hidden_states"] = True

            teacher.eval()
            #teacher_loss, teacher_logits, t_hidden, t_atts = teacher(**inputs)
            #loss, student_logits, s_hidden, s_atts = student(**inputs)
            a=teacher(**inputs)
            b=student(**inputs)
            teacher_loss, teacher_logits, t_hidden, t_atts=a[0],a[1],a[2],a[3]
            loss, student_logits, s_hidden, s_atts = b[0],b[1],b[2],b[3]
            student_loss = loss
            #teacher_logits=teacher(**inputs)[1]
            teacher_logits_detached = teacher_logits.detach() 
            #t_atts =  teacher(**inputs)[3]
            t_atts = tuple([item.detach() for item in t_atts])
            if args.local_rank == -1:
                loss_log["ce"] += loss.item()
            else:
                loss_tmp = loss.clone()
                dist.all_reduce(loss_tmp, op=dist.ReduceOp.SUM)
                loss_log["ce"] += (loss_tmp / dist.get_world_size()).item()

            if "kd" in args.distill_loss.split('+'):
                if args.task_name == "sts-b":
                    loss_kd = kd_function(student_logits, teacher_logits_detached, mode="regression")
                else:
                    loss_kd = kd_function(student_logits, teacher_logits_detached)
                loss = args.alpha * loss_kd + (1-args.alpha) * loss
                if args.local_rank == -1:
                    loss_log["kd"] += loss_kd.item()
                else:
                    dist.all_reduce(loss_kd, op=dist.ReduceOp.SUM)
                    loss_log["kd"] += (loss_kd / dist.get_world_size()).item()

            if "kd_anneal" in args.distill_loss.split('+'):
                if args.task_name == "sts-b":
                    loss_kd = kd_function(epoch, args.num_train_epochs, student_logits, teacher_logits_detached, mode="regression")
                else:
                    loss_kd = kd_function(epoch, args.num_train_epochs, student_logits, teacher_logits_detached)
                loss = args.alpha * loss_kd + (1-args.alpha) * loss
                if args.local_rank == -1:
                    loss_log["kd_anneal"] += loss_kd.item()
                else:
                    dist.all_reduce(loss_kd, op=dist.ReduceOp.SUM)
                    loss_log["kd_anneal"] += (loss_kd / dist.get_world_size()).item()

            if "saliency" in args.distill_loss.split('+'):   
                if args.task_name=='mnli':
                    saliency_loss = saliency_function(student_logits, teacher_logits)
                    #saliency_loss = saliency_function(student_logits, teacher_logits, s_hidden, t_hidden)
                else:          
                    saliency_loss = saliency_function(student_logits, teacher_logits)
                    #saliency_loss = saliency_function(student_loss, teacher_loss, s_hidden, t_hidden)
                loss = loss + args.beta * saliency_loss
                if args.local_rank == -1:
                    loss_log["saliency"] += saliency_loss.item()
                else:
                    dist.all_reduce(saliency_loss, op=dist.ReduceOp.SUM)
                    loss_log["saliency"] += (saliency_loss / dist.get_world_size()).item()

            if "integrated_saliency" in args.distill_loss.split('+'):
                if args.task_name=='mnli':
                    saliency_loss = saliency_function(inputs, student, teacher, 3)
                else:
                    saliency_loss = saliency_function(inputs, student, teacher)
                loss = loss + args.saliency_w * saliency_loss
                if args.local_rank == -1:
                    loss_log["integrated_saliency"] += saliency_loss.item()
                else:
                    dist.all_reduce(saliency_loss, op=dist.ReduceOp.SUM)
                    loss_log["integrated_saliency"] += (saliency_loss / dist.get_world_size()).item()

            if "mgskd" in args.distill_loss.split('+'):
                t_hidden, s_hidden = matching_alignment(t_hidden, s_hidden,
                                                        matching_strategy=args.matching_strategy)
                mgskd_loss = mgskd_function(s_hidden, t_hidden, inputs['attention_mask'])
                if args.only_mgskd:
                    loss = (loss - loss) + mgskd_loss
                else:
                    loss = loss + args.mgskd_w * mgskd_loss
                if args.local_rank == -1:
                    loss_log["mgskd"] += mgskd_loss.item()
                else:
                    dist.all_reduce(mgskd_loss, op=dist.ReduceOp.SUM)
                    loss_log["mgskd"] += (mgskd_loss / dist.get_world_size()).item()
                

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            train_losses.append(saliency_loss.item())  # 添加当前损失值到train_losses列表
            
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                len(epoch_iterator)<=args.gradient_accumulation_steps 
                and (step+1)==len(epoch_iterator)):

                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                student.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("[TRAIN] Current [Epoch/Iter] : [{}|{}/{}|{}] | LR - {:.7f}"
                                                .format(epoch, args.num_train_epochs, step, len(epoch_iterator),scheduler.get_last_lr()[0]))
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    train_text = "[TRAIN] Train Loss : "
                    for name, value in loss_log.items():
                        tb_writer.add_scalar("loss_{}".format(name), (value / args.logging_steps), global_step)
                        train_text += "{} : {:.8f} |".format(name.upper(), (value / args.logging_steps))
                        loss_log[name] = 0.0
                    logger.info(train_text)
                if args.task_name == "mnli":
                    if global_step>30000:
                        args.eval_steps=200
                else:
                    if global_step>0:
                        args.eval_steps=5
                if args.local_rank in [-1, 0] and args.eval_steps > 0 and (global_step % args.eval_steps == 0):
                    results = evaluate(args, student, dev_dataset, tokenizer, logger)
                    print("loss=",loss)
                    if args.task_name == "mnli":
                        current_eval_list = []
                        metric_name_list = []
                        for (i, result) in enumerate(results):
                            for key, value in result.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            if i == 0:
                                args.task_name = "mnli"
                            else:
                                args.task_name = "mnli-mm"
                            eval_temp, metric_temp = get_eval_metric(args.task_name, result)
                            current_eval_list.append(eval_temp)
                            metric_name_list.append(metric_temp)
                        args.task_name = "mnli"

                    else:
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        current_eval, metric_name = get_eval_metric(args.task_name, results)
                    
                    if args.task_name == "mnli":
                        current_eval = sum(current_eval_list)
                        if best_eval_list[0] < current_eval_list[0]:
                            best_eval_list[0] = current_eval_list[0]
                        if best_eval_list[1] < current_eval_list[1]:
                            best_eval_list[1] = current_eval_list[1]
            
                    if current_eval > best_eval:
                        if args.task_name == "mnli":
                            if best_eval_list[0] < current_eval_list[0]:
                                best_eval_list[0] = current_eval_list[0]
                            if best_eval_list[1] < current_eval_list[1]:
                                best_eval_list[1] = current_eval_list[1]
                            # best_eval_list[0] = current_eval_list[0]
                            # best_eval_list[1] = current_eval_list[1]
                        best_eval = current_eval
                        # best_output_dir = os.path.join(args.checkpoint_dir, "best_checkpoint")
                        # model_to_save = (student.module if hasattr(student, "module") else student)
                        # model_to_save.save_pretrained(best_output_dir)
                        # tokenizer.save_pretrained(best_output_dir)
                        # torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
                        # torch.save(optimizer.state_dict(), os.path.join(best_output_dir, "optimizer.pt"))
                        # torch.save(scheduler.state_dict(), os.path.join(best_output_dir, "scheduler.pt"))
                        # logger.info("Best Model Saving optimizer and scheduler states to %s", best_output_dir)
                    if args.task_name == "mnli":
                        for (j, value) in enumerate(current_eval_list):
                            logger.info("[EVAL] Current Results : {} - {}".format(metric_name_list[j], value))                    
                            logger.info("[EVAL] Best Results : {} - {}".format(metric_name_list[j], best_eval_list[j]))
                    else:
                        logger.info("[EVAL] Current Results : {} - {}".format(metric_name, current_eval))                    
                        logger.info("[EVAL] Best Results : {} - {}".format(metric_name, best_eval))

                # Save model checkpoint
                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     output_dir = os.path.join(args.checkpoint_dir, "checkpoint-{}".format(global_step))
                #     model_to_save = student.module if hasattr(student, "module") else student
                #     model_to_save.save_pretrained(output_dir)
                #     tokenizer.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
                #     logger.info("Saving model checkpoint to %s", output_dir)
                #     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                #     logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    if args.task_name == "mnli":
        logger.info("Fianl Best Results : {} - {}".format(metric_name_list, best_eval_list))
        return metric_name_list, best_eval_list
    else:
        logger.info("Fianl Best Results : {} - {}".format(metric_name, best_eval))
        return metric_name, best_eval

def evaluate(args, model, dataset, tokenizer, logger, prefix=""):

    args.eval_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)

    if args.task_name == "mnli":
        output = []
        datasets = []
        datasets.append(dataset)
        args.task_name = "mnli-mm"
        datasets.append(GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="dev"))
        for i, dataset in enumerate(datasets):
            eval_sampler = SequentialSampler(dataset)
            eval_dataloader = DataLoader(dataset, 
                                sampler=eval_sampler, 
                                batch_size=args.eval_batch_size,
                                collate_fn=default_data_collator)
            if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_losses = []
            preds = []
            labels = []
            start_time = timeit.default_timer()
            for inputs in tqdm(eval_dataloader, desc="Evaluating"):
                has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(args.device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    if has_labels:
                        loss, logits = outputs[:2]
                        loss = loss.mean().item()
                        labels.append(inputs.get("labels").detach())
                        preds.append(logits)
                        eval_losses.append(loss)
                    else:
                        loss = None
                        logits = outputs[0]
                        labels=None
            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
            if i == 0:
                args.task_name = "mnli"
            else:
                args.task_name = "mnli-mm"
            metrics  = build_compute_metrics_fn(args.task_name)(EvalPrediction(predictions=preds, label_ids=labels))
            metrics["loss"] = np.mean(eval_losses)
            output.append(PredictionOutput(predictions=preds, label_ids=labels, metrics=metrics).metrics)
        args.task_name ="mnli"

    else:
        eval_sampler = SequentialSampler(dataset)

        eval_dataloader = DataLoader(dataset, 
                            sampler=eval_sampler, 
                            batch_size=args.eval_batch_size,
                            collate_fn=default_data_collator)

        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        eval_losses = []
        preds = []
        labels = []
        start_time = timeit.default_timer()

        for inputs in tqdm(eval_dataloader, desc="Evaluating"):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    loss, logits = outputs[:2]
                    loss = loss.mean().item()
                    labels.append(inputs.get("labels").detach())
                    preds.append(logits)
                    eval_losses.append(loss)
                else:
                    loss = None
                    logits = outputs[0]
                    labels=None

        preds = torch.cat(preds, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        metrics  = build_compute_metrics_fn(args.task_name)(EvalPrediction(predictions=preds, label_ids=labels))
        metrics["loss"] = np.mean(eval_losses)
        output = PredictionOutput(predictions=preds, label_ids=labels, metrics=metrics).metrics
    model.train()
    return output


def predict_test(args, model, test_dataset, logger, prefix=""):

    output_mode = glue_output_modes[args.task_name]
    args.eval_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset,
                        sampler=eval_sampler, 
                        batch_size=args.eval_batch_size,
                        collate_fn=default_data_collator)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Make Prediction File {} *****".format(prefix))
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = []
    start_time = timeit.default_timer()

    for inputs in tqdm(eval_dataloader, desc="Prediction"):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs[0]

        preds.append(logits)

    preds = torch.cat(preds, dim=0).cpu().numpy()
    prediction = PredictionOutput(predictions=preds, label_ids=None, metrics=None).predictions
    if output_mode =="classification":
        prediction = np.argmax(prediction, axis=1)
    return prediction
