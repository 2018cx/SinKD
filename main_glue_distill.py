import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import glue_tasks_num_labels, glue_output_modes
from dataset import GlueDataset
from utils import set_seed, set_experiments, pkd_initialization
from glue_train import train, evaluate, predict_test, distill_train 
from torch.utils.data import DataLoader, RandomSampler
import transformers
from transformers import AutoModel
from torch.utils.data import Dataset
import os

#local_rank = int(os.environ.get("LOCAL_RANK", -1))

class SubDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

def parse_args():
    parser = argparse.ArgumentParser()

    # Experimental Setting
    parser.add_argument('--exp_name', type=str, default='student',
                        help='Name of the experiment')
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", 
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", 
                        help="Whether to run eval on the test set.")
    parser.add_argument("--test_output_dir", type=str, default="test_output_student",
                        help="Test output directory")

    # Distillation Setting
    parser.add_argument("--student_init", type=str, default="pretrained",
                        help="The way of initializing the student model [copy | pretrained]")
    parser.add_argument("--distill_loss",type=str, default="kd",
                        help="Distillation Loss type. ")
    
    # Hyperparameters for vanilla KD
    parser.add_argument("--alpha", default=0.8, type=float,
                        help="The weight to balance cross-entropy loss and logit-based KD loss")
    parser.add_argument("--temperature", default=2.0, type=float, 
                        help="Temperature for logit-based KD loss.")                   
    
    # Hyperparameters for AD-KD 
    parser.add_argument("--beta", default=1.0, type=float,
                        help="The weight to control the attribution distillation loss")
    parser.add_argument("--topk", default=768, type=int,
                        help="Number of embedding dimensions to calculate the token-level attribution score")
    parser.add_argument("--ig_steps", default=1, type=int,
                        help="Number of steps in Integrated Gradients (IG)")
    parser.add_argument("--normalization_type", default=2, type=int,
                        help="Which norm (L1 / L2) to normalize the attribution map [1 | 2]")       
    parser.add_argument("--loss_func", default="L2", type=str,
                        help="Metrics for the difference between attribution maps [L1 | L2 | smoothL1]")                     
                    
    # Hyperparameters for MGSKD
    parser.add_argument("--only_mgskd", action="store_true",
                        help="Whether to optimze MGSKD loss only (without cross-entropy loss and ogit-based KD loss)")
    parser.add_argument("--mgskd_w", default=1.0, type=float,
                        help="The weight to control MGSKD loss. Only available when mgskd_only set to False.")
    parser.add_argument("--matching_strategy", type=str, default="emb+uniform",
                        help="Layer mapping stategy [emb | uniform | emb+uniform | last | final | emb+final | triple]")
    
    # Model Setting
    parser.add_argument("--teacher_path", default=None, type=str, required=True,
                        help="Path to the teacher model")
    parser.add_argument("--student_path", default=None, type=str, required=True,
                        help="Path to the student model")
    parser.add_argument("--do_lower_case", action="store_true", 
                        help="Set this flag if you are using an uncased model.")

    # Dataset Setting
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="Specify the task name in glue for training")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--data_dir", default="./datas/glue", type=str, 
                        help="Saved Data`s directory,"
                        "download from the huggingface transformer utils/download_glue_data.py")

    # Training Setting
    parser.add_argument("--per_gpu_batch_size", default=16, type=int, 
                    help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_train_epochs", default=8.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=0, type=int, 
                        help="Number of steps of linear warmup.")
    parser.add_argument("--weight_decay", default=0.0, type=float, 
                        help="Weight decay.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, 
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--logging_steps", type=int, default=100, 
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    return parser.parse_args()

args = parse_args()

logger, args.checkpoint_dir = set_experiments(args, "distill_{}".format(args.task_name))

if args.local_rank == -1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1

args.device = device
args.data_dir = os.path.join(args.data_dir, args.task_name.upper())

logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1))

set_seed(args.seed)

if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()
try:
    args.num_labels = glue_tasks_num_labels[args.task_name]
except KeyError:
    raise ValueError("Task not found: %s" % (args.task_name))

# Path for task-specific teacher for each tasks
args.teacher_path = os.path.join(args.teacher_path, "checkpoints/best_checkpoint")
tokenizer = BertTokenizer.from_pretrained(args.teacher_path,
                                    do_lower_case=args.do_lower_case)
teacher_config = BertConfig.from_pretrained(args.teacher_path)
teacher = BertForSequenceClassification.from_pretrained(args.teacher_path, config=teacher_config)

# Path for student (small EBRT)
student_config = BertConfig.from_pretrained(args.student_path,
                                            num_labels=args.num_labels,
                                            finetuning_task=args.task_name)
student = BertForSequenceClassification.from_pretrained(args.student_path, 
                                                config=student_config)




# Initialize the student with the first layers of the teacher like PKD
if args.student_init=="copy":
    pkd_initialization(teacher, student)
elif args.student_init=="pretrained":
    pass    # Should correctly specify the path for pre-trained small BERT
else:
    raise NotImplementedError


args.s_hidden_size = student_config.hidden_size
args.t_hidden_size = teacher_config.hidden_size

if args.local_rank == 0:
    torch.distributed.barrier()

teacher.to(args.device)
student.to(args.device)
logger.info("Training/Evaluation Parameters : ")
for attr, value in sorted(args.__dict__.items()):
    logger.info("\t{}={}".format(attr.upper(), value))

if args.do_train:
    train_dataset = GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="train") 
    dev_dataset = GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="dev")
    metric_name, eval_best = distill_train(args, train_dataset, dev_dataset, student, teacher, tokenizer, logger)
    logger.info(" End Training ")
    if args.local_rank in [-1, 0]:
        last_output_dir = os.path.join(args.checkpoint_dir, "last_checkpoint")
        model_to_save = student.module if hasattr(student, "module") else student
        model_to_save.save_pretrained(last_output_dir)
        tokenizer.save_pretrained(last_output_dir)
        torch.save(args, os.path.join(last_output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", last_output_dir)
        os.makedirs("./results_student",exist_ok=True)
        with open("./results_student/dev_results_{}.txt".format(args.task_name), "a") as f:
            f.write("{}\t{}\t{}\n".format(args.exp_name, metric_name, eval_best))
        f.close()

if args.do_eval and args.local_rank in [-1, 0]:
    dev_dataset = GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="dev")
    logger.info("Loading checkpoint %s for evaluation", args.student_path)
    logger.info("Evaluate the following checkpoints: %s", args.student_path)
    config = BertConfig.from_pretrained(args.student_path)
    model = BertForSequenceClassification.from_pretrained(args.student_path, config=config)
    model.to(args.device)
    # Evaluate
    results = evaluate(args, model, dev_dataset, tokenizer, logger)
    logger.info("Results : {}".format(results))

if args.do_test and args.local_rank in [-1, 0]:
    config = BertConfig.from_pretrained(args.student_path)
    model = BertForSequenceClassification.from_pretrained(args.student_path, config=config)

    model.to(args.device)
    output_mode = glue_output_modes[args.task_name]
    logger.info("Test Trained Network in \n {}".format(args.student_path))

    if args.task_name == "mnli":
        test_dataset=GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="test")
        prediction = predict_test(args, model, test_dataset, logger)

        output_test_file = os.path.join(args.test_output_dir, 
                                    "{}.tsv".format(args.task_name))
        with open(output_test_file, "w") as f:
            f.write("index\tprediction\n")
            for index, item in enumerate(prediction):
                item = test_dataset.get_labels()[item]
                f.write("%d\t%s\n" % (index, item))
            f.close()

        args.task_name = "mnli-mm"
        test_dataset = GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="test")
        prediction = predict_test(args, model, test_dataset, logger)

        output_test_file = os.path.join(args.test_output_dir, 
                                    "{}.tsv".format(args.task_name))
        with open(output_test_file, "w") as f:
            f.write("index\tprediction\n")
            for index, item in enumerate(prediction):
                item = test_dataset.get_labels()[item]
                f.write("%d\t%s\n" % (index, item))
            f.close()
    else:
        test_dataset = GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="test")
        prediction = predict_test(args, model, test_dataset, logger)

        output_test_file = os.path.join(args.test_output_dir, 
                                        "{}.tsv".format(args.task_name.upper()))

        with open(output_test_file, "w") as f:
            f.write("index\tprediction\n")
            for index, item in enumerate(prediction):
                if output_mode == "regression":
                    if item > 5.0:
                        item = 5.0
                    elif item < 0:
                        item = 0
                    f.write("%d\t%3.3f\n" % (index, item))
                else:
                    item = test_dataset.get_labels()[item]
                    f.write("%d\t%s\n" % (index, item))
            f.close()
