import torch
import random
import numpy as np
import os
import logging
import datetime
import math

def set_seed(seed):
    if seed is None:
        seed = 0
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_experiments(args, prefix=""):
    base_dir = os.path.join('./experiments', "exp_"+prefix, args.exp_name)
    os.makedirs(base_dir,exist_ok=True)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir,exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.addHandler(logging.FileHandler(base_dir+'/logs.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    return logger, checkpoint_dir

def pkd_initialization(teacher, student):
    assert len(teacher.bert.encoder.layer) > len(student.bert.encoder.layer)
    student_dict = student.state_dict()
    pretrained_dict = {}
    for k, v in teacher.state_dict().items():
        if 'qa_outputs' in k:
            continue
        if k in student_dict:
            pretrained_dict[k] = v
    student_dict.update(pretrained_dict)
    student.load_state_dict(student_dict)

def matching_alignment(t_hidden, s_hidden, matching_strategy):
    def compute_gcd(x, y):
        while y != 0:
            (x, y) = (y, x % y)
        return x
    assert matching_strategy in ["emb","uniform", "emb+uniform", "last", "final", "emb+final", "triple"]

    if matching_strategy == "emb":
        t_hidden = t_hidden[0].unsqueeze(1)
        s_hidden = s_hidden[0].unsqueeze(1)

    elif matching_strategy == "uniform":
        gcd = compute_gcd(len(t_hidden)-1, len(s_hidden)-1)
        step_t = (len(t_hidden)-1) / gcd
        step_s = (len(s_hidden)-1) / gcd
        t_hidden = torch.stack(t_hidden[step_t::step_t], dim=1)
        s_hidden = torch.stack(s_hidden[step_s::step_s], dim=1)

    elif matching_strategy == "emb+uniform":
        gcd = compute_gcd(len(t_hidden)-1, len(s_hidden)-1)
        step_t = int((len(t_hidden)-1) / gcd)
        step_s = int((len(s_hidden)-1) / gcd)
        t_hidden = torch.cat((t_hidden[0].unsqueeze(1),torch.stack(t_hidden[step_t::step_t], dim=1)), dim=1)
        s_hidden = torch.cat((s_hidden[0].unsqueeze(1),torch.stack(s_hidden[step_s::step_s], dim=1)), dim=1)

    elif matching_strategy == "last":
        start = len(t_hidden) - len(s_hidden) + 1
        t_hidden = torch.stack(t_hidden[start:], dim=1)
        s_hidden = torch.stack(s_hidden[1:], dim=1)

    elif matching_strategy == "final":
        t_hidden = t_hidden[-1].unsqueeze(1)
        s_hidden = s_hidden[-1].unsqueeze(1)

    elif matching_strategy == "emb+final":
        t_hidden = torch.stack((t_hidden[0], t_hidden[-1]), dim=1)
        s_hidden = torch.stack((s_hidden[0], s_hidden[-1]), dim=1)

    elif matching_strategy == "triple":
        t_middle = int((len(t_hidden)-1)/2)
        s_middle = int((len(s_hidden)-1)/2)
        t_hidden = torch.stack((t_hidden[0], t_hidden[t_middle], t_hidden[-1]), dim=1)
        s_hidden = torch.stack((s_hidden[0], s_hidden[s_middle], s_hidden[-1]), dim=1)   

    else:
        raise NotImplementedError

    return t_hidden, s_hidden