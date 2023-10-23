import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class KL(nn.Module):
    def __init__(self):
        super(KL, self).__init__()  
        self.T = 1
    def forward(self, y_s, y_t, mode="classification"):
        y_s = y_s.view(-1, 32128)
        y_t = y_t.view(-1, 32128)
        y_s = torch.log_softmax(y_s, dim=-1) 
        y_t = torch.log_softmax(y_t, dim=-1)
        if mode == "regression":
            loss = F.mse_loss((y_s/self.T).view(-1), (y_t/self.T).view(-1))
        else:
            p_s = F.log_softmax(y_s/self.T, dim=-1)
            p_t = F.softmax(y_t/self.T, dim=-1)
            loss = -torch.sum(p_t * p_s, dim=-1).mean()
        return loss
    
class Sinkhorn(nn.Module):
    def __init__(self):
        super(Sinkhorn, self).__init__()
        self.T = 2   #0.55 #2
    def sinkhorn_normalized(self,x, n_iters=10):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self,x, y, epsilon=0.1, n_iters=20):
        Wxy = torch.cdist(x, y, p=1)  # 计算成本矩阵
        K = torch.exp(-Wxy / epsilon)  # 计算内核矩阵
        P = self.sinkhorn_normalized(K, n_iters)  # 计算 Sinkhorn 迭代的结果
        return torch.sum(P * Wxy)  # 计算近似 EMD 损失
    def forward(self, y_s, y_t, mode="classification"):
        softmax = nn.Softmax(dim=1)
        # selected_dims = [465,2163]
        # y_s = torch.index_select(y_s, dim=-1, index=torch.tensor(selected_dims).to(y_s.device))
        # y_t = torch.index_select(y_t, dim=-1, index=torch.tensor(selected_dims).to(y_t.device))
        p_s = softmax(y_s/self.T)
        p_t = softmax(y_t/self.T)
        emd_loss = 0.0008*self.sinkhorn_loss(x=p_s,y=p_t)   #  8
        return emd_loss
    
class RKL(nn.Module):
    def __init__(self):
        super(RKL, self).__init__()
        self.T = 2
    def forward(self,  y_s, y_t, mode="classification"):
        temperature = self.T
        if mode == "regression":
            
            loss = F.mse_loss((y_s/temperature).view(-1), (y_t/temperature).view(-1))
        else:
            p_s = F.softmax(y_s/temperature, dim=-1)
            p_s1 = F.log_softmax(y_s/temperature, dim=-1)
            p_t = F.log_softmax(y_t/temperature, dim=-1)
            loss =torch.sum(p_s1 * p_s, dim=-1).mean() -torch.sum(p_t * p_s, dim=-1).mean()
        return 0.1*loss
    
class JSKL(nn.Module):
    def __init__(self):
        super(JSKL, self).__init__()
        self.T = 2
    def js_divergence(self,p, q):
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(p, m, reduction='batchmean') +
                    F.kl_div(q, m, reduction='batchmean'))
    def forward(self, y_s, y_t, mode="classification"):
        temperature = self.T
        if mode == "regression":
            loss = F.mse_loss((y_s/temperature).view(-1), (y_t/temperature).view(-1))
        else:
            p_s = F.softmax(y_s/temperature, dim=-1)
            p_t = F.softmax(y_t/temperature, dim=-1)
            loss = (0.5 * self.js_divergence(p_s, p_t)).mean()
        return 0.1 * loss

