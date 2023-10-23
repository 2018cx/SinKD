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
        y_s = y_s.view(-1, 50257)
        y_t = y_t.view(-1, 50257)
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
        self.T = 2  
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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear = nn.Linear(2048, 4096)

    def forward(self, x):
        x= x.cuda(1)
        x = x.view(-1, 2048)
        x = self.linear(x)
        #x = x.view(-1, 27, 4096)
        return x
    
class saliency_mse(nn.Module):
    def __init__(self):
        super(saliency_mse, self).__init__()
        self.top_k = 500
        self.norm = 2

    def forward(self, s_loss, t_loss, s_hidden, t_hidden,mlp):
        
        with torch.enable_grad():
            t_grad = torch.autograd.grad(t_loss, t_hidden, create_graph=False,retain_graph=True)
        t_input_grad = t_grad[0] # (bsz, max_len, hidden_dim)
        t_saliency = t_input_grad * t_hidden.detach()  
        # t_topk_saliency = torch.topk(torch.abs(t_saliency),self.top_k,dim=-1)[0] # (bsz, max_len, top_k)
        # t_saliency = torch.norm(t_topk_saliency,dim=-1) # (bsz, max_len)
        t_saliency = t_saliency[:,0,:]
        t_saliency = F.normalize(t_saliency, p=self.norm, dim=1)

        s_grad = torch.autograd.grad(s_loss, s_hidden, create_graph=True, retain_graph=True)
        s_input_grad = s_grad[0] # (bsz, max_len, hidden_dim)
        s_saliency = s_input_grad * s_hidden
        s_saliency = mlp(s_saliency[:,0,:])
        # s_saliency = torch.norm(s_saliency,dim=-1) # (bsz, max_len)
        s_saliency = F.normalize(s_saliency, p=self.norm, dim=1)
        return F.mse_loss(t_saliency, s_saliency.to(t_saliency.device), reduction='sum') / (t_saliency!=0).sum()