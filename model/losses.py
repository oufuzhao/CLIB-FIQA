import torch
from torch import nn

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def Dist_distance(p, q, r=2):
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    cdf_p = torch.cumsum(p, dim=1)
    cdf_q = torch.cumsum(q, dim=1)
    cdf_diff = torch.abs(cdf_p - cdf_q)
    cdf_diff = torch.mean(cdf_diff ** r, dim=1)
    single_dist = cdf_diff ** (1. / r)
    return torch.mean(single_dist)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class CR_FIQA_LOSS(nn.Module):
    r"""Implement of ArcFace with Dynamic Quality-Based Margin Scaling:
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
            adaptive_margin: boolean to enable dynamic margin scaling based on sample difficulty
        """

    def __init__(self, in_features, out_features, s=64.0, m=0.50, adaptive_margin=True):
        super(CR_FIQA_LOSS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.adaptive_margin = adaptive_margin
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        
        with torch.no_grad():
           distmat=cos_theta[index,label.view(-1)].detach().clone()
           max_negative_cloned=cos_theta.detach().clone()
           max_negative_cloned[index,label.view(-1)]= -1e-12
           max_negative, _=max_negative_cloned.max(dim=1)
           
           # --- NEW FEATURE: Quality-Based Margin Scaling ---
           # Calculate sample difficulty: higher margins for harder/low-quality samples
           if self.adaptive_margin:
               # Margin multiplier decreases if positive similarity is high and negative is low
               quality_ratio = (max_negative[index] / (distmat + 1e-5)).clamp(0.1, 2.0)
               dynamic_m = self.m * quality_ratio.unsqueeze(1)
           else:
               dynamic_m = self.m

        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], dynamic_m)
        
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta, 0 ,distmat[index,None],max_negative[index,None]
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(CR_FIQA_LOSS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        with torch.no_grad():
           distmat=cos_theta[index,label.view(-1)].detach().clone()
           max_negative_cloned=cos_theta.detach().clone()
           max_negative_cloned[index,label.view(-1)]= -1e-12
           max_negative, _=max_negative_cloned.max(dim=1)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta, 0 ,distmat[index,None],max_negative[index,None]
