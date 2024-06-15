import os
import torch
from torch.nn.functional import kl_div, softmax, log_softmax

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

def get_image_paths(directory):
    image_paths = []
    people_set = set()
    id_dict = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                people_set.add(image_path.split('/')[-2])
    peop_list = sorted(list(people_set), key=lambda x: x.lower())
    for i, v in enumerate(peop_list): id_dict[v] = i
    peop_num = len(peop_list)
    return image_paths, id_dict, peop_num

def load_net_param(net, weight_path):
    net_dict = net.state_dict()
    pretrained_dict = torch.load(weight_path, map_location='cuda')
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    same_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    net_dict.update(same_dict)
    net.load_state_dict(net_dict)
    return net

def dist_to_score(x):
    anchor_bins = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).cuda()
    anchor_bins = anchor_bins.repeat(x.size(0), 1).cuda()
    norm_scores = x * anchor_bins
    one_scores = torch.sum(norm_scores, dim=1).cuda()
    return one_scores

def score_to_dist(x):
    x = torch.reshape(x, [-1,1])
    anchor_bins = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).cuda()
    anchor_bins = anchor_bins.repeat(x.size(0), 1).cuda()
    beta = torch.tensor(-32).cuda()
    dist_anchors = torch.exp(beta * torch.square(x - anchor_bins)).cuda()
    norm_dist_anchors = dist_anchors / torch.reshape(torch.sum(dist_anchors, dim=1), [-1, 1]).cuda()
    return norm_dist_anchors

def score_to_dist_confid(x, confid):
    x = torch.reshape(x, [-1,1])
    confid = confid.cuda()
    anchor_bins = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).cuda()
    anchor_bins = anchor_bins.repeat(x.size(0), 1).cuda()
    beta = torch.tensor(-32).cuda()
    confid_zoom = 1 / (1 + torch.e**(32*confid)) + 0.5
    beta_confid = (confid_zoom * beta).view(-1, 1)
    dist_anchors = torch.exp(beta_confid * torch.square(x - anchor_bins)).cuda()
    norm_dist_anchors = dist_anchors / torch.reshape(torch.sum(dist_anchors, dim=1), [-1, 1]).cuda()
    return norm_dist_anchors

def cal_distance(x1, x2):
    bs = x1.size(0)
    x1_softmax = softmax(x1, dim=1)
    x2_softmax = softmax(x2, dim=1)
    m = 0.5 * (x1_softmax + x2_softmax)
    kl_p_m = kl_div(log_softmax(x1, dim=1), m, reduction='none').sum(dim=1)
    kl_q_m = kl_div(log_softmax(x2, dim=1), m, reduction='none').sum(dim=1)
    js_dist = 0.5 * (kl_p_m + kl_q_m)
    return js_dist

def print_conf(conf):
    obj_list = []
    for item in dir(conf): 
        if '__' not in item: obj_list.append(item)
    print('='*20 + "CONFIG" + '='*20)
    for i in obj_list:
        i_value = eval(f"conf.{i}")
        print(f"{i}: {i_value}")
    
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
