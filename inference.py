import os
import torch
from model import clip
from model.models import convert_weights
import numpy as np
from utilities import *

from itertools import product
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

quality_list = ['bad', 'poor', 'fair', 'good', 'perfect']
blur_list = ['hazy', 'blurry', 'clear']
occ_list = ['obstructed', 'unobstructed']
pose_list = ['profile', 'slight angle', 'frontal']
exp_list = ['exaggerated expression', 'typical expression']
ill_list = ['extreme lighting', 'normal lighting']
joint_texts = torch.cat([clip.tokenize(f"a photo of a {b}, {o}, and {p} face with {e} under {l}, which is of {q} quality") 
                for b, o, p, e, l, q in product(blur_list, occ_list, pose_list, exp_list, ill_list, quality_list)]).cuda()

pose_map = {0:pose_list[0], 1:pose_list[1], 2:pose_list[2]}
blur_map = {0:blur_list[0], 1:blur_list[1], 2:blur_list[2]}
occ_map  = {0:occ_list[0],  1:occ_list[1]}
ill_map  = {0:ill_list[0],  1:ill_list[1]}
exp_map =  {0:exp_list[0],  1:exp_list[1]}

def img_tensor(imgPath):
    img = Image.open(imgPath).convert("RGB")
    transform = T.Compose([
                T.Resize([224, 224]),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                ])
    img_tensor = transform(img)
    data = img_tensor.unsqueeze(dim=0)
    return data

def backboneSet(clip_model):
    net, _ = clip.load(clip_model, device='cuda', jit=False)
    return net

@torch.no_grad()
def do_batch(model, x, text):
    batch_size = x.size(0)
    x = x.view(-1, x.size(1), x.size(2), x.size(3))
    logits_per_image, logits_per_text = model.forward(x, text)
    logits_per_image = logits_per_image.view(batch_size, -1)
    logits_per_text = logits_per_text.view(-1, batch_size)
    logits_per_image = F.softmax(logits_per_image, dim=1)
    logits_per_text = F.softmax(logits_per_text, dim=1)
    return logits_per_image, logits_per_text

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    clip_model = "./weights/RN50.pt"
    clip_weights = "./weights/CLIB-FIQA_R50.pth"
    image_path = "./samples/1.jpg"

    model = backboneSet(clip_model)
    model = load_net_param(model, clip_weights)
    tensor_data = img_tensor(image_path).cuda()
    logits_per_image, _, = do_batch(model, tensor_data, joint_texts)
    logits_per_image = logits_per_image.view(-1, len(blur_list), len(occ_list), len(pose_list), len(exp_list), len(ill_list), len(quality_list))
    logits_quality  = logits_per_image.sum(1).sum(1).sum(1).sum(1).sum(1)
    logits_blur     = torch.max(logits_per_image.sum(6).sum(5).sum(4).sum(3).sum(2), dim=1)[1].cpu().detach().numpy().squeeze(0)
    logits_occ      = torch.max(logits_per_image.sum(6).sum(5).sum(4).sum(3).sum(1), dim=1)[1].cpu().detach().numpy().squeeze(0)
    logits_pose     = torch.max(logits_per_image.sum(6).sum(5).sum(4).sum(2).sum(1), dim=1)[1].cpu().detach().numpy().squeeze(0)
    logits_exp      = torch.max(logits_per_image.sum(6).sum(5).sum(3).sum(2).sum(1), dim=1)[1].cpu().detach().numpy().squeeze(0)
    logits_ill      = torch.max(logits_per_image.sum(6).sum(4).sum(3).sum(2).sum(1), dim=1)[1].cpu().detach().numpy().squeeze(0)
    quality_preds = dist_to_score(logits_quality).cpu().detach().numpy().squeeze(0)

    output_msg = f"a photo of a [{blur_map[int(logits_blur)]}], [{occ_map[int(logits_occ)]}], and [{pose_map[int(logits_pose)]}] face with [{exp_map[int(logits_exp)]}] under [{ill_map[int(logits_ill)]}]"
    print(output_msg)
    print(f"quality prediction = {quality_preds}")
