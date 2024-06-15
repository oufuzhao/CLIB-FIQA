import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from model import clip
from model.models import convert_weights, MLP
from tqdm import tqdm
import numpy as np
from utilities import *
from config_train import config as conf
from model.weight_methods import WeightMethods
import torch.nn.functional as F
from dataset.dataset_txt_list import load_data
from torch.nn.utils import clip_grad_norm_
from itertools import product
from model.losses import FocalLoss, Dist_distance
import random

quality_list = ['bad', 'poor', 'fair', 'good', 'perfect']
pose_list = ['profile', 'slight angle', 'frontal']
blur_list = ['hazy', 'blurry', 'clear']
occ_list = ['obstructed', 'unobstructed']
ill_list = ['extreme lighting', 'normal lighting']
exp_list = ['exaggerated expression', 'typical expression']

joint_texts = torch.cat([clip.tokenize(f"a photo of a {b}, {o}, and {p} face with {e} under {l}, which is of {q} quality") 
                for b, o, p, e, l, q in product(blur_list, occ_list, pose_list, exp_list, ill_list, quality_list)]).to(conf.device)

def dataSet(conf):
    train_loader = load_data(conf)
    return train_loader
    
def backboneSet(conf):
    net, _ = clip.load(conf.clip_model, device=conf.device, jit=False)
    return net


def trainSet(conf, net, batches, model_MLP):
    print('='*20 + 'LOSSES SETTING' + '='*20)
    print(f"LOSS TYPE")

    loss_foc = FocalLoss(gamma=2)
    loss_dist = Dist_distance
    criterion = [loss_dist, loss_foc]
    print(criterion)

    # Optimizer
    optimizer = optim.AdamW(net.parameters(),
                           lr=conf.lr,
                           weight_decay=conf.weight_decay)

    optimizer_mlp = optim.AdamW(model_MLP.parameters(),
                           lr=conf.lr * 100,
                           weight_decay=conf.weight_decay)

    print('='*20 + 'OPTIMIZER SETTING' + '='*20)
    print(optimizer)
    print(optimizer_mlp)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scheduler_mlp = optim.lr_scheduler.MultiStepLR(optimizer_mlp, milestones=[3, 4], gamma=0.1)

    weighting_method = WeightMethods(
            method='dwa',
            n_tasks=6,
            alpha=1.5,
            temp=2.0,
            n_train_batch=batches,
            n_epochs=conf.epoch,
            main_task=0,
            device=conf.device
        )
    return criterion, optimizer, scheduler, weighting_method, optimizer_mlp, scheduler_mlp

def do_batch(model, x, text):
    batch_size = x.size(0)
    x = x.view(-1, x.size(1), x.size(2), x.size(3))
    logits_per_image, logits_per_text = model.forward(x, text)
    logits_per_image = logits_per_image.view(batch_size, -1)
    logits_per_text = logits_per_text.view(-1, batch_size)
    logits_per_image = F.softmax(logits_per_image, dim=1)
    logits_per_text = F.softmax(logits_per_text, dim=1)
    return logits_per_image, logits_per_text

def train(model, trainloader, optimizer, scheduler, epoch, criterion, model_MLP, optimizer_mlp, scheduler_mlp):
    model.train()
    model_MLP.train()
    rec_losses_1 = AverageMeter()
    rec_losses_2 = AverageMeter()
    rec_losses_3 = AverageMeter()
    rec_losses_4 = AverageMeter()
    rec_losses_5 = AverageMeter()
    rec_losses_6 = AverageMeter()
    rec_losses_7 = AverageMeter()
    rec_losses_all = AverageMeter()

    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()

    if optimizer_mlp.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler_mlp.step()

    itersNum = 0
    for imgPath, data, qs, att_labels1, att_labels2, att_labels3, att_labels4, att_labels5 in tqdm(trainloader, desc=f"Epoch {epoch+1}", total=len(trainloader)):
        data = data.to(conf.device)
        qs = qs.to(conf.device)
        qs_batch = score_to_dist(qs)

        att_labels_blr = att_labels1.to(conf.device)
        att_labels_pos = att_labels2.to(conf.device)
        att_labels_exp = att_labels3.to(conf.device)
        att_labels_ill = att_labels4.to(conf.device)
        att_labels_occ = att_labels5.to(conf.device)

        logits_per_image, _ = do_batch(model, data, joint_texts)
        logits_per_image = logits_per_image.view(-1, len(blur_list), len(occ_list), len(pose_list), len(exp_list), len(ill_list), len(quality_list))

        logits_quality  = logits_per_image.sum(1).sum(1).sum(1).sum(1).sum(1)
        logits_blur     = logits_per_image.sum(6).sum(5).sum(4).sum(3).sum(2)
        logits_occ      = logits_per_image.sum(6).sum(5).sum(4).sum(3).sum(1)
        logits_pose     = logits_per_image.sum(6).sum(5).sum(4).sum(2).sum(1)
        logits_exp      = logits_per_image.sum(6).sum(5).sum(3).sum(2).sum(1)
        logits_ill      = logits_per_image.sum(6).sum(4).sum(3).sum(2).sum(1)
        logits_joint    = logits_per_image.sum(6).view(-1, len(blur_list) * len(occ_list) * len(pose_list) * len(exp_list) * len(ill_list)).detach()

        att_labels_blr = att_labels1.to(conf.device)
        att_labels_pos = att_labels2.to(conf.device)
        att_labels_exp = att_labels3.to(conf.device)
        att_labels_ill = att_labels4.to(conf.device)
        att_labels_occ = att_labels5.to(conf.device)
        logits_meta = model_MLP(logits_joint.float())

        loss1 = 10 * criterion[0](logits_quality, qs_batch)
        loss2 = criterion[1](logits_blur, att_labels_blr)
        loss3 = criterion[1](logits_occ, att_labels_occ)
        loss4 = criterion[1](logits_ill, att_labels_ill)
        loss5 = criterion[1](logits_pose, att_labels_pos)
        loss6 = criterion[1](logits_exp, att_labels_exp)
        total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        all_loss = [loss1, loss2, loss3, loss4, loss5, loss6]

        loss7 = criterion[0](logits_meta, qs_batch)
        rec_losses_1.update(loss1.detach().data.item(), data.size(0))
        rec_losses_2.update(loss2.detach().data.item(), data.size(0))
        rec_losses_3.update(loss3.detach().data.item(), data.size(0))
        rec_losses_4.update(loss4.detach().data.item(), data.size(0))
        rec_losses_5.update(loss5.detach().data.item(), data.size(0))
        rec_losses_6.update(loss6.detach().data.item(), data.size(0))
        rec_losses_7.update(loss7.detach().data.item(), data.size(0))
        rec_losses_all.update(total_loss.detach().data.item(), data.size(0))

        optimizer.zero_grad()
        optimizer_mlp.zero_grad()
        if not torch.isnan(total_loss):
            total_loss = weighting_method.backwards(
                all_loss,
                epoch=epoch,
                logsigmas=None,
                shared_parameters=None,
                last_shared_params=None,
                returns=True
            )
        else:
            total_loss.backward()
            continue
        loss7.backward()
        clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        clip_grad_norm_(model_MLP.parameters(), max_norm=5, norm_type=2)
        if conf.device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            convert_weights(model)
        optimizer_mlp.step()
        
        if itersNum % (int(len(trainloader)/50)+2) == 0:
            loss_item = f"Loss_q={rec_losses_1.avg:.5f} | Loss_blu={rec_losses_2.avg:.5f} | Loss_occ={rec_losses_3.avg:.5f} | Loss_lig={rec_losses_4.avg:.5f} | Loss_pos={rec_losses_5.avg:.5f} | Loss_exp={rec_losses_6.avg:.5f} | Loss_MLP={rec_losses_7.avg:.5f}"
            print(f"({epoch+1}Epo / {itersNum}Iter) [LR = {optimizer.param_groups[0]['lr']} | [{loss_item} ||| Loss_all={rec_losses_all.avg:.5f}]]")
        itersNum += 1

    return model, model_MLP

def obtain_confid(model, trainloader, model_MLP):
    dict_conf = {}
    itersNum = 0
    for imgPath, data, qs, _, _, _, _, _ in tqdm(trainloader):
        with torch.no_grad():
            data = data.to(conf.device)
            qs = qs.to(conf.device)
            logits_per_image, _ = do_batch(model, data, joint_texts)
            logits_per_image = logits_per_image.view(-1, len(blur_list), len(occ_list), len(pose_list), len(exp_list), len(ill_list), len(quality_list))
            logits_quality  = logits_per_image.sum(1).sum(1).sum(1).sum(1).sum(1)
            logits_joint    = logits_per_image.sum(6).view(-1, len(blur_list) * len(occ_list) * len(pose_list) * len(exp_list) * len(ill_list)).detach()
            logits_meta = model_MLP(logits_joint.float())
            dist = cal_distance(logits_quality.detach(), logits_meta.detach())
            dist_npy = dist.cpu().data.numpy()
            for i, v in enumerate(imgPath): dict_conf[v] = str(dist_npy[i])
    return dict_conf


def train_clib(model, trainloader, optimizer, scheduler, epoch, criterion, dict_conf):
    model.train()
    rec_losses_1 = AverageMeter()
    rec_losses_2 = AverageMeter()
    rec_losses_3 = AverageMeter()
    rec_losses_4 = AverageMeter()
    rec_losses_5 = AverageMeter()
    rec_losses_6 = AverageMeter()
    rec_losses_all = AverageMeter()

    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()

    itersNum = 0
    for imgPath, data, qs, att_labels1, att_labels2, att_labels3, att_labels4, att_labels5 in tqdm(trainloader, desc=f"Epoch {epoch+1}", total=len(trainloader)):
        data = data.to(conf.device)
        qs = qs.to(conf.device)
        cond_list = [float(dict_conf[i]) for i in imgPath]
        cond_list_npy = np.asarray(cond_list)
        cond_tensor = torch.from_numpy(cond_list_npy).to('cuda')
        qs_confid = score_to_dist_confid(qs, cond_tensor)

        att_labels_blr = att_labels1.to(conf.device)
        att_labels_pos = att_labels2.to(conf.device)
        att_labels_exp = att_labels3.to(conf.device)
        att_labels_ill = att_labels4.to(conf.device)
        att_labels_occ = att_labels5.to(conf.device)

        logits_per_image, _ = do_batch(model, data, joint_texts)
        logits_per_image = logits_per_image.view(-1, len(blur_list), len(occ_list), len(pose_list), len(exp_list), len(ill_list), len(quality_list))
        logits_quality  = logits_per_image.sum(1).sum(1).sum(1).sum(1).sum(1)
        logits_blur     = logits_per_image.sum(6).sum(5).sum(4).sum(3).sum(2)
        logits_occ      = logits_per_image.sum(6).sum(5).sum(4).sum(3).sum(1)
        logits_pose     = logits_per_image.sum(6).sum(5).sum(4).sum(2).sum(1)
        logits_exp      = logits_per_image.sum(6).sum(5).sum(3).sum(2).sum(1)
        logits_ill      = logits_per_image.sum(6).sum(4).sum(3).sum(2).sum(1)

        att_labels_blr = att_labels1.to(conf.device)
        att_labels_pos = att_labels2.to(conf.device)
        att_labels_exp = att_labels3.to(conf.device)
        att_labels_ill = att_labels4.to(conf.device)
        att_labels_occ = att_labels5.to(conf.device)
        
        loss1 = 10 * criterion[0](logits_quality, qs_confid)
        loss2 = criterion[1](logits_blur, att_labels_blr)
        loss3 = criterion[1](logits_occ, att_labels_occ)
        loss4 = criterion[1](logits_ill, att_labels_ill)
        loss5 = criterion[1](logits_pose, att_labels_pos)
        loss6 = criterion[1](logits_exp, att_labels_exp)
        total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        all_loss = [loss1, loss2, loss3, loss4, loss5, loss6]

        rec_losses_1.update(loss1.detach().data.item(), data.size(0))
        rec_losses_2.update(loss2.detach().data.item(), data.size(0))
        rec_losses_3.update(loss3.detach().data.item(), data.size(0))
        rec_losses_4.update(loss4.detach().data.item(), data.size(0))
        rec_losses_5.update(loss5.detach().data.item(), data.size(0))
        rec_losses_6.update(loss6.detach().data.item(), data.size(0))
        rec_losses_all.update(total_loss.detach().data.item(), data.size(0))

        optimizer.zero_grad()
        if not torch.isnan(total_loss):
            total_loss = weighting_method.backwards(
                all_loss,
                epoch=epoch,
                logsigmas=None,
                shared_parameters=None,
                last_shared_params=None,
                returns=True
            )
        else:
            total_loss.backward()
            continue
        clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        if conf.device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            convert_weights(model)
    
        if itersNum % (int(len(trainloader)/50)+2) == 0:
            loss_item = f"Loss_q={rec_losses_1.avg:.5f} | Loss_blu={rec_losses_2.avg:.5f} | Loss_occ={rec_losses_3.avg:.5f} | Loss_lig={rec_losses_4.avg:.5f} | Loss_pos={rec_losses_5.avg:.5f} | Loss_exp={rec_losses_6.avg:.5f}" #| Loss_MLP={rec_losses_7.avg:.5f}"
            print(f"({epoch+1}Epo / {itersNum}Iter) [LR = {optimizer.param_groups[0]['lr']} | [{loss_item} ||| Loss_all={rec_losses_all.avg:.5f}]]")
        itersNum += 1

    return model



if __name__ == "__main__":
    print_conf(conf)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(conf.seed)

    model = backboneSet(conf)
    model_MLP = MLP().to('cuda')

    # Froze the parameter of textual encoder
    for name, param in model.named_parameters():
        if 'transformer' in name or 'positional_embedding' in name or 'text_projection' in name:
            param.requires_grad=False

    trainloader = dataSet(conf)
    criterion, optimizer, scheduler, weighting_method, optimizer_mlp, scheduler_mlp = trainSet(conf, model, len(trainloader), model_MLP)

    print('='*20 + 'Training' + '='*20)
    for epoch in range(0, conf.Epo_th):
        model, model_MLP = train(model, trainloader, optimizer, scheduler, epoch, criterion, model_MLP, optimizer_mlp, scheduler_mlp)
        scheduler.step()
        scheduler_mlp.step()

    dict_conf = obtain_confid(model, trainloader, model_MLP)

    for epoch in range(conf.Epo_th, conf.epoch):
        model = train_clib(model, trainloader, optimizer, scheduler, epoch, criterion, dict_conf)
        scheduler.step()
        if (epoch+1)%conf.saveModel_epoch==0:
            os.makedirs(conf.checkpoints, exist_ok=True)
            savePath = os.path.join(conf.checkpoints, f"CLIB_Epo{epoch+1}.pth")
            torch.save(model.state_dict(), savePath)               
            print(f'Saving model at {conf.checkpoints}')
