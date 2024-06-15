import torch
import torchvision.transforms as T

class Config(object):
# training dataset
    data_list = f"./"
    clip_model = './weights/RN50.pt'
# save settings
    checkpoints = f"./checkpoints"   
# training settings
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 323
    pin_memory = True
    num_workers = 12
    batch_size = 256         
    epoch = 25
    lr = 1e-5
    weight_decay = 0.001
    Epo_th = 5
    saveModel_epoch = 5

config = Config()
