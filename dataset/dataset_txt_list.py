import torch
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import torchvision.transforms as T
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, conf):
        super().__init__()
        self.img_list = conf.data_list
        self.transform = T.Compose([
                         T.RandomHorizontalFlip(),
                         T.Resize([224, 224]),
                         T.ToTensor(),
                         T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        self.batch_size = conf.batch_size

        with open(self.img_list, 'r') as f:
            self.imgPath = []
            self.qs = []
            self.id = []
            self.qs = []
            self.label1, self.label2, self.label3, self.label4, self.label5 = [], [], [], [], []
            for index, value in enumerate(f):
                value = value.split()
                self.imgPath.append(value[0])
                self.qs.append(float(value[1]))
                self.label1.append(int(value[2]))
                self.label2.append(int(value[3]))
                self.label3.append(int(value[4]))
                self.label4.append(int(value[5]))
                self.label5.append(int(value[6]))
            self.qs = np.asarray(self.qs)
            self.qs = (self.qs - np.min(self.qs)) / (np.max(self.qs) - np.min(self.qs))

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        qs = self.qs[index]
        label1 = self.label1[index]
        label2 = self.label2[index]
        label3 = self.label3[index]
        label4 = self.label4[index]
        label5 = self.label5[index]
        img = Image.open(imgPath).convert("RGB")
        assert img.size[0] == 112
        data = self.transform(img)
        return imgPath, data, qs, label1, label2, label3, label4, label5

    def __len__(self):
        return(len(self.imgPath))
                
def load_data(conf):
    dataset = Dataset(conf)
    batch_size = conf.batch_size
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        pin_memory=conf.pin_memory, 
                        num_workers=conf.num_workers)
    return loader

