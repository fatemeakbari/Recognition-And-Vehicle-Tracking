
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
from skimage.transform import resize
import pandas as pds
from torchvision import transforms
import torch
import numpy as np
import cv2

def detection_collate(batch):

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
class Rescale():

    def __init__(self, out_size):
        self.out_size = out_size

    def __call__(self,sample):

        img = sample[0]
        labels = sample[1]
        h,w = img.shape[:2]

        img = resize(img, (self.out_size,self.out_size),mode='constant')
        labels[:,0] = labels[:,0]*self.out_size/w
        labels[:,1] = labels[:,1]*self.out_size/h
        labels[:,2] = labels[:,2]*self.out_size/w
        labels[:,3] = labels[:,3]*self.out_size/h

        return img, labels
import numpy as np
class ToTensor(object):
    def __call__(self, sample):
        img = torch.from_numpy(sample[0].transpose(2, 0, 1))
        labels = sample[1]
        return img, labels

class Transform(Dataset):

    def __init__(self, imgs_path, csv_path=None, transform=None, name="custom"):

        self.imgs_path = imgs_path
        self.labels = pds.read_csv(csv_path)
        self.transform = transform
        self.name = name

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #index or image name

        img_name = self.labels.iloc[index,0]

        label = self.labels.iloc[index, 1:].values.astype('float')
        label = label[~np.isnan(label)].reshape(-1,4) #delete Nan value from label label.shape=[num_objs, 4]

        img_path = os.path.join(self.imgs_path,img_name)
        img = io.imread(img_path)

        height,width,channels= img.shape

        label[:,0] /= width
        label[:,2] /= width
        label[:,1] /= height
        label[:,3] /= height

        new_label = np.ones((label.shape[0],5))
        new_label[:,:-1] = label

        #sample = [img,new_label]

        if self.transform:
          
            img, boxes, labs = self.transform(img, new_label[:,:4], new_label[:,4])
            boxes = new_label[:,:4]
            labs = new_label[:,4]
        img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
        target = np.hstack((boxes, np.expand_dims(labs, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target
