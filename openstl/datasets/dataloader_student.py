import numpy as np
import os
from PIL import Image
import random
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T

from openstl.datasets.utils import create_loader

class Student(Dataset):
    def __init__(self, root, is_train=True, is_test=False, n_frames_input=11, n_frames_output=11):
        super(Student, self).__init__()
        self.is_train = is_train
        self.is_test = is_test
        self.root = root
        self.video_dir_lst = self._load_dataset_folder()
        print
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = T.Compose([T.ToTensor()])
        self.mean = 0
        self.std = 1
    
    def _load_dataset_folder(self):
        corrupt = ['video_13688', 'video_14125', 'video_13268', 'video_13534']
        video_dir_lst = []
        if self.is_train:
            labeled_videos = os.listdir(os.path.join(self.root, 'train'))

            for v in labeled_videos:
                if 'video' in v:
                    video_dir_lst.append(os.path.join(self.root, 'train', v))
                
            unlabeled_videos = os.listdir(os.path.join(self.root, 'unlabeled'))

            for v in unlabeled_videos:
                if 'video' in v and v not in corrupt:
                    video_dir_lst.append(os.path.join(self.root, 'unlabeled', v))
        else:
            if not self.is_test:
                # videos = os.listdir(os.path.join(self.root, 'val'))
                # for v in videos:
                #     video_dir_lst.append(os.path.join(self.root, 'val', v))
                for i in range(1000, 2000):
                    video_dir_lst.append(os.path.join(self.root, 'val', 'video_'+str(i)))
            else:
                
                for i in range(15000, 17000):
                    video_dir_lst.append(os.path.join(self.root, 'hidden', 'video_'+str(i)))
                
                # videos = os.listdir(os.path.join(self.root, 'hidden'))
                # for v in videos:
                #     video_dir_lst.append(os.path.join(self.root, 'hidden', v))

        return video_dir_lst

    def __getitem__(self, idx):
        video_dir = self.video_dir_lst[idx]
        X = []
        for i in range(self.n_frames_input):

            # x = Image.open(os.path.join(video_dir, f'image_{i}.png'))
            x = cv2.imread(os.path.join(video_dir, f'image_{i}.png'))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = self.transform(x)
            # x = torch.permute(x, (1, 2, 0))
            X.append(x)
        X = torch.stack(X, dim=0)

        Y = []
        if not self.is_test:
            for i in range(self.n_frames_input, self.n_frames_input+self.n_frames_output):

                # y = Image.open(os.path.join(video_dir, f'image_{i}.png'))
                # y = self.transform(y)
                y = cv2.imread(os.path.join(video_dir, f'image_{i}.png'))
                y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
                y = self.transform(y)
                # y = torch.permute(y, (1, 2, 0))
                Y.append(y)
            Y = torch.stack(Y, dim=0)
        else:
            Y = torch.zeros_like(X)
        return X, Y
        
    def __len__(self):
        return len(self.video_dir_lst)

def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=11, aft_seq_length=11, distributed=False):

    train_set = Student(root=data_root, is_train=True)
    val_set = Student(root=data_root, is_train=False)
    test_set = Student(root=data_root, is_train=False, is_test=True)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers, distributed=distributed)
    
    dataloader_vali = create_loader(val_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed)
    
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed)

    return dataloader_train, dataloader_vali, dataloader_vali

