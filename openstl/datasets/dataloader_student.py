import numpy as np
import os
from PIL import Image
import random
import torch

from torch.utils.data import Dataset
from torchvision import transforms as T

from openstl.datasets.utils import create_loader

class Student(Dataset):
    def __init__(self, root, is_train=True, n_frames_input=11, n_frames_output=11):
        super(Student, self).__init__()
        self.is_train = is_train
        self.root = root
        self.video_dir_lst = self._load_dataset_folder()
        print
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = T.Compose([T.ToTensor()])
    
    def _load_dataset_folder(self):
        video_dir_lst = []
        if self.is_train:
            labeled_videos = os.listdir(os.path.join(self.root, 'train'))

            for v in labeled_videos:
                if 'video' in v:
                    video_dir_lst.append(os.path.join(self.root, 'train', v))
                
            unlabeled_videos = os.listdir(os.path.join(self.root, 'unlabeled'))

            for v in unlabeled_videos:
                if 'video' in v:
                    video_dir_lst.append(os.path.join(self.root, 'unlabeled', v))
        else:
            videos = os.listdir(os.path.join(self.root, 'val'))
            for v in videos:
                video_dir_lst.append(os.path.join(self.root, 'val', v))
        return video_dir_lst

    def __getitem__(self, idx):
        video_dir = self.video_dir_lst[idx]
        X = []
        for i in range(self.n_frames_input):

            x = Image.open(os.path.join(video_dir, f'image_{i}.png'))
            x = self.transform(x)
            X.append(x)
        X = torch.stack(X, dim=0)

        Y = []
        for i in range(self.n_frames_input, self.n_frames_input+self.n_frames_output):

            y = Image.open(os.path.join(video_dir, f'image_{i}.png'))
            y = self.transform(y)
            Y.append(y)
        Y = torch.stack(Y, dim=0)
        
        
        return X, Y
        
    def __len__(self):
        return len(self.video_dir_lst)

def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=11, aft_seq_length=11, distributed=False):

    train_set = Student(root=data_root, is_train=True)
    test_set = Student(root=data_root, is_train=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers, distributed=distributed)
    
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed)
    
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    num_workers=num_workers, distributed=distributed)

    return dataloader_train, dataloader_vali, dataloader_test

if __name__ == '__main__':
    root = '/Users/new/Desktop/course/2023spring/DL/project/Dataset_Student'
    dataset = Student(root=root, is_train=True)
    for i, (x, y) in enumerate(dataset):
        print(i, x.shape, y.shape)