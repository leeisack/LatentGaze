import os
import PIL
import torch

from glob import glob
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
import pickle

class MPII(torch.utils.data.Dataset):
    def __init__(self, dir_name, transforms=pil_to_tensor):

        self.root_dir = os.path.join("./dataset", dir_name)
        self.imgs = os.listdir(self.root_dir)
        self.transform = transforms
        self.data = []
        self.label = []
        
        if dir_name == "MPII_train":
            # self.latent = "MPII_latent_train"
            # self.latent = "D:\\download\\resized_mpii_pt_eth_encoder\\latent_pt_files"
            self.latent = "D:\\download\\mpii_latent_pt_files_with_mpii\\latent_pt_files"
        elif dir_name == "MPII_validation":
            # self.latent = "MPII_latent_validation"
            # self.latent = "D:\\download\\resized_mpii_pt_eth_encoder\\latent_pt_files"
            # self.latent = "D:\\download\\mpii_latent_pt_files_with_mpii\\latent_pt_files"
            self.latent = "D:\\Gaze_estimator_implementation\\dataset\\test1\\latent_pt_files"

        for i, img in enumerate(self.imgs):
            img_path = os.path.join(self.root_dir, img)
            self.data.append(img_path)
            self.label.append(i)
            
        with open("./dataset/MPII_label_dict.pickle", "rb") as f:
            label_dict = pickle.load(f)
        self.label = label_dict

    def __getitem__(self, idx):
    
        img_path = self.data[idx]
        img_name = img_path.split('\\')[-1]
        img_name = img_name.split('.')[0]
        label = self.label[img_name]
 

        latent_name = os.path.splitext(img_name)[0] + ".pt"

       
        latent_path = os.path.join(self.latent,latent_name)
        latent = torch.load(latent_path)
        latent = latent.type('torch.FloatTensor')
        
        img_name = os.path.basename(img_path)
        img = PIL.Image.open(img_path)
        img = self.transform(img)

        sample = {"image" : img, "label" : label, "name" : img_name, 'latent' : latent}

        return sample

    def __len__(self):
        return len(self.data)


    