import os
import cv2
import glob
import torch
import config
import numpy as np

from torch.utils.data import Dataset


class SobelDataset(Dataset):    
    def __init__(self, split: str, device: str, transform=None) -> None:        
        self.transform = transform
        self.device = device
        cwd = os.getcwd()
        self.input_paths = sorted(glob.glob(os.path.join(cwd, "custom_data", split, "input/*")))[:config.DATA_SIZE]
        self.output_paths = sorted(glob.glob(os.path.join(cwd, "custom_data", split, "output/*")))[:config.DATA_SIZE]    

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.input_paths)
    
    def __getitem__(self, index: int) -> tuple:
        "Returns one sample of data, input and output image (X, y)."
        input_img = cv2.imread(self.input_paths[index], 0)
        input_img = (input_img / 255.0).astype(np.float32)
        input_img = torch.from_numpy(input_img).unsqueeze(0).to(self.device)

        output_img = cv2.imread(self.output_paths[index], 0)
        output_img = (output_img / 255.0).astype(np.float32)
        output_img = torch.from_numpy(output_img).unsqueeze(0).to(self.device)

        # Transform if necessary
        if self.transform:
            return self.transform(input_img), self.transform(output_img)
        else:
            return input_img, output_img # return tuple