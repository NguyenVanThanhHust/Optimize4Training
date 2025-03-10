import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MiniImageNetDataset(Dataset):
    def __init__(self, dataset_dir, ds_type="train", transform=None):
        self.dataset_dir = dataset_dir
        assert os.path.isdir(self.dataset_dir), f"This is not a folder {self.dataset_dir}"
        self.transform = transform
        folders = next(os.walk(self.dataset_dir))[1]
        self.classes = {}
        if ds_type == "train":
            indices = [0, ]
        self.data_info = []
        for idx, folder in enumerate(folders):
            self.classes[idx] = folder
            files = next(os.walk(os.path.join(self.dataset_dir, folder)))[2]
            for idx, file in enumerate(files):               
                if ds_type == "train":
                    if idx >= 60:
                        continue
                elif ds_type == "val":
                    if idx <60 or idx >=80:
                        continue
                else:
                    if idx < 80:
                        continue
                fp = os.path.join(self.dataset_dir, folder, file)
                self.data_info.append((fp, idx))
            
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        im_path, label = self.data_info[index]
        im = Image.open(im_path).convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        label = torch.tensor(label, dtype=torch.long)
        return im, label

def build_transform():
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
        transforms.RandomCrop((224, 224)),  # Resize the image to 256x256 pixels
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with a probability of 0.5
        transforms.ToTensor()  # Convert the image to a tensor
    ])
    return train_transforms
    
if __name__ == '__main__':
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
        transforms.RandomCrop((224, 224)),  # Resize the image to 256x256 pixels
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with a probability of 0.5
        transforms.ToTensor()  # Convert the image to a tensor
    ])
    mini_imagenet_dataset = MiniImageNetDataset(dataset_dir='../../Datasets/mini_imagenet', transform=train_transforms)
    train_dataloader = DataLoader(mini_imagenet_dataset, batch_size=2)

    for img, label in train_dataloader:
        print(img.shape)
        print(label)
        break