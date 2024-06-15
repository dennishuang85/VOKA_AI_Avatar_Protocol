from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
import torch
from torch import nn, optim
from resnet import *
from tqdm import tqdm
from shutil import copyfile
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
PngImagePlugin.MAX_TEXT_MEMORY = 100 * (1024**3)
import os

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_set_folder = ImageFolderWithPaths("D:\\ML\\数据集\\classes", transform = transforms.Compose([
            transforms.ToTensor()]))
batch_size = 32
data_set = DataLoader(data_set_folder, batch_size = batch_size, shuffle = True)

def main():
    model = resnet56(num_classes=2).to(device)
    model.load_state_dict(torch.load("model\\0.83186.pkl"))

    model.eval()
    with torch.no_grad():
        # test
        total_good = 0
        total_bad = 0
        loop = tqdm(data_set)
        tp, fp, fn, tn = 0, 0, 0, 0
        for x, label, path in loop:
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            # [b, 10]
            logits = model(x)
            # [b]
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            for i in range(len(path)):
                name = os.path.basename(path[i])
                if pred[i].item() == 0 and label[i].item() == 0:
                    tn += 1
                if pred[i].item() == 1 and label[i].item() == 1:
                    tp += 1
                if pred[i].item() == 0 and label[i].item() == 1:
                    fp += 1
                if pred[i].item() == 1 and label[i].item() == 0:
                    fn += 1
            # print(correct)
            loop.set_description('TEST pre:%.4f rec:%.4f' %(1.0 * tp / (tp + fp), 1.0 * tp / (tp + fn)))
if __name__ == '__main__':
    main()
