from torchvision import datasets, transforms, utils
from torch.utils.data import random_split, DataLoader
import torch
from torch import nn, optim
from resnet import *
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_set_folder = datasets.ImageFolder("D:\\ML\\数据集\\classes", transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
                                        )
valid_size = int(len(data_set_folder.imgs) * 0.1)
train_size = len(data_set_folder.imgs) - valid_size
train_set, valid_set = random_split(data_set_folder, [train_size, valid_size])
batch_size = 16
train_set = DataLoader(train_set, batch_size = batch_size, shuffle = True)
valid_set = DataLoader(valid_set, batch_size = batch_size, shuffle = True)

def main():
    model = resnet20(num_classes=2).to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_acc = 0
    for epoch in range(1000):
        model.train()
        loop = tqdm(train_set)
        total_correct, total_num = 0, 0
        for x, label in loop:
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
            acc = total_correct / total_num
            loop.set_description('TRAIN loss:%.5f acc:%.5f' %(loss.item(), acc))

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            loop = tqdm(valid_set)
            for x, label in loop:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                loss = criteon(logits, label)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

                acc = total_correct / total_num
                loop.set_description('TEST loss:%.5f acc:%.5f' %(loss.item(), acc))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'model/%.5f.pkl' % acc)
if __name__ == '__main__':
    main()
