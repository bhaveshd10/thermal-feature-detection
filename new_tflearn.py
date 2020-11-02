import torch
import numpy as np
from torch.autograd.variable import Variable
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data
from new_dataloadertflearn import Data_loader
import os
from new_tflearngetpatches import patches


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)
npatch_train = 210
npatch_test = 10
batch_size = 1
workers = 0

rgb_path_train = 'C:/Users/bhave/PycharmProjects/thesis/kaist/train/new selected trainA/'
thermal_path_train = 'C:/Users/bhave/PycharmProjects/thesis/kaist/train/new selected trainB/'
rgb_path_test = 'C:/Users/bhave/PycharmProjects/thesis/kaist/test/a/'
thermal_path_test = 'C:/Users/bhave/PycharmProjects/thesis/kaist/test/b/'

rgb_train = os.listdir('C:/Users/bhave/PycharmProjects/thesis/kaist/train/new selected trainA/')
thermal_train = os.listdir('C:/Users/bhave/PycharmProjects/thesis/kaist/train/new selected trainB/')
rgb_test = os.listdir('C:/Users/bhave/PycharmProjects/thesis/kaist/test/a/')
thermal_test = os.listdir('C:/Users/bhave/PycharmProjects/thesis/kaist/test/b/')

getpatches = patches(rgb_path_train,thermal_path_train,rgb_train,thermal_train,npatch_train)
matches_train,gt_train = getpatches.getpatches()
print(len(matches_train))
getpatches = patches(rgb_path_test,thermal_path_test,rgb_test,thermal_test,npatch_test)
matches_test,gt_test = getpatches.getpatches()
print(len(matches_test))


transform = transforms.Compose([transforms.ToTensor()
                                   , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

dataset_train = Data_loader(matches_train, transform=transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train, num_workers=workers, batch_size=batch_size, shuffle=False)
print('Len dataloader train:' + str(len(dataloader_train)))

dataset_test = Data_loader(matches_test, transform=transform)

dataloader_test = torch.utils.data.DataLoader(dataset_test, num_workers=workers, batch_size=batch_size, shuffle=False)
print('Len dataloader test:' + str(len(dataloader_test)))

model = torch.load('C:/Users/bhave/PycharmProjects/thesis/new_train/new_train_70.pth')

for param in model.parameters():
    param.requires_grad = False

lin = model.fc2
new_lin = torch.nn.Sequential(lin, torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
model.fc2 = new_lin
print(model)

model.to(device)

criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
acc_train = []
acc_test = []


def accuracy_(output,gt):
    if gt == 1 and output >= 0.7:
        return 1
    elif gt == 0 and output < 0.7:
        return 1
    else:
        return 0


def train(model,dataloader,criterion,optimizer,target):

    running_loss, correct = 0.0, 0
    i = 0
    for batch_idx, data in enumerate(dataloader):

        d,gt = data
        d, gt = Variable(d) , Variable(gt)
        output = model(d.to(device))

        y = torch.tensor([[gt]], dtype = torch.float32).to(device)
        y = Variable(y)
        i+=1

        loss = criterion(output, y)
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += accuracy_(output,y)

    return running_loss, correct

def test(model,dataloader,criterion,optimizer,target):
    model.eval()
    with torch.no_grad():
        running_loss, correct = 0.0, 0
        i = 0
        for batch_idx, data in enumerate(dataloader):

            d, gt = data
            d, gt = Variable(d), Variable(gt)
            output = model(d.to(device))

            y = torch.tensor([target[i]], dtype = torch.float32).to(device)
            # print((output.item(),y))
            y = Variable(y)
            i+=1

            loss = criterion(output, y)
            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += accuracy_(output,y)

        return running_loss, correct

savepath = 'C:/Users/bhave/PycharmProjects/thesis/new train'
if not os.path.exists(savepath):
    os.makedirs(savepath)

for epoch in range(1, 30 + 1):
    print(f"Epoch {epoch}/{30}")
    loss_train, accuracy_train = train(model,dataloader_train,criterion,optimizer,gt_train)
    acc_train.append(accuracy_train)
    print(f"  Train Loss: {loss_train / len(dataloader_train)}")
    # acc_train.append(loss_acc_train[1])
    print(f"  Train Acc: {accuracy_train / len(dataloader_train)}")

    loss_test, accuracy_test = train(model,dataloader_test,criterion,optimizer,gt_test)
    acc_test.append(accuracy_test)
    print(f"  Train Loss: {loss_test / len(dataloader_test)}")
    # acc_train.append(loss_acc_train[1])
    print(f"  Train Acc: {accuracy_test / len(dataloader_test)}")

    if epoch % 10 == 0:
        torch.save(model, os.path.join(savepath, "new_tflearn"+'_'+ str(epoch) + ".pth"))


print('xxxxxxxxxxx finished  xxxxxxxxxxx')

