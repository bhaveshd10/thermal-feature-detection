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
parser = argparse.ArgumentParser()
parser.add_argument('rgb_path_train', action="store")
parser.add_argument('thermal_path_train', action="store")
parser.add_argument('rgb_path_test', action="store")
parser.add_argument('thermal_path_test', action="store")
parser.add_argument('npatch_train', action="store")
parser.add_argument('npatch_test', action="store")
parser.add_argument('batch_size', action="store")
parser.add_argument('workers', action="store")
parser.add_argument('save_path', action="store")
parser.add_argument('model_path', action="store")

npatch_train = args.npatch_train
npatch_test = args.npatch_test
batch_size = args.batch_size
workers = args.workers

# Get Image patches
getpatches = patches(args.rgb_path_train,args.thermal_path_train,os.listdir(args.rgb_path_train),os.listdir(args.thermal_path_train),npatch_train)
matches_train,descript_train = getpatches.getpatches()
print(len(matches_train))
getpatches = patches(args.rgb_path_test,args.thermal_path_test,os.listdir(args.rgb_path_test),os.listdir(args.thermal_path_test),npatch_test)
matches_test,descript_test = getpatches.getpatches()
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

# Load pretrained model
model = torch.load(args.model_path)

for param in model.parameters():
    param.requires_grad = False

# Additional layers added to pretrained model for transfer learning
lin = model.fc2
new_lin = torch.nn.Sequential(lin, torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
model.fc2 = new_lin
print(model)

model.to(device)

# Define criterion and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
acc_train = []
acc_test = []

# Accuracy calculations
def accuracy_(output,gt):
    if gt == 1 and output >= 0.7:
        return 1
    elif gt == 0 and output < 0.7:
        return 1
    else:
        return 0

# Train function
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

# Eval Function
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

# Save results
savepath = os.path.join(args.save_path, directory)
if not os.path.exists(savepath):
    os.makedirs(savepath)

# Train/Test model overtime
for epoch in range(1, 30 + 1):
    print(f"Epoch {epoch}/{30}")
    loss_train, accuracy_train = train(model,dataloader_train,criterion,optimizer,gt_train)
    acc_train.append(accuracy_train)
    print(f"  Train Loss: {loss_train / len(dataloader_train)}")
    print(f"  Train Acc: {accuracy_train / len(dataloader_train)}")

    loss_test, accuracy_test = train(model,dataloader_test,criterion,optimizer,gt_test)
    acc_test.append(accuracy_test)
    print(f"  Train Loss: {loss_test / len(dataloader_test)}")
    print(f"  Train Acc: {accuracy_test / len(dataloader_test)}")

    if epoch % 10 == 0:
        torch.save(model, os.path.join(savepath, "new_tflearn"+'_'+ str(epoch) + ".pth"))


print('xxxxxxxxxxx finished  xxxxxxxxxxx')

