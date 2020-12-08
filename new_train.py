import torch
import numpy as np
from tripletnet import Tripletnet
from new_model import Get_model
from torch.autograd.variable import Variable
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data
from new_dataloader import Data_loader
import os
from new_getpatches import patches
import argeparse


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

npatch_train = args.npatch_train
npatch_test = args.npatch_test
batch_size = args.batch_size
workers = args.workers

# Get image patche pairs (Anchor/Positive/Negative)
getpatches = patches(args.rgb_path_train,args.thermal_path_train,os.listdir(args.rgb_path_train),os.listdir(args.thermal_path_train),npatch_train)
matches_train,descript_train = getpatches.getpatches()
print(len(matches_train))
getpatches = patches(args.rgb_path_test,args.thermal_path_test,os.listdir(args.rgb_path_test),os.listdir(args.thermal_path_test),npatch_test)
matches_test,descript_test = getpatches.getpatches()
print(len(matches_test))

transform = transforms.Compose([transforms.ToTensor()
                                   , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

dataset_train = Data_loader(matches_train,descript_train, transform=transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train, num_workers=workers, batch_size=batch_size, shuffle=False)
print('Len dataloader train:' + str(len(dataloader_train)))

dataset_test = Data_loader(matches_test,descript_test, transform=transform)

dataloader_test = torch.utils.data.DataLoader(dataset_test, num_workers=workers, batch_size=batch_size, shuffle=False)
print('Len dataloader test:' + str(len(dataloader_test)))

# Load model
model = Get_model()
model = model.to(device)
tnet = Tripletnet(model)

# Define criterion and optimizer
criterion1 = torch.nn.MarginRankingLoss(margin=0.2)
criterion2 = torch.nn.MSELoss()
optimizer = optim.SGD(tnet.parameters(), lr=0.001, momentum=0.5)

# accuracy calculations
def accuracy(dista, distb):
    dista = dista.cpu().detach().numpy().reshape(dista.shape[0], -1)
    distb = distb.cpu().detach().numpy().reshape(dista.shape[0], -1)
    y = np.zeros(dista.shape)
    y[dista < distb] = 1
    return sum(y) / dista.shape[0], y

# Train function
def train(tnet, dataloader, criterion1,criterion2, optimizer, epoch):
    tnet.train()

    prediction_train = np.array([]).reshape(batch_size, -1)
    running_loss, correct = 0.0, 0
    count = 0
    for batch_idx, (d1, d2, d3, d4) in enumerate(dataloader):

        d1, d2, d3, d4 = Variable(d1), Variable(d2), Variable(d3), Variable(d4)
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(d1.to(device), d2.to(device), d3.to(device))

        target = torch.FloatTensor(dista.size()).fill_(-1)
        count += 1
        if device:
            target = target.to(device)
        target = Variable(target)

        loss_triplet = criterion1(dista, distb, target)
        loss_mse = criterion2(embedded_x, d4.to(device))

        # loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + loss_mse # + 0.001 * loss_embedd

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc, pred = accuracy(dista, distb)

        running_loss += loss.item()
        correct += acc.item()
        # prediction_train = np.hstack([prediction_train, pred])
        prediction_train = None

    return [running_loss, correct], prediction_train

# Eval Function
def test(tnet, dataloader, criterion1,criterion2, epoch):
    tnet.eval()

    running_loss, correct = 0.0, 0
    prediction_test = np.array([]).reshape(batch_size, -1)
    count = 0
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, data4) in enumerate(dataloader):

            data1, data2, data3, data4 = Variable(data1), Variable(data2), Variable(data3), Variable(data4)

            dista, distb, _, _, _ = tnet(data1.to(device), data2.to(device), data3.to(device))

            target = torch.FloatTensor(dista.size()).fill_(-1)
            count += 1
            if device:
                target = target.to(device)
            target = Variable(target)

            test_loss = criterion1(dista, distb, target)

            acc, pred = accuracy(dista, distb)

            running_loss += test_loss.item()
            # print(running_loss)
            correct += acc.item()
            # prediction_test = np.hstack([prediction_test, pred])

        return [running_loss, correct], None

# Save results
directory = 'new_train'
savepath = os.path.join(args.save_path, directory)

if not os.path.exists(savepath):
    os.makedirs(savepath)

# Train/Test model overtime
loss_train = []
acc_train = []
loss_test = []
acc_test = []

for epoch in range(1, 70 + 1):
    print(f"Epoch {epoch}/{70}")
    loss_acc_train, prediction_train = train(tnet, dataloader_train, criterion1, criterion2, optimizer, epoch)
    loss_train.append(loss_acc_train[0])
    print(f"  Train Loss: {loss_acc_train[0] / len(dataloader_train)}")
    acc_train.append(loss_acc_train[1])
    print(f"  Train Acc: {loss_acc_train[1] / len(dataloader_train)}")

    loss_acc_test, prediction_test = test(tnet, dataloader_test, criterion1, criterion2, epoch)
    loss_test.append(loss_acc_test[0])

    print(f"  Test Loss: {loss_acc_test[0] / len(dataloader_test)}")
    acc_test.append(loss_acc_test[1])
    print(f"  Test Acc: {loss_acc_test[1] / len(dataloader_test)}")

    if epoch % 10 == 0:
        torch.save(model, os.path.join(savepath, "new_train"+ '_' + str(epoch) + ".pth"))

# Plot the Train/Test results
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2)
ax1.plot(loss_train)
ax1.set_title("Train Loss")
# ax1.axis('off')
ax2.plot(acc_train)
ax2.set_title("Train Accuracy")
# ax2.axis('off')
ax3.plot(loss_test)
ax3.set_title("Test Loss")
# ax3.axis('off')
ax4.plot(acc_test)
ax4.set_title("Test Accuracy")

plt.show()

print('xxxxxxxxxxx finished  xxxxxxxxxxx')


