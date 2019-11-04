# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)

# Cifar-10 data
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Data
batchSize = 32
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=0)
testLoader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)

# Data classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 卷積層1，3通道輸入，6個卷積核，核大小5*5
        self.conv2 = nn.Conv2d(6, 16, 5) # 卷積層2，6輸入通道，16個卷積核，核大小5*5
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Parameters
PATH = './cifar_net.pth'
net = Net().to(device)
print(net)
criterion = nn.CrossEntropyLoss()
lr = 0.001
epochs = 50
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

dataiter = iter(testLoader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images[:10]))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))

# 5.1 randomly show 10 images and labels respectively.
def Q5_1():
    print('# 5.1 ...')
    dataiter = iter(testLoader)
    images, labels = dataiter.next()
    randIndex = random.randint(0,21)
    # print images
    imshow(torchvision.utils.make_grid(images[randIndex:randIndex + 10]))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[randIndex + j]] for j in range(10)))


# 5.2 Print out training hyperparameters (batch size, learning rate, optimizer) 
def Q5_2():
    print('# 5.2 ...\nhyperparameters:\nbatch size: {}\nlearning rate: {}\noptimizer: {}'.format(batchSize, lr, optimizer.__class__.__name__))


# 5.3 Train 1 epoch and show loss graph
def Q5_3():
    print('# 5.3 ...')
    running_loss = 0.0
    loss_history = []
    for times, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        loss_history.append(loss.item())

    print('5.3 Finished Training\nPlotting loss graph')
    plt.plot(loss_history)


def drawTrainingLoss_and_Accuracy(epoch_loss_history, epoch_accuracy_history):
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Q5.4 Training loss and Accuracy', fontsize=16)

    axs[0].plot(epoch_loss_history)
    axs[0].set_title('Loss')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')

    axs[1].plot(epoch_accuracy_history)
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('%')

    plt.show()

def Q5_4():
    print('# 5.4 ...')
    epoch_loss_history = []
    epoch_accuracy_history = []
    # Train
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        total_train = 0
        correct_train = 0
        train_accuracy = 0
        
        for times, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss
            running_loss += loss.item()
            
            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.nelement()
            correct_train += sum(predicted.eq(labels.data)).item()
            
        train_accuracy = 100 * correct_train / total_train
        epoch_loss = running_loss/len(trainLoader)
        
        epoch_accuracy_history.append(train_accuracy)
        epoch_loss_history.append(epoch_loss)
        
        print('[%d/%d] loss: %.3f accuracy:　%.3f' % (epoch+1, epochs, epoch_loss, train_accuracy))
    
    torch.save(net.state_dict(), PATH)
    print('Finished Training')
    drawTrainingLoss_and_Accuracy(epoch_loss_history, epoch_accuracy_history)
    

def visualize_model(model, iters=1, index=0):
    images_so_far = 0
    fig = plt.figure()

    dataiter = iter(testLoader)
    
    if iters > 0:
        for i in range(0, iters):
            dataiter.next()
            
    images, labels = dataiter.next()
    images, labels = images.to(device), images.to(device)

    outputs = model(images)
    results = outputs.data[index].tolist()
    minR = min(results)
    maxR = max(results)
    diff = maxR-minR
    poResults = [ (i-minR)/diff for i in results]
    sumR = sum(poResults)
    denormResults = [i/sumR for i in poResults]

    _, preds = torch.max(outputs.data, 1)
    imshow(images.cpu().data[index])
    y_pos = np.arange(10)
    plt.bar(y_pos, denormResults, align='center', alpha=0.5)
    plt.xticks(y_pos, classes)
    plt.ylabel('Prob.')
    plt.title('Estimation result')
    plt.show()
    

def Q5_5(num=0):
    print('# 5.5 ...')
    net = Net().to(device)
    net.load_state_dict(torch.load(PATH))
    iters = int(num/batchSize)
    index = num%batchSize
    if iters > 0 and index > 0:
        index = index - 1
    visualize_model(net, iters, index)


# Q5_1()
# Q5_2()
# Q5_3()
# Q5_5(3)
