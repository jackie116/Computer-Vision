import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


def compute_num_parameters(net:nn.Module):
    """compute the number of trainable parameters in *net* e.g., ResNet-34.  
    Return the estimated number of parameters Q1. 
    """
    num_para = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return num_para


def CIFAR10_dataset_a():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    return images, labels


class GAPNet(nn.Module):
    """
    Insert your code here
    """
    def __init__(self):
        super().__init__()
        # Inits the model layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(6, 10, kernel_size=(5, 5), stride=(1, 1))
        self.gap = nn.AvgPool2d(kernel_size=10, stride=10, padding=0)
        self.fc = nn.Linear(in_features=10, out_features=10, bias=True)

    def forward(self, x):
        # Defines the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.gap(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x
    
def train_GAPNet():
    """
    Insert your code here
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size=4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Creates Network 
    net = GAPNet()

    # Defines loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset for 10 iterations
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    print('Finished Training')

    # Saves the model weights after training
    PATH = './Gap_net_10epoch.pth'
    torch.save(net.state_dict(), PATH)


def eval_GAPNet():
    """
    Insert your code here
    """
    # Initialized the network and load from the saved weights
    PATH = './Gap_net_10epoch.pth'
    net = GAPNet()
    net.load_state_dict(torch.load(PATH))
    # Loads dataset
    batch_size =4
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # Evaluates samples
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def backbone():
    """
    Insert your code here, Q3
    """
    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # Remove the final fully connected layer
    backbone = nn.Sequential(*list(resnet18.children())[:-1])

    backbone.eval()  # Set to evaluation mode

    img = torchvision.io.read_image('cat_eye.jpg')  # (C, H, W), uint8
    img = transforms.ToPILImage()(img)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = transform(img)  # (C, H, W)
    img = img.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)

    with torch.no_grad():
        features = backbone(img)  # (1, 512, 1, 1)
        # features = torch.flatten(features, 1)  # (1, 512)
        # features = features.squeeze(0)  # (512,)

    return features

def transfer_learning():
    """
    Insert your code here, Q4
    """
    # device selection: prefer MPS (Apple GPU) when available, otherwise CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)  # optional debug print

    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # Freeze all layers
    for param in resnet18.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)  # Assuming 10 classes for new task

    # Now only the parameters of the final layer will be updated
    for param in resnet18.fc.parameters():
        param.requires_grad = True
    
    # Training
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size=32

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Creates Network 
    net = resnet18

    net = net.to(device)

    # Defines loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)

    net.train()

    for epoch in range(10):  # loop over the dataset for 10 iterations
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    
    print('Finished Training')

    # Saves the model weights after training
    PATH = './Res_net_10epoch.pth'
    torch.save(net.state_dict(), PATH)

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # eval
    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    net.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # Evaluates samples
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the 10000 test images: {100.0 * correct // total} %')

def deepwise_separable_conv(ch_in, ch_out, stride):
    layer = nn.Sequential(
        nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=stride, padding=1, groups=ch_in, bias=False),
        nn.BatchNorm2d(ch_in),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True)
    )
    return layer

class MobileNetV1(nn.Module):
    """Define MobileNetV1 please keep the strucutre of the class Q5"""
    def __init__(self, ch_in, n_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.deepwise_separable_layers = nn.Sequential(
            deepwise_separable_conv(32, 64, stride=1),
            deepwise_separable_conv(64, 128, stride=2),
            deepwise_separable_conv(128, 128, stride=1),
            deepwise_separable_conv(128, 256, stride=2),
            deepwise_separable_conv(256, 256, stride=1),
            deepwise_separable_conv(256, 512, stride=2),
            # 5 times
            deepwise_separable_conv(512, 512, stride=1),
            deepwise_separable_conv(512, 512, stride=1),
            deepwise_separable_conv(512, 512, stride=1),
            deepwise_separable_conv(512, 512, stride=1),
            deepwise_separable_conv(512, 512, stride=1),
            # continue
            deepwise_separable_conv(512, 1024, stride=2),
            deepwise_separable_conv(1024, 1024, stride=1)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deepwise_separable_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    #Q1
    # from torchvision import models
    # resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    # num_para = compute_num_parameters(resnet34)
    # print(f"Number of trainable parameters in ResNet-34: {num_para}")
    #Q2
    # train_GAPNet()
    # eval_GAPNet()
    #Q3
    # features = backbone()
    # print(features.shape)  # Should print torch.Size([512])
    #Q4
    # transfer_learning()
    # Q5
    ch_in=3
    n_classes=1000
    model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)

    # device selection: prefer MPS (Apple GPU) when available, otherwise CPU
    # if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    # print("Using device:", device)  # optional debug print

    # img = torchvision.io.read_image('cat_eye.jpg')  # (C, H, W), uint8
    # img = transforms.ToPILImage()(img)

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # img = transform(img)  # (C, H, W)
    # img = img.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)

    # model = model.to(device)
    # model.eval()

    # with torch.no_grad():
    #     img = img.to(device)
    #     outputs = model(img)
    #     print(outputs.shape)  # Should print torch.Size([1, 1000])
