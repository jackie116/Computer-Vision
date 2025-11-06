import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def CIFAR10_dataset_a():
    """write the code to grab a single mini-batch of 4 images from the training set, at random. 
   Return:
    1. A batch of images as a torch array with type torch.FloatTensor. 
    The first dimension of the array should be batch dimension, the second channel dimension, 
    followed by image height and image width. 
    2. Labels of the images in a torch array

    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size=4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    images, labels = next(iter(trainloader))
    
    return images, labels

class Net(nn.Module):
    # Use this function to define your network
    # Creates the network
    def __init__(self):
        super().__init__()
        # Inits the model layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Defines forward apth
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_classifier():
    # Creates dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size=4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Creates Network 
    net = Net()

    # Defines loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset for 2 iteration
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
    PATH = './cifar_net_2epoch.pth'
    torch.save(net.state_dict(), PATH)

def evalNetwork():
    # Initialized the network and load from the saved weights
    PATH = './cifar_net_2epoch.pth'
    net = Net()
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

def get_first_layer_weights():
    net = Net()
    # TODO: load the trained weights
    PATH = './cifar_net_2epoch.pth'
    net.load_state_dict(torch.load(PATH))
    first_weight = net.conv1.weight  # TODO: get conv1 weights (exclude bias)
    return first_weight

def get_second_layer_weights():
    net = Net()
    # TODO: load the trained weights
    PATH = './cifar_net_2epoch.pth'
    net.load_state_dict(torch.load(PATH))
    second_weight = net.conv2.weight # TODO: get conv2 weights (exclude bias)
    return second_weight

def hyperparameter_sweep():
    '''
    Reuse the CNN and training code from Question 2
    Train the network three times using different learning rates: 0.01, 0.001, and 0.0001
    During training, record the training loss every 2000 iterations
    compute and record the training and test errors every 2000 iterations by randomly sampling 1000 images from each dataset
    After training, plot three curves
    '''
    learning_rates = [0.01, 0.001, 0.0001]
    logs = {
        'index':    {0.01: [], 0.001: [], 0.0001: []},
        'loss':     {0.01: [], 0.001: [], 0.0001: []},
        'train_error':{0.01: [], 0.001: [], 0.0001: []},
        'test_error': {0.01: [], 0.001: [], 0.0001: []},
    }

    # Creates dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size=4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    for lr in learning_rates:
        # Creates Network 
        net = Net()

        # Defines loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        
        for epoch in range(2):  # loop over the dataset for 2 iteration
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
                    logs['index'][lr].append(epoch * len(trainloader) + (i + 1))
                    logs['loss'][lr].append(running_loss / 2000)
                    running_loss = 0.0

                    # Computes training error
                    train_correct = 0
                    train_total = 0
                    # since we're not training, we don't need to calculate the gradients for our outputs
                    train_index = torch.randperm(len(trainset))[:1000]
                    train_subset = torch.utils.data.Subset(trainset, train_index.tolist())
                    train_subset_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=2)

                    with torch.no_grad():
                        for data in train_subset_loader:
                            # Evaluates samples
                            images, labels = data
                            # calculate outputs by running images through the network
                            outputs = net(images)
                            # the class with the highest energy is what we choose as prediction
                            _, predicted = torch.max(outputs, 1)
                            train_total += labels.size(0)
                            train_correct += (predicted == labels).sum().item()

                    logs['train_error'][lr].append(100 * (1 - train_correct / train_total))

                    # Computes test error
                    test_correct = 0
                    test_total = 0
                    # since we're not training, we don't need to calculate the gradients for our outputs
                    test_index = torch.randperm(len(testset))[:1000]
                    test_subset = torch.utils.data.Subset(testset, test_index.tolist())
                    test_subset_loader = torch.utils.data.DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=2)
                    with torch.no_grad():
                        for data in test_subset_loader:
                            # Evaluates samples
                            images, labels = data
                            # calculate outputs by running images through the network
                            outputs = net(images)
                            # the class with the highest energy is what we choose as prediction
                            _, predicted = torch.max(outputs, 1)
                            test_total += labels.size(0)
                            test_correct += (predicted == labels).sum().item()

                    logs['test_error'][lr].append(100 * (1 - test_correct / test_total))

    return logs

def imshow(img, labels, pad=2):
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    for i, label in enumerate(labels):
        cx = pad + i * (32 + pad) + 32 / 2
        plt.text(cx, -2, classes[label.item()],
                 color='red', fontsize=14, weight='bold',
                 ha='center', va='bottom')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_hyperparameter_sweep(logs):
    learning_rates = [0.01, 0.001, 0.0001]
    # Plot Loss
    plt.figure(figsize=(7, 5))
    for lr in learning_rates:
        plt.plot(logs['index'][lr], logs['loss'][lr], label=f'lr={lr}')
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot Training Error
    plt.figure(figsize=(7, 5))
    for lr in learning_rates:
        plt.plot(logs['index'][lr], logs['train_error'][lr], label=f'lr={lr}')
    plt.xlabel('Iteration')
    plt.ylabel('Training Error (%)')
    plt.title('Training Error')
    plt.legend()

    # Plot Test Error
    plt.figure(figsize=(7, 5))
    for lr in learning_rates:
        plt.plot(logs['index'][lr], logs['test_error'][lr], label=f'lr={lr}')
    plt.xlabel('Iteration')
    plt.ylabel('Test Error (%)')
    plt.title('Test Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # your text code here
    # images, labels = CIFAR10_dataset_a()
    # imshow(torchvision.utils.make_grid(images), labels)
    # train_classifier()
    evalNetwork()
    first_weight = get_first_layer_weights()
    second_weight = get_second_layer_weights()
    # logs = hyperparameter_sweep()
    # plot_hyperparameter_sweep(logs)