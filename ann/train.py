import copy
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from cnn.dataset import DayClassifierDataset
from cnn.load_data import load_data
from cnn.network import SimpleNet

def adjust_learning_rate(epoch):
    lr = 0.001
    if epoch > 100:
        lr = lr / 1000000
    elif epoch > 80:
        lr = lr / 100000
    elif epoch > 60:
        lr = lr / 10000
    elif epoch > 50:
        lr = lr / 1000
    elif epoch > 40:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    torch.save(model.state_dict(), "cnn{}.model".format(epoch))
    print("Improved model saved")


accuracies_by_classes = []
classes_names = ['non-critical', 'critical']
confmat = confusion_matrix([], [], labels=classes_names)
best_confmat = confusion_matrix([], [], labels=classes_names)


def test():
    global confmat
    model.eval()
    test_acc = 0.0
    labels_for_matrix = []
    predictions = []
    for i, (images, labels) in enumerate(test_loader):
        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        # prediction = prediction.cpu().numpy()
        labels_for_matrix.extend(labels.data)
        predictions.extend(prediction)
        test_acc += torch.sum(torch.eq(prediction, labels.data))

    confmat = confusion_matrix(labels_for_matrix, predictions)
    accuracies_by_classes.append(confmat.diagonal() / confmat.sum(axis=1))
    # Compute the average acc and loss over all test images
    test_acc = test_acc / 546
    return test_acc


test_accuracies = []
losses = []
acc = []


def train(num_epochs):
    global best_confmat
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("training...")
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (days, labels) in enumerate(train_loader):
            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the train set
            outputs = model(days)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data.item() * days.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 50000 training images
        number = 2737
        train_acc = train_acc / number
        acc.append(train_acc)
        train_loss = train_loss / number
        losses.append(train_loss)

        # Evaluate on the test set
        test_acc = test()
        test_accuracies.append(test_acc)

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc
            best_confmat = copy.deepcopy(confmat)

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,
                                                                                        test_acc))


train_meteo = './train/meteo.xlsx'
train_med = './train/med.txt'
test_meteo = './test/meteo.xlsx'
test_med = './test/med.txt'

batch_size = 32

images, classes = load_data(train_meteo, train_med)
train_set = DayClassifierDataset(images, classes)
# Create a loader for the training set
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

images, classes = load_data(test_meteo, test_med)
test_set = DayClassifierDataset(images, classes)
# Create a loader for the test set, note that both shuffle is set to false for the test loader
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

# Create model, optimizer and loss function
model = SimpleNet(num_classes=2)

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    train(50)
    plt.figure(figsize=(12, 7))
    plt.plot(losses, color='blue')
    plt.savefig('plot1.png')
    plt.figure(figsize=(12, 7))
    plt.plot(acc, color='blue')
    plt.savefig('plot4.png')
    plt.figure(figsize=(12, 7))
    plt.plot(test_accuracies, color='blue')
    plt.savefig('plot2.png')

    plt.figure(figsize=(12, 7))
    plt.plot([acc[0] for acc in accuracies_by_classes], color='blue', label='non-critical')  # none
    plt.plot([acc[1] for acc in accuracies_by_classes], color='red', label='critical')  # cold
    plt.legend(loc="upper left")
    plt.savefig('plot3.png')

    plt.figure(figsize=(12, 7))
    sns.heatmap(best_confmat/np.sum(best_confmat), annot=True, yticklabels=classes_names,
                xticklabels=classes_names)
    plt.savefig('output.png')
