from dataloader import read_bci_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.utils.data as torch_data
import argparse

BATCH_SIZE = 64
LEARNING_RATE = 1e-2
EPOCHS = 300
MODELS_TO_TRY = ['DeepConvNet', 'EEGNet']
ACTIVATIONS_TO_TRY = ['ReLU', 'LeakyReLU', 'ELU']
PATH = {
    'training_process': './pics/{}.png',
    'model_weights': './weights/{}.pth'
}


class EEGNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )

        self.depthWiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            getattr(nn, activation)(),
            nn.AvgPool2d((1, 4), (1, 4)),
            nn.Dropout(0.4)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            getattr(nn, activation)(),
            nn.AvgPool2d((1, 8), (1, 8)),
            nn.Dropout(0.4)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.depthWiseConv(x)
        x = self.separableConv(x)
        return self.classify(x)


class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (2, 1)),
            nn.BatchNorm2d(25),
            getattr(nn, activation)(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50),
            getattr(nn, activation)(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100),
            getattr(nn, activation)(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 5)),
            nn.BatchNorm2d(200),
            getattr(nn, activation)(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8600, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return self.classify(x)


def eval_acc(net, data, device):
    correct = 0
    with torch.no_grad():
        for datum in data:
            x, y = datum[0].to(device), datum[1].to(device)
            outputs = net(x)
            y_pred = torch.argmax(outputs.data, dim=-1)
            correct += (y_pred == y).float().sum()

    return correct / len(data.dataset)


def setup():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_test, y_test = read_bci_data()
    tensor_data = torch_data.TensorDataset(torch.from_numpy(x_train).float(),
                                           torch.from_numpy(y_train).long())
    train_loader = torch_data.DataLoader(tensor_data,
                                         batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=2)
    tensor_data = torch_data.TensorDataset(torch.from_numpy(x_test).float(),
                                           torch.from_numpy(y_test).long())
    test_loader = torch_data.DataLoader(tensor_data,
                                        batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=2)

    return device, train_loader, test_loader


def train(net, criterion, optimizer,
          train_loader, test_loader,
          device, experiment_name, epochs=10):
    history = {
        experiment_name + '_train': [],
        experiment_name + '_test': []
    }
    for epoch in range(epochs):
        net.train()
        for i, datum in enumerate(train_loader):
            x, y = datum[0].to(device), datum[1].to(device)
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        net.eval()
        train_acc = eval_acc(net, train_loader, device)
        history[experiment_name + '_train'].append(train_acc.item())
        test_acc = eval_acc(net, test_loader, device)
        history[experiment_name + '_test'].append(test_acc.item())
        print(f'{net.__class__.__name__}({experiment_name}),'
              f' epoch: {epoch + 1},'
              f' train_acc: {train_acc:.3f}, test_acc: {test_acc:.3f}')
    history[experiment_name + '_train'] = np.array(history[experiment_name + '_train']).round(3)
    history[experiment_name + '_test'] = np.array(history[experiment_name + '_test']).round(3)

    return history


def plot_result(history, model_name):
    ax = plt.subplot()
    for n in history:
        ax.plot(history[n])
    ax.legend(list(history.keys()))
    ax.set_title(f'Activation function comparison ({model_name})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy(%)')
    plt.savefig(PATH['training_process'].format(model_name))
    plt.cla()


def try_6_models(*args):
    device, train_loader, test_loader = args
    histories = {}

    for model_name in MODELS_TO_TRY:
        for act in ACTIVATIONS_TO_TRY:
            model = globals()[model_name](act)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            history = (
                train(model, nn.CrossEntropyLoss(),
                      optimizer,
                      train_loader, test_loader, device=device,
                      epochs=EPOCHS, experiment_name=act)
            )
            histories.update(history)
            torch.save(model.state_dict(),
                       PATH['model_weights'].format(model_name + '_' + act))
        plot_result(histories, model_name)


def test_model(*args):
    device, _, test_loader = args
    final_test_acc_records = pd.DataFrame(index=MODELS_TO_TRY, columns=ACTIVATIONS_TO_TRY)

    for model_name in MODELS_TO_TRY:
        for act in ACTIVATIONS_TO_TRY:
            model = globals()[model_name](act)
            model.load_state_dict(torch.load(PATH['model_weights'].format(model_name + '_' + act)))
            model.to(device)
            model.eval()
            test_acc = eval_acc(model, test_loader, device)
            final_test_acc_records.loc[model_name, act] = f'{test_acc.item()*100:.2f}%'
    print(final_test_acc_records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    arguments = parser.parse_args()

    if arguments.test:
        test_model(*setup())
    else:
        try_6_models(*setup())
