import numpy as np
import pandas as pd
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import copy
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

BATCH_SIZE = 4
MODELS = ['resnet18', 'resnet50']
LEARNING_RATE = 1e-3
EPOCHS = [10, 5]
PATH = {
    'training_process': '{}.png',
    'model_weights': '{}.pth',
    'confusion_plot': '{}.png'
}


def setup():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data = RetinopathyLoader('data', 'train', transformations=trans)
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=BATCH_SIZE, num_workers=2)
    test_data = RetinopathyLoader('data', 'test', transformations=trans)
    test_loader = DataLoader(test_data, shuffle=False,
                             batch_size=BATCH_SIZE, num_workers=2)

    return {'train': train_loader, 'test': test_loader}, device


def eval_acc(model, data, device, return_all=False):
    correct = 0
    y_hat_all, y_true = np.array([]), np.array([])
    with torch.no_grad():
        for datum in data:
            x, y = datum[0].to(device), datum[1].to(device)
            outputs = model(x)
            y_pred = torch.argmax(outputs.data, dim=-1)
            correct += (y_pred == y).float().sum()

            if return_all:
                y_hat_all = np.append(y_hat_all, y_pred.cpu().numpy())
                y_true = np.append(y_true, y.cpu().numpy())
    if return_all:
        return correct / len(data.dataset), y_hat_all, y_true
    return correct / len(data.dataset)


def train(model, criterion, optimizer,
          data_loaders,
          device, experiment_name, epochs=10):
    print(experiment_name)
    history = {
        experiment_name + '_train': [],
        experiment_name + '_test': []
    }
    train_test_acc = {}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_correct = 0
            with torch.set_grad_enabled(phase == 'train'):
                for datum in data_loaders[phase]:
                    x, y = datum[0].to(device), datum[1].to(device)
                    optimizer.zero_grad()
                    outputs = model(x)
                    y_pred = torch.argmax(outputs.data, dim=-1)
                    loss = criterion(outputs, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_correct += (y_pred == y).float().sum()
            acc = (running_correct / len(data_loaders[phase].dataset)).item()
            train_test_acc[phase] = acc
            if phase == 'test' and acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())
        history[experiment_name + '_train'].append(train_test_acc['train'])
        history[experiment_name + '_test'].append(train_test_acc['test'])
        print(f'     epoch: {epoch + 1},'
              f' train_acc: {train_test_acc["train"]:.3f}, test_acc: {train_test_acc["test"]:.3f}')
    history[experiment_name + '_train'] = np.array(history[experiment_name + '_train']).round(3)
    history[experiment_name + '_test'] = np.array(history[experiment_name + '_test']).round(3)
    torch.save(best_model_wts,
               PATH['model_weights'].format(experiment_name))
    print('\n\n')
    return history


def plot_result(history, model_name):
    ax = plt.subplot()
    for n in history:
        ax.plot(history[n])
    ax.legend(list(history.keys()))
    ax.set_title(f'Result Comparison ({model_name})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy(%)')
    plt.savefig(PATH['training_process'].format(model_name))
    plt.cla()


def train_diff_spec(*args):
    data_loaders, device = args
    histories = {}
    for model_name, epoch in zip(MODELS, EPOCHS):
        for pretrained in [True, False]:
            pre = 'pretrained' if pretrained else 'non_pretrained'
            exp_name = model_name + '_' + pre
            model = getattr(models, model_name)(pretrained)
            # if pretrained:
            #     for param in model.parameters():
            #         param.requires_grad = False
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, 5)
            model.to(device)
            history = (
                train(model, nn.CrossEntropyLoss(),
                      torch.optim.SGD(model.parameters(),
                                      lr=LEARNING_RATE,
                                      momentum=0.9,
                                      weight_decay=5e-4),
                      data_loaders,
                      device, exp_name, epochs=epoch)
            )
            histories.update(history)
        plot_result(histories, model_name)
        histories.clear()


def test_model(*args):
    data_loaders, device = args
    test_loader = data_loaders['test']
    final_test_acc_records = pd.DataFrame(index=MODELS, columns=['pretrained', 'non_pretrained'])

    best_acc, best_model = 0, None
    best_y_hat, y_true = None, None
    for model_name in MODELS:
        for pretrained in [True, False]:
            pre = 'pretrained' if pretrained else 'non_pretrained'
            exp_name = model_name + '_' + pre
            model = getattr(models, model_name)(pretrained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 5)
            model.load_state_dict(torch.load(PATH['model_weights'].format(exp_name)))
            model.to(device)
            model.eval()
            test_acc, y_hat, y_true = (
                eval_acc(model, test_loader, device, return_all=True)
            )
            if test_acc > best_acc:
                best_acc = test_acc
                best_y_hat = y_hat
            final_test_acc_records.loc[model_name, pre] = f'{test_acc.item() * 100:.2f}%'
    print(final_test_acc_records)
    disp = sns.heatmap(
        pd.DataFrame(
            confusion_matrix(y_true, best_y_hat, normalize='true'),
            index=range(5), columns=range(5)
        ).rename_axis('True label', axis='rows').rename_axis('Predicted label', axis='columns'),
        cmap=plt.get_cmap('Blues'),
        annot=True
    )
    disp.set_title('Normalized confusion matrix')
    plt.savefig(PATH['confusion_plot'].format('best_model_confusion_matrix'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    arguments = parser.parse_args()

    if arguments.test:
        test_model(*setup())
    else:
        train_diff_spec(*setup())
