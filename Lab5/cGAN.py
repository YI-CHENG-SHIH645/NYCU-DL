import json
import argparse
from os import path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from multiprocessing import cpu_count
from PIL import Image


DATA_DIR = './lab5_gans'
BATCH_SIZE = 32
IMG_SIZE = 64
INIT_SIZE = IMG_SIZE // 4
LATENT_SIZE = 32
EPOCH = 100
LR = 0.01
EXPERIMENT_NAME = 'GAN'
PATH = {
    'd_model_weights': f'{EXPERIMENT_NAME}_g.pth',
    'g_model_weights': f'{EXPERIMENT_NAME}_d.pth'
}


def get_iCLEVR_data(data_dir, mode):
    if mode == 'train':
        data = json.load(open(path.join(data_dir, 'train.json')))
        obj = json.load(open(path.join(data_dir, 'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        # obj 換 int
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            # 把 int 轉 24 種 obj 的 n hot vector
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return img, label
    else:
        data = json.load(open(path.join(data_dir, 'test.json')))
        obj = json.load(open(path.join(data_dir, 'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class ICLEVRData(Dataset):
    def __init__(self, data_dir, mode, trans=None):
        super(ICLEVRData, self).__init__()
        self.data_dir = data_dir
        assert mode in ['train', 'test']
        self.mode = mode
        self.trans = trans
        self.img_list, self.label_list = get_iCLEVR_data(data_dir, mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        label = self.label_list[index]
        if self.mode == 'train':
            img = Image.open(path.join(DATA_DIR, 'images', self.img_list[index])).convert('RGB')
            if self.trans is not None:
                img = self.trans(img)
            return img, label
        return label


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(LATENT_SIZE+24, 128*INIT_SIZE**2)
        # shape: (batch, 3, 64, 64)
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 3, (3, 3), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, z, c):
        # c shape: (batch, 24)
        # z shape: (batch, 32)
        batch_size = c.size(0)
        gen_input = torch.cat([c, z], dim=-1)
        x = self.l1(gen_input).view(batch_size, 128, INIT_SIZE, INIT_SIZE)
        img = self.model(x)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # shape: (batch, 3, 64, 64)
        self.conv_img = nn.Conv2d(3, 32, (3, 3), (2, 2), (1, 1))
        self.conv_c = nn.Conv2d(24, 32, (3, 3), (2, 2), (1, 1))

        def conv_block(in_filters, out_filters):
            return [
                nn.Conv2d(in_filters, out_filters, (3, 3), (2, 2), (1, 1)),
                nn.BatchNorm2d(out_filters, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]

        self.model = nn.Sequential(
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512),
            nn.Flatten(),
            nn.Linear(512 * IMG_SIZE**2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, c):
        # img shape: (batch, 3, IMG_SIZE, IMG_SIZE)
        img = self.conv_img(img)
        # c shape: (batch, 24, IMG_SIZE, IMG_SIZE)
        c = self.conv_c(c)
        x = torch.cat([img, c], dim=1)
        x = self.model(x)
        return x


class EvaluationModel:
    def __init__(self):
        checkpoint = torch.load(path.join(DATA_DIR, 'classifier_weight.pth'))
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24

    @staticmethod
    def compute_acc(out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
        return acc / total

    def eval(self, images, labels):
        with torch.no_grad():
            # your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc


def train(g: Generator, d: Discriminator, loader):
    g_opt = torch.optim.Adam(g.parameters(), lr=LR)
    d_opt = torch.optim.Adam(d.parameters(), lr=LR)
    adversarial_loss = nn.BCELoss().cuda()
    valid = torch.ones(BATCH_SIZE, 1, requires_grad=False)
    fake = torch.zeros(BATCH_SIZE, 1, requires_grad=False)
    best_loss = np.inf
    for epoch in range(EPOCH):
        epoch_g_loss = 0.0
        for img, label in loader:
            # label shape: (batch, 24)
            img = img.cuda()
            label = label.cuda()
            g_opt.zero_grad()
            z = torch.randn(BATCH_SIZE, LATENT_SIZE, dtype=torch.float32).cuda()
            gen_img = g(z, label)

            # d need label shape: (batch, 24, 64, 64)
            label_2d = torch.zeros(label.size(0), 24, IMG_SIZE, IMG_SIZE,
                                   dtype=torch.float32)
            for i in range(label.size(0)):
                label_2d[i, label[i], :, :] = 1
            g_loss = adversarial_loss(d(gen_img, label_2d), valid)
            g_loss.backward()
            g_opt.step()
            epoch_g_loss += g_loss.item()

            d_opt.zero_grad()
            valid_loss = adversarial_loss(d(img, label_2d), valid)
            fake_loss = adversarial_loss(d(gen_img, label_2d), fake)
            d_loss = (valid_loss + fake_loss)/2

            d_loss.backward()
            d_opt.step()
        epoch_g_loss /= len(loader)
        if epoch_g_loss < best_loss:
            best_loss = epoch_g_loss
            torch.save(g.state_dict(), PATH['g_model_weights'].format())


def evaluate(g, d, loader):
    g.eval()
    eval_m = EvaluationModel()
    acc_total = 0.0
    for label in loader:
        z = torch.randn(BATCH_SIZE, LATENT_SIZE).cuda()
        gen_img = g(z, label)
        acc_total += eval_m.eval(gen_img, label)
    return acc_total / len(loader)


def setup(mode: str):
    trans = transforms.Compose([
        transforms.Resize([IMG_SIZE, IMG_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    data = ICLEVRData(DATA_DIR, mode=mode, trans=trans)
    dl = DataLoader(data,
                    batch_size=BATCH_SIZE,
                    shuffle=mode == 'train',
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available())
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    if mode == 'test':
        generator.load_state_dict(torch.load(PATH['g_model_weights']))
    return generator, discriminator, dl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False)
    arguments = parser.parse_args()

    if arguments.test:
        accuracy = evaluate(*setup(mode='test'))
        print(f'acc: {accuracy:.2f}')
    else:
        train(*setup(mode='train'))
