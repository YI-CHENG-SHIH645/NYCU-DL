import argparse
import os
from os import path
import json
import numpy as np
from multiprocessing import cpu_count
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision import models

import torch.nn as nn
import torch
import torch.autograd as autograd

DATA_DIR = './lab5_gans'
os.makedirs("images", exist_ok=True)
EXPERIMENT_NAME = 'GAN'
PATH = {
    'g_model_weights': 'WGAN_GP_generator_{}.pth'
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


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def create_condition_channel(z, c):
    # z shape: (batch_size, 100/3, 4/64, 4/64)
    # c shape: (batch_size, 24)
    batch_size = c.size(0)
    channel_c = torch.zeros(batch_size, opt.img_size, opt.img_size).cuda()
    detach_c = c.clone().detach().unsqueeze(1)
    for i in range(batch_size):
        channel_c[i].scatter_(0, detach_c[i], 1)
    channel_c = channel_c.unsqueeze(1)
    return torch.cat([z, channel_c], 1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def deconv_block(in_filters, out_filters, *deconv_args, up_sampling=True):
            if not deconv_args:
                deconv_args = (4, 2, 1) if up_sampling else (4, 1, 0)
            deconv_args = (in_filters, out_filters, *deconv_args)
            block = [
                nn.ConvTranspose2d(*deconv_args),
            ]
            if up_sampling:
                block.extend([
                    nn.BatchNorm2d(out_filters),
                    nn.LeakyReLU(0.2, inplace=True)
                ])

            return block

        d = 128
        self.d = d
        self.l1 = nn.Linear(opt.latent_dim+24, d * 4**2)
        # self.deconv_z = deconv_block(d, d, up_sampling=False)[0]
        self.deconv_blocks = nn.Sequential(
            *deconv_block(d, d//2, up_sampling=True),
            *deconv_block(d//2, d//2, up_sampling=True),
            *deconv_block(d//2, d//4, up_sampling=True),
            # *deconv_block(d//4, d//4, up_sampling=True),  # for 2x2
            *deconv_block(d//4, d//8, up_sampling=True),
            deconv_block(d//8, 3, 3, 1, 1, up_sampling=False)[0],
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self._modules:
            normal_init(self._modules[m], 0.0, 0.02)

    def forward(self, z, c):
        x = self.l1(torch.cat([z, c], 1)).view(-1, self.d, 4, 4)
        return self.deconv_blocks(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def conv_block(in_filters, out_filters, *conv_args, down_sampling=True):
            if not conv_args:
                conv_args = (4, 2, 1) if down_sampling else (4, 1, 0)
            conv_args = (in_filters, out_filters, *conv_args)
            block = [
                nn.Conv2d(*conv_args),
            ]
            if down_sampling:
                block.extend([
                    nn.BatchNorm2d(out_filters),
                    nn.LeakyReLU(0.2, inplace=True)
                ])

            return block

        d = 32
        self.conv_img = conv_block(3+1, d, down_sampling=True)[0]
        self.conv_blocks = nn.Sequential(
            *conv_block(d, d*2, down_sampling=True),
            *conv_block(d*2, d*2, down_sampling=True),
            *conv_block(d*2, d*4, down_sampling=True),
            conv_block(d*4, 1, down_sampling=False)[0],
        )
        self._init_weights()

    def _init_weights(self):
        for m in self._modules:
            normal_init(self._modules[m], 0.0, 0.02)

    def forward(self, img, c):
        x = create_condition_channel(img, c)
        x = self.conv_img(x)
        return self.conv_blocks(x).view(-1, 1)


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


def evaluate(g, loader):
    g.eval()
    eval_m = EvaluationModel()
    acc_total = 0.0
    # while acc_total/len(loader) < 0.5:
    # acc_total = 0.0
    gen_img = None
    for label in loader:
        batch_size = label.size(0)
        label = label.long().cuda()
        z = torch.randn(batch_size, opt.latent_dim).cuda()
        gen_img = g(z, label)
        acc_total += eval_m.eval(gen_img, label)
    save_image(gen_img.data, "test_images.png", nrow=8, normalize=True)
    return acc_total / len(loader)


def setup(mode: str):
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    trs = [
        transforms.Resize([opt.img_size, opt.img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    trans = transforms.Compose(trs)

    data = ICLEVRData(DATA_DIR, mode=mode, trans=trans)
    dl = DataLoader(data,
                    batch_size=opt.batch_size,
                    shuffle=mode == 'train',
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available())
    if mode == 'test':
        generator.load_state_dict(torch.load('WGAN_GP3_generator_360.pth'))
        generator.eval()
    return generator, discriminator, dl


def compute_gradient_penalty(D, real_samples, fake_samples, c):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    batch_size = real_samples.size(0)
    alpha = Tensor(np.random.random((batch_size, 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, c)
    fake = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size()[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def main():
    generator, discriminator, dataloader = setup(mode='test' if opt.test else 'train')
    if opt.test:
        acc = evaluate(generator, dataloader)
        print(f'acc: {acc:.2f}')
        return

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------
    for epoch in range(1, opt.n_epochs+1):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = labels.size(0)
            real_imgs = Variable(imgs.type(Tensor))
            labels = Variable(labels.type(LongTensor))

            # Sample noise as generator input
            np_z = np.random.normal(0, 1, (batch_size, opt.latent_dim))
            z = Variable(Tensor(np_z))

            # Generate a batch of images

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            gen_imgs = generator(z, labels)
            real_validity = discriminator(real_imgs, labels)
            fake_validity = discriminator(gen_imgs.detach(), labels)
            gp = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, labels)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10. * gp
            d_loss.backward()
            optimizer_D.step()

            # for p in discriminator.parameters():
            #     p.data.clamp_(-100., 100.)

            # -----------------
            #  Train Generator
            # -----------------
            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()
                gen_imgs = generator(z, labels)

                g_loss = -torch.mean(discriminator(gen_imgs, labels))
                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=8, normalize=True)
        if epoch % 30 == 0:
            torch.save(generator.state_dict(), PATH['g_model_weights'].format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    parser.add_argument('--test', action='store_true', default=False)
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    main()
