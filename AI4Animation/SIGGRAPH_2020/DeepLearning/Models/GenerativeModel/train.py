'''
Author : Yiwei Zhao
Date : June 30, 2019

'''

from optparse import OptionParser
import os, sys

# Fix for cloud training system path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../'))

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

from torch.autograd import Variable


from model.network import ControlNetAutoEncoder
from model.discriminator import DiscriminatorConless
import model.data_loader as DL

from tensorboardX import SummaryWriter

import json


training_config = {
    'input': ['Input.txt', [i for i in range(0, 221)]],  # if None, then trained on all
    'output': ['Input.txt', [i for i in range(0, 221)]]
}


def load_average_from_txt(name_list, training_config):
    average_list = {}
    std_list = {}
    for name in name_list.keys():
        load = np.loadtxt(name_list[name])
        if training_config[name][1] is None:
            average_list[name] = load[0, :]
            std_list[name] = load[1, :]
        else:
            average_list[name] = load[0, training_config[name][1]]
            std_list[name] = load[1, training_config[name][1]]

        std_list[name][np.where(std_list[name] == 0)] = 1
    return average_list, std_list


def train_net(epochs=5,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              gpu=False,
              dir_checkpoint='./checkpoints/',
              dataset_dir='./dataset/',
              regularization=0.1,
              save_f=15,
              gan_weight=0.1,
              l1_weight=1.0,
              dropout=0.2,
              load=False,
              load_dis=False,
              non_linear="relu",
              args_d=None):

    average_list, std_list = load_average_from_txt(
            {'input': dataset_dir + training_config['input'][0].replace('.txt', 'Norm.txt'),
             'output': dataset_dir + training_config['output'][0].replace('.txt', 'Norm.txt')},
        training_config=training_config)

# Setup the tensorboard
    writer = SummaryWriter(dir_checkpoint+'tb/')

    input_file = os.path.join(dir_path, './dataset/' + training_config['input'][0])
    output_file = os.path.join(dir_path, './dataset/' + training_config['output'][0])
    toTensor_list = ('input', 'output')

    train_set = DL.ControlDatasetArbitraryChannels(input_file,
                                                   output_file,
                                                   training_config,
                                                   transform=transforms.Compose([
                                                       DL.sub_average(average_list, std_list),
                                                       DL.ToTensor(list=toTensor_list)
                                                   ]))

    N_train = len(train_set)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")

    train_dataloader = DataLoader(train_set, batch_size=batch_size, # collate_fn=DL.my_collate,
                                  shuffle=True, num_workers=1)

    net = ControlNetAutoEncoder(n_channels=train_set.get_channel(), time_range=1, dropout=dropout,
                                output_channel=train_set.get_output_channel(), non_linear=non_linear)
    dis_net = DiscriminatorConless(n_channels=train_set.get_output_channel(), time_range=1, dropout=dropout,
                                   non_linear=non_linear)

    if load:
        net.load_state_dict(torch.load(load))
        print('Model loaded from {}'.format(load))
        dis_net.load_state_dict(torch.load(load_dis))
        print('Discriminator Model loaded from {}'.format(load_dis))

    if gpu and torch.cuda.is_available():
        net.cuda()
        dis_net.cuda()

    optimizer_gen = optim.Adam(net.parameters(),
                           lr=lr,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=regularization)

    optimizer_dis = optim.Adam(dis_net.parameters(),
                               lr=lr,
                               betas=(0.9, 0.999),
                               eps=1e-08,
                               weight_decay=regularization)

    criterion = nn.MSELoss()
    cross_entropy_loss = nn.BCELoss()  # for GANs

    # a short cut for creating tensors later
    Tensor = torch.cuda.FloatTensor if (torch.cuda.is_available() and gpu) else torch.FloatTensor

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_set), str(save_cp), str(gpu)))

    if args_d == None:
        args_d = {}

    args_d['input_channel'] = train_set.get_channel()
    args_d['time_range'] = 1
    args_d['output_channel'] = train_set.get_output_channel()
    args_d['training_config'] = training_config

    with open(args.checkpoint + 'configure.txt', 'w') as f:
        json.dump(args_d, f, indent=2)

    net = net.to(device)
    dis_net = dis_net.to(device)

    global_step = 0

    for epoch in range(epochs):
        net.train()
        dis_net.train()

        epoch_loss_gen = 0
        epoch_loss_dis = 0
        epoch_loss_l = 0

        latents = []
        for step_inx, sample in enumerate(train_dataloader):
            # Concate whatever you think is necessary
            # i.e. You can add preprocessing and use result as a new channel of input
            # inputs = sample['input_smooth'].type('torch.FloatTensor')
            inputs = sample['input'].type('torch.FloatTensor')
            labels = sample['output'].type('torch.FloatTensor')
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.view((-1, inputs.shape[1]*inputs.shape[2]))
            labels = labels.view((-1, labels.shape[1]*labels.shape[2]))

            # Adversarial target labels
            valid = Variable(Tensor(inputs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(inputs.shape[0], 1).fill_(0.0), requires_grad=False)

            outputs, latent = net(inputs, device, latent=True)

            latents.append(latent)

            ######################
            # Generator training #
            ######################
            l_loss = l1_weight * criterion(outputs, labels)

            generator_loss_gan = gan_weight * cross_entropy_loss(dis_net(outputs, device), valid)
            g_loss = l_loss + generator_loss_gan

            epoch_loss_gen += g_loss.item()
            epoch_loss_l += l_loss.item()

            optimizer_gen.zero_grad()
            g_loss.backward()
            optimizer_gen.step()

            ##########################
            # Discriminator training #
            ##########################
            real_loss = cross_entropy_loss(dis_net(labels, device), valid)
            fake_loss = cross_entropy_loss(dis_net(outputs.detach(), device), fake)
            d_loss = 0.5 * gan_weight * (real_loss + fake_loss)

            epoch_loss_dis += d_loss.item()

            optimizer_dis.zero_grad()
            d_loss.backward()
            optimizer_dis.step()

            ############################
            # Write to the tensorboard #
            ############################
            writer.add_scalar('Generator loss', g_loss.item(), global_step)
            writer.add_scalar('L1 loss', l_loss.item(), global_step)
            writer.add_scalar('Gan loss', generator_loss_gan.item(), global_step)
            writer.add_scalar('Discriminator loss', d_loss.item(), global_step)
            writer.add_scalar('Real loss', real_loss.item(), global_step)
            writer.add_scalar('Fake loss', fake_loss.item(), global_step)


            global_step += 1
            # print('{0:.4f} --- loss: {1:.6f}'.format(step_inx * batch_size / N_train, loss.item()))

        print('Epoch finished ! Loss: {}, {}, {}'.format(epoch_loss_gen / step_inx, epoch_loss_dis / step_inx, epoch_loss_l / step_inx))


        # TODO: visualize result using Tensorboard
        if save_cp and epoch % save_f == 0:
            latents = np.concatenate(latents, axis=0)
            latent_mean = np.mean(latents, axis=0)
            latent_std = np.std(latents, axis=0)
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP_gen_{}.pth'.format(epoch + 1))
            torch.save(dis_net.state_dict(),
                       dir_checkpoint + 'CP_dis_{}.pth'.format(epoch + 1))
            np.save(dir_checkpoint + 'CP_latent_mean_{}.npy'.format(epoch + 1),
                    latent_mean)
            np.save(dir_checkpoint + 'CP_latent_std_{}.npy'.format(epoch + 1),
                    latent_std)
            print('Checkpoint {} saved !'.format(epoch + 1))

    writer.close()


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=500000, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=40,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')

    parser.add_option('--gan-weight', dest='gw', default=1.0,
                      type='float', help='gan training weight')
    parser.add_option('--l1-weight', dest='l1w', default=1.0,
                      type='float', help='gan training weight')

    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('--load_dis', dest='load_dis',
                      default=False, help='load file model')
    parser.add_option('-d', '--dir_checkpoint', dest='checkpoint', type='string',
                      default='checkpoints/', help='dir to save the checkpoint')
    parser.add_option('-i', '--dir_dataset', dest='dataset', type='string',
                      default='dataset/', help='dir of the dataset')
    parser.add_option('-r', '--regularization', dest='regularization', default=0.0001,
                      type='float', help='l2 model regularization')
    parser.add_option('-m', '--move_to_ground', dest='move_to_ground', default=None,
                      type='float', help='l2 model regularization')
    parser.add_option('--dropout', dest='dropout', default=0.3,
                      type='float', help='dropout rate')
    parser.add_option('--window', dest='window', default=5,
                      type='int', help='conv window size')
    parser.add_option('--non_linear', dest='non_linear',
                      default="relu", help='non_linear layer')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(os.path.join(dir_path, args.checkpoint)):
        os.mkdir(os.path.join(dir_path, args.checkpoint))
        os.mkdir(os.path.join(dir_path, args.checkpoint) + 'tb/')

    try:
        train_net(epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  dir_checkpoint=os.path.join(dir_path, args.checkpoint),
                  dataset_dir=os.path.join(dir_path, args.dataset),
                  regularization=args.regularization,
                  gan_weight=args.gw,
                  l1_weight=args.l1w,
                  dropout=args.dropout,
                  load=args.load,
                  load_dis=args.load_dis,
                  non_linear=args.non_linear,
                  args_d=args.__dict__,)

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
