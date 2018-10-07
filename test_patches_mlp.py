"""
Copyright (c) 2018, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen

-----------------------------------------------------
Script for evaluating the network on patches dataset using the MLP classifier

"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
from torch.autograd import Variable
from sklearn import metrics
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default ='datasets/dataset_1', help='path to root dataset')
    parser.add_argument('--test_set', default ='test', help='path to test dataset')
    parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--name', default ='dataset_1_output', help='name of training output')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=100, help='the height / width of the input image to network')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--id', type=int, help='checkpoint ID')

    opt = parser.parse_args()
    print(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    model_path = os.path.join(opt.outf, opt.name)
    text_test = open(os.path.join(model_path, 'test_mlp_patches_test_full.csv'), 'w')

    transform_fwd = transforms.Compose([
        #transforms.Scale(opt.imageSize),
        #transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # folder dataset

    dataset_val = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    class VggExtractor(nn.Module):
        def __init__(self, vgg, begin, end):
            super(VggExtractor, self).__init__()
            self.features = nn.Sequential(*list(vgg.features.children())[begin:(end+1)])

        def print(self):
            print(self.features)

        def forward(self, input):
            output = self.features(input)

            return output

    vgg_net = models.vgg19(pretrained=True)
    if opt.cuda:
        vgg_net = vgg_net.cuda()

    # before ReLU
    vgg_1 = VggExtractor(vgg_net, 0, 2)
    vgg_2 = VggExtractor(vgg_net, 3, 7)
    vgg_3 = VggExtractor(vgg_net, 8, 16)
    #vgg_4 = VggExtractor(vgg_net, 17, 25)
    #vgg_5 = VggExtractor(vgg_net, 26, 34)

    # after ReLU
    # vgg_1 = VggExtractor(vgg_net, 0, 4)
    # vgg_2 = VggExtractor(vgg_net, 5, 9)
    # vgg_3 = VggExtractor(vgg_net, 10, 18)
    # vgg_4 = VggExtractor(vgg_net, 19, 27)
    # vgg_5 = VggExtractor(vgg_net, 28, 36)

    class _netStats(nn.Module):
        def __init__(self, depth, n=64):
            super(_netStats, self).__init__()
            self.depth = depth
            self.n = n
            self.conv_1 = nn.Sequential(
                nn.Conv2d(self.depth, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )

        def forward(self, input):
            x = self.conv_1(input)

            y = x.view(x.data.shape[0], x.data.shape[1], x.data.shape[2]*x.data.shape[3])

            mean = torch.mean(y, 2)
            std = torch.std(y, 2)
            result = torch.cat((mean, std), 1)

            return result

    class _netDis(nn.Module):
        def __init__(self):
            super(_netDis, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(384, 512),
                nn.Dropout(p=0.33),
                nn.Linear(512, 2),
                nn.Softmax()
                )

        def forward(self, input_1, input_2, input_3):
            input = torch.cat((input_1, input_2, input_3), 1)
            return self.main(input)


    model_id = opt.id

    netStats_1 = _netStats(64)
    netStats_1.load_state_dict(torch.load('%s/stats_1_%d.pth' % (model_path, model_id)))

    netStats_2 = _netStats(128)
    netStats_2.load_state_dict(torch.load('%s/stats_2_%d.pth' % (model_path, model_id)))

    netStats_3 = _netStats(256)
    netStats_3.load_state_dict(torch.load('%s/stats_3_%d.pth' % (model_path, model_id)))

    # netStats_4 = _netStats(512)
    # netStats_4.load_state_dict(torch.load('%s/stats_4_%d.pth' % (model_path, model_id)))

    # netStats_5 = _netStats(512)
    # netStats_5.load_state_dict(torch.load('%s/stats_5_%d.pth' % (model_path, model_id)))

    netDis = _netDis()
    netDis.load_state_dict(torch.load('%s/dis_%d.pth' % (model_path, model_id)))

    netStats_1.eval()
    netStats_2.eval()
    netStats_3.eval()
    # netStats_4.eval()
    # netStats_5.eval()
    netDis.eval()

    if opt.cuda:
        netStats_1.cuda()
        netStats_2.cuda()
        netStats_3.cuda()
        # netStats_4.cuda()
        # netStats_5.cuda()
        netDis.cuda()

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_prob = np.array([], dtype=np.float)

    for img_data, labels_data in dataloader_val:

        img_label = labels_data.numpy().astype(np.float)

        if opt.cuda:
            img_data = img_data.cuda()

        input_v = Variable(img_data, requires_grad = False)

        vgg_output = vgg_1(input_v)
        input_v = Variable(vgg_output.detach().data, requires_grad = False)
        output_1 = netStats_1(input_v)

        vgg_output = vgg_2(vgg_output)
        input_v = Variable(vgg_output.detach().data, requires_grad = False)
        output_2 = netStats_2(input_v)

        vgg_output = vgg_3(vgg_output)
        input_v = Variable(vgg_output.detach().data, requires_grad = False)
        output_3 = netStats_3(input_v)

        # vgg_output = vgg_4(vgg_output)
        # input_v = Variable(vgg_output.detach().data, requires_grad = False)
        # output_4 = netStats_4(input_v)

        # vgg_output = vgg_5(vgg_output)
        # input_v = Variable(vgg_output.detach().data, requires_grad = False)
        # output_5 = netStats_5(input_v)

        output_dis = netDis(output_1, output_2, output_3).data.cpu().numpy()

        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)
        output_prob = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
            if output_dis[i,1] >= output_dis[i,0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

            output_prob[i] = output_dis[i,1]

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))
        tol_prob = np.concatenate((tol_prob, output_prob))


    acc = metrics.accuracy_score(tol_label, tol_pred)
    print('Val accuracy: %.4f' % (acc))

    for i in range(tol_prob.shape[0]):
        text_test.write('%d,%.2f\n' % (tol_label[i], tol_prob[i]))

    text_test.flush()
    text_test.close()
