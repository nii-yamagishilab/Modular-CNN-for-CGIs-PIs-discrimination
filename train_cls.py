"""
Copyright (c) 2018, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen

-----------------------------------------------------
Script for training the LDA classifier

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
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import gc
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default ='datasets/dataset_1', help='path to root dataset')
    parser.add_argument('--train_set', default ='train', help='path to train dataset')
    parser.add_argument('--val_set', default ='validation', help='path to validation dataset')
    parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--name', default ='dataset_1_output', help='name of training output')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=100, help='the height / width of the input image to network')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--resume', action='store_true', default=False, help="choose a epochs to resume from (0 to train from scratch)")
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=50)
    parser.add_argument('--step_len', type=int, default=1)

    opt = parser.parse_args()
    print(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    model_path = os.path.join(opt.outf, opt.name)

    transform_fwd = transforms.Compose([
        #transforms.Scale(opt.imageSize),
        #transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # folder dataset
    dataset_train = dset.ImageFolder(root=os.path.join(opt.dataset, opt.train_set), transform=transform_fwd)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    dataset_val = dset.ImageFolder(root=os.path.join(opt.dataset, opt.val_set), transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))


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
    # vgg_4 = VggExtractor(vgg_net, 17, 25)
    # vgg_5 = VggExtractor(vgg_net, 26, 34)

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
    if opt.resume:
        text_test = open(os.path.join(model_path, 'validation_lda.txt'), 'a')
    else:
        text_test = open(os.path.join(model_path, 'validation_lda.txt'), 'w')

    steps = int((opt.end - opt.begin)/opt.step_len)

    for i in range(steps):
        model_id = opt.begin + (i + 1)*opt.step_len

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

        # netDis = _netDis()
        # netDis.load_state_dict(torch.load('%s/dis_%d.pth' % (model_path, model_id)))

        if opt.cuda:
            netStats_1.cuda()
            netStats_2.cuda()
            netStats_3.cuda()
            # netStats_4.cuda()
            # netStats_5.cuda()
            # netDis.cuda()

        ##################################################################################

        features_lst = np.array([], dtype=np.float).reshape(0,384)
        labels_lst = np.array([], dtype=np.float)

        netStats_1.eval()
        netStats_2.eval()
        netStats_3.eval()
        # netStats_4.eval()
        # netStats_5.eval()
        # netDis.eval()


        for img_data, labels_data in dataloader_train:

            if opt.cuda:
                img_data = img_data.cuda()

            input_v = Variable(img_data, requires_grad = False)

            vgg_output = vgg_1(input_v)
            input_v = Variable(vgg_output.detach().data, requires_grad = False)
            output_1 = netStats_1(input_v).data.cpu().numpy()

            vgg_output = vgg_2(vgg_output)
            input_v = Variable(vgg_output.detach().data, requires_grad = False)
            output_2 = netStats_2(input_v).data.cpu().numpy()

            vgg_output = vgg_3(vgg_output)
            input_v = Variable(vgg_output.detach().data, requires_grad = False)
            output_3 = netStats_3(input_v).data.cpu().numpy()

            # vgg_output = vgg_4(vgg_output)
            # input_v = Variable(vgg_output.detach().data, requires_grad = False)
            # output_4 = netStats_4(input_v)

            # vgg_output = vgg_5(vgg_output)
            # input_v = Variable(vgg_output.detach().data, requires_grad = False)
            # output_5 = netStats_5(input_v)

            # output_dis = netDis(output_1, output_2, output_3)
            # output_pred = output_dis.data.cpu().numpy()

            output_t = np.concatenate((output_1, output_2, output_3), axis=1)

            features_lst = np.vstack((features_lst, output_t))
            labels_lst = np.concatenate((labels_lst, labels_data.numpy()))

        # training
        #clf = SVC()
        clf = LinearDiscriminantAnalysis()
        clf.fit(features_lst, labels_lst)


        ##################################################################################
        features_lst = np.array([], dtype=np.float).reshape(0,384)
        labels_lst = np.array([], dtype=np.float)


        for img_data, labels_data in dataloader_val:

            if opt.cuda:
                img_data = img_data.cuda()

            input_v = Variable(img_data, requires_grad = False)

            vgg_output = vgg_1(input_v)
            input_v = Variable(vgg_output.detach().data, requires_grad = False)
            output_1 = netStats_1(input_v).data.cpu().numpy()

            vgg_output = vgg_2(vgg_output)
            input_v = Variable(vgg_output.detach().data, requires_grad = False)
            output_2 = netStats_2(input_v).data.cpu().numpy()

            vgg_output = vgg_3(vgg_output)
            input_v = Variable(vgg_output.detach().data, requires_grad = False)
            output_3 = netStats_3(input_v).data.cpu().numpy()

            # vgg_output = vgg_4(vgg_output)
            # input_v = Variable(vgg_output.detach().data, requires_grad = False)
            # output_4 = netStats_4(input_v)

            # vgg_output = vgg_5(vgg_output)
            # input_v = Variable(vgg_output.detach().data, requires_grad = False)
            # output_5 = netStats_5(input_v)

            # output_dis = netDis(output_1, output_2, output_3)
            # output_pred = output_dis.data.cpu().numpy()

            output_t = np.concatenate((output_1, output_2, output_3), axis=1)

            features_lst = np.vstack((features_lst, output_t))
            labels_lst = np.concatenate((labels_lst, labels_data.numpy()))


        acc = clf.score(features_lst, labels_lst)

        abspath = os.path.abspath('%s/lda_%d.pickle' % (model_path, model_id))
        pickle.dump(clf, open(abspath, 'wb'))

        print('%d\t%.4f' % (model_id, acc))
        text_test.write('%d\t%.4f\n' % (model_id, acc))

        gc.collect()

    text_test.flush()
    text_test.close()
