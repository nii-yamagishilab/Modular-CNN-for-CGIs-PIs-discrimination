"""
Copyright (c) 2018, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen

-----------------------------------------------------
Script for evaluating the network on full-size dataset using the MLP classifier

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
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default ='datasets/dataset_1', help='path to root dataset')
    parser.add_argument('--test_set', default ='test', help='path to test dataset')
    parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--name', default ='dataset_1_output', help='name of training output')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--imageSize', type=int, default=100, help='the height / width of the input image to network')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--id', type=int, help='checkpoint ID')
    parser.add_argument('--random_sample', type=int, default=0, help='number of random sample to test')

    opt = parser.parse_args()
    print(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    model_path = os.path.join(opt.outf, opt.name)
    text_test = open(os.path.join(model_path, 'test_mlp_dndt.csv'), 'w')

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
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=int(opt.workers))


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

    del vgg_net

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


    def extract_subimages(image, subimage_size=256):
        subimages = []
        width = image.shape[3]
        height = image.shape[2]

        current_height = 0

        while current_height + subimage_size <= height:
            current_width = 0
            while current_width + subimage_size <= width:
                sub = image[:,:,current_height:current_height+subimage_size, current_width:current_width+subimage_size]

                subimages.append(sub)
                current_width += subimage_size
            current_height += subimage_size

        return subimages

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

    ##################################################################################
    predict_lst = np.array([], dtype=np.float)
    #prob_lst = np.array([], dtype=np.float)
    labels_lst = np.array([], dtype=np.float)

    for img_data, labels_data in dataloader_val:

        img_label = labels_data.numpy().astype(np.float)

        subimages = extract_subimages(img_data, opt.imageSize)
        prob = np.array([[0.0, 0.0]])
        n_sub_imgs = len(subimages)

        if (opt.random_sample > 0):
            if n_sub_imgs > opt.random_sample:
                np.random.shuffle(subimages)
                n_sub_imgs = opt.random_sample

            img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

            for i in range(n_sub_imgs):
                img_tmp = torch.cat((img_tmp, subimages[i]), dim=0)

            if opt.cuda:
                img_tmp = img_tmp.cuda()

            input_v = Variable(img_tmp, requires_grad = False)

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

            output_dis = netDis(output_1, output_2, output_3)
            output_pred = output_dis.data.cpu().numpy()

        else:
            batchSize = 10
            steps = int(math.ceil(n_sub_imgs*1.0/batchSize))

            output_pred = np.array([], dtype=np.float).reshape(0,2)

            for i in range(steps):

                img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

                end = (i + 1)*batchSize
                if end > n_sub_imgs:
                    end = n_sub_imgs - i*batchSize
                else:
                    end = batchSize

                for j in range(end):
                    img_tmp = torch.cat((img_tmp, subimages[i*batchSize + j]), dim=0)

                if opt.cuda:
                    img_tmp = img_tmp.cuda()

                input_v = Variable(img_tmp, requires_grad = False)

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

                output_dis = netDis(output_1, output_2, output_3)
                output_p = output_dis.data.cpu().numpy()

                output_pred = np.concatenate((output_pred, output_p), axis=0)

        output_pred = output_pred.mean(0)

        if output_pred[1] >= output_pred[0]:
            pred = 1.0
        else:
            pred = 0.0

        print('%d - %d' %(pred, img_label))
        text_test.write('%d,%.2f\n' % (img_label, output_pred[1]))

        predict_lst = np.concatenate((predict_lst, np.array([pred])), axis=0)
        #prob_lst = np.concatenate((prob_lst, output_pred[1]), axis=0)
        labels_lst = np.concatenate((labels_lst, img_label), axis=0)


    acc = metrics.accuracy_score(labels_lst, predict_lst)

    print(len(predict_lst))
    print('%d\t%.4f' % (model_id, acc))


    text_test.flush()
    text_test.close()
