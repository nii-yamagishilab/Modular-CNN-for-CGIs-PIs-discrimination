"""
Copyright (c) 2018, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen

-----------------------------------------------------
Script for evaluating the network on splicing images using the LDA classifier

"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import math
from PIL import Image
from os import listdir
from os.path import isfile, join
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default ='datasets/splicing', help='path to dataset')
    parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--name', default ='dataset_1_output', help='name of training output')
    parser.add_argument('--imageSize', type=int, default=100, help='the height / width of the input image to network')
    parser.add_argument('--stepSize', type=int, default=20, help='the step size')
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--id', type=int, default=45, help='checkpoint ID')
    parser.add_argument('--random_sample', type=int, default=0, help='number of random sample to test')
    parser.add_argument('--output', default ='splicing', help='path to store enhanced images')

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


    def extract_subimages(image, subimage_size=100, step = 20):
        subimages = []
        width = image.shape[3]
        height = image.shape[2]

        current_height = 0

        while current_height + subimage_size <= height:
            current_width = 0
            while current_width + subimage_size <= width:
                sub = image[:,:,current_height:current_height+subimage_size, current_width:current_width+subimage_size]

                subimages.append(sub)
                current_width += step
            current_height += step

        return subimages

    def calculate_probabilities(image, prob_lst, subimage_size=100, step = 20):
        width = math.floor(image.shape[3] / step * step)
        height = math.floor(image.shape[2] / step * step)
        sub_step = int(subimage_size / step)

        prob_map = np.zeros((height, width, 2), dtype=float)

        current_height = 0
        count = 0

        while current_height + subimage_size <= height:
            current_width = 0
            while current_width + subimage_size <= width:
                prob_map[current_height:current_height+subimage_size, current_width:current_width+subimage_size, 0] += prob_lst[count]
                prob_map[current_height:current_height+subimage_size, current_width:current_width+subimage_size, 1] += 1

                current_width += step
                count += 1
            current_height += step

        return prob_map[:,:,0]/prob_map[:,:,1]

    def fill_color(prob_img):
        color_img = np.zeros((prob_img.shape[0], prob_img.shape[1], 3), dtype = np.uint8)

        for i in range(prob_img.shape[0]):
            for j in range(prob_img.shape[1]):
                if prob_img[i,j] > 0.5:
                    scale = int((1.5 - prob_img[i,j])*2*255)
                    color_img[i,j] = np.array([255, scale, scale])
                else:
                    scale = int((prob_img[i,j]-0.5)*2*255)
                    color_img[i,j] = np.array([scale, 255, scale])

        return color_img

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

    abspath = os.path.abspath('%s/lda_%d.pickle' % (model_path, model_id))
    #abspath = os.path.abspath('%s/lda_%d.pickle' % (model_path, model_id))
    clf = pickle.load(open(abspath, 'rb'))

    netStats_1.eval()
    netStats_2.eval()
    netStats_3.eval()
    # netStats_4.eval()
    # netStats_5.eval()

    if opt.cuda:
        netStats_1.cuda()
        netStats_2.cuda()
        netStats_3.cuda()
        # netStats_4.cuda()
        # netStats_5.cuda()

    ##################################################################################
    #prob_lst = np.array([], dtype=np.float)
    labels_lst = np.array([], dtype=np.float)

    data_path = opt.dataset

    for f in listdir(data_path):
        if isfile(join(data_path, f)):
            if f.lower().endswith(('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'gif', 'tiff')):
                test_img = Image.open(join(data_path, f))
                test_img = transform_fwd(test_img)

                test_img.unsqueeze_(0)

                features_lst = np.array([], dtype=np.float).reshape(0,384)
                subimages = extract_subimages(test_img, opt.imageSize, opt.stepSize)
                n_sub_imgs = len(subimages)
                steps = int(math.ceil(n_sub_imgs*1.0/opt.batchSize))

                for i in range(steps):

                    img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

                    end = (i + 1)*opt.batchSize
                    if end > n_sub_imgs:
                        end = n_sub_imgs - i*opt.batchSize
                    else:
                        end = opt.batchSize

                    for j in range(end):
                        img_tmp = torch.cat((img_tmp, subimages[i*opt.batchSize + j]), dim=0)

                    if opt.cuda:
                        img_tmp = img_tmp.cuda()

                    input_v = Variable(img_tmp, requires_grad = False)

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
                    # output_4 = netStats_4(input_v).data.cpu().numpy()

                    # vgg_output = vgg_5(vgg_output)
                    # input_v = Variable(vgg_output.detach().data, requires_grad = False)
                    # output_5 = netStats_5(input_v).data.cpu().numpy()

                    output_t = np.concatenate((output_1, output_2, output_3), axis=1)
                    features_lst = np.vstack((features_lst, output_t))


                output_pred = clf.predict_proba(features_lst)

                prob_map = calculate_probabilities(test_img, output_pred[:,0], opt.imageSize, opt.stepSize)
                color_img = fill_color(prob_map)
                color_img = Image.fromarray(color_img)
                color_img.save(os.path.join(opt.output, f))
