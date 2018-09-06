
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
import gc
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default ='datasets', help='path to root dataset')
    parser.add_argument('--train_set', default ='train', help='path to train dataset')
    parser.add_argument('--val_set', default ='validation', help='path to validation dataset')
    parser.add_argument('--val', action='store_true', default=True, help='enables validation')
    parser.add_argument('--name', default ='output', help='name of training output')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=100, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.01')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
    parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--checkpoint', type=int, default=1, help='number of epochs for checkpointing')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--debug', action='store_true', help='enables printing detail loss information')

    opt = parser.parse_args()
    print(opt)

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    model_path = os.path.join(opt.outf, opt.name)

    if opt.resume > 0:
        text_train = open(os.path.join(model_path, 'train.log'), 'a')
        text_train_detail = open(os.path.join(model_path, 'train_detail.log'), 'a')
    else:
        text_train = open(os.path.join(model_path, 'train.log'), 'w')
        text_train_detail = open(os.path.join(model_path, 'train_detail.log'), 'w')


    transform_fwd = transforms.Compose([
        #transforms.Scale(opt.imageSize),
        #transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # folder dataset
    dataset_train = dset.ImageFolder(root=os.path.join(opt.dataset, opt.train_set), transform=transform_fwd)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    if opt.val:
        dataset_val = dset.ImageFolder(root=os.path.join(opt.dataset, opt.val_set), transform=transform_fwd)
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


    netStats_1 = _netStats(64)
    netStats_1.apply(weights_init)
    if opt.resume > 0:
        netStats_1.load_state_dict(torch.load('%s/stats_1_%d.pth' % (model_path, opt.resume)))

    netStats_2 = _netStats(128)
    netStats_2.apply(weights_init)
    if opt.resume > 0:
        netStats_2.load_state_dict(torch.load('%s/stats_2_%d.pth' % (model_path, opt.resume)))

    netStats_3 = _netStats(256)
    netStats_3.apply(weights_init)
    if opt.resume > 0:
        netStats_3.load_state_dict(torch.load('%s/stats_3_%d.pth' % (model_path, opt.resume)))

    # netStats_4 = _netStats(512)
    # netStats_4.apply(weights_init)
    # if opt.resume > 0:
    #     netStats_4.load_state_dict(torch.load('%s/stats_4_%d.pth' % (model_path, opt.resume)))

    # netStats_5 = _netStats(512)
    # netStats_5.apply(weights_init)
    # if opt.resume > 0:
    #     netStats_5.load_state_dict(torch.load('%s/stats_5_%d.pth' % (model_path, opt.resume)))

    netDis = _netDis()
    if opt.resume > 0:
        netDis.load_state_dict(torch.load('%s/dis_%d.pth' % (model_path, opt.resume)))
    else:
        netDis.apply(weights_init)

    cel_criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        netStats_1.cuda()
        netStats_2.cuda()
        netStats_3.cuda()
        # netStats_4.cuda()
        # netStats_5.cuda()
        netDis.cuda()

        cel_criterion.cuda()


    # setup optimizer
    optimizer_stats_1 = optim.Adam(netStats_1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_stats_2 = optim.Adam(netStats_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_stats_3 = optim.Adam(netStats_3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizer_stats_4 = optim.Adam(netStats_4.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    # optimizer_stats_5 = optim.Adam(netStats_5.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerDis = optim.Adam(netDis.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_cum = 0

        for img_data, labels_data in dataloader_train:

            optimizer_stats_1.zero_grad()
            optimizer_stats_2.zero_grad()
            optimizer_stats_3.zero_grad()
            # optimizer_stats_4.zero_grad()
            # optimizer_stats_5.zero_grad()
            optimizerDis.zero_grad()

            if opt.cuda:
                img_data = img_data.cuda()
                labels_data = labels_data.cuda()

            input_v = Variable(img_data, requires_grad = False)

            vgg_output = vgg_1(input_v)
            input_v = Variable(vgg_output.detach().data, requires_grad = True)
            output_1 = netStats_1(input_v)

            vgg_output = vgg_2(vgg_output)
            input_v = Variable(vgg_output.detach().data, requires_grad = True)
            output_2 = netStats_2(input_v)

            vgg_output = vgg_3(vgg_output)
            input_v = Variable(vgg_output.detach().data, requires_grad = True)
            output_3 = netStats_3(input_v)

            # vgg_output = vgg_4(vgg_output)
            # input_v = Variable(vgg_output.detach().data, requires_grad = True)
            # output_4 = netStats_4(input_v)

            # vgg_output = vgg_5(vgg_output)
            # input_v = Variable(vgg_output.detach().data, requires_grad = True)
            # output_5 = netStats_5(input_v)

            output_dis = netDis(output_1, output_2, output_3)

            loss_dis = cel_criterion(output_dis, Variable(labels_data))
            loss_dis_data = loss_dis.data[0]

            loss_dis.backward()

            optimizerDis.step()
            # optimizer_stats_5.step()
            # optimizer_stats_4.step()
            optimizer_stats_3.step()
            optimizer_stats_2.step()
            optimizer_stats_1.step()

            loss_cum += loss_dis_data
            count += 1

    ########################################################################
            if opt.debug == 1:
                print('[Debug] %d - %d - Loss: %.4f' % (epoch, count, loss_dis_data))

            text_train_detail.write('%d\t%d\t%.4f\n' % (epoch, count, loss_dis_data))
            text_train_detail.flush()

        loss_cum /= count
        print('Epoch %d - Loss: %.4f' % (epoch, loss_cum))

        text_train.write('%d\t%.4f\n' % (epoch, loss_cum))
        text_train.flush()

        gc.collect()

        # do checkpointing & validation
        if epoch % opt.checkpoint == 0:
            torch.save(netStats_1.state_dict(), '%s/stats_1_%d.pth' % (model_path, epoch))
            torch.save(netStats_2.state_dict(), '%s/stats_2_%d.pth' % (model_path, epoch))
            torch.save(netStats_3.state_dict(), '%s/stats_3_%d.pth' % (model_path, epoch))
            # torch.save(netStats_4.state_dict(), '%s/stats_4_%d.pth' % (model_path, epoch))
            # torch.save(netStats_5.state_dict(), '%s/stats_5_%d.pth' % (model_path, epoch))
            torch.save(netDis.state_dict(), '%s/dis_%d.pth' % (model_path, epoch))

            if opt.val:
                netStats_1.eval()
                netStats_2.eval()
                netStats_3.eval()
                #netStats_4.eval()
                #netStats_5.eval()
                netDis.eval()

                tol_label = np.array([], dtype=np.float)
                tol_pred = np.array([], dtype=np.float)

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

                    for i in range(output_dis.shape[0]):
                        if output_dis[i,1] >= output_dis[i,0]:
                            output_pred[i] = 1.0
                        else:
                            output_pred[i] = 0.0

                    tol_label = np.concatenate((tol_label, img_label))
                    tol_pred = np.concatenate((tol_pred, output_pred))


                acc = metrics.accuracy_score(tol_label, tol_pred)
                print('Val accuracy: %.4f' % (acc))
                text_train.write('%.4f\n' % (acc))

                netStats_1.train(mode=True)
                netStats_2.train(mode=True)
                netStats_3.train(mode=True)
                #netStats_4.train(mode=True)
                #netStats_5.train(mode=True)
                netDis.train(mode=True)

    text_train_detail.close()
    text_train.close()
