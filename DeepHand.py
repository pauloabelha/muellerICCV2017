import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn.init as init
import numpy as np
import math

#scale_factor = Variable(torch.FloatTensor[2], requires_grad=True)
        #self.scale1 = self.conv1 * scale_factor

def _print_layer_output_shape(layer_name, output_shape):
    print("Layer " + layer_name + " output shape: " + str(output_shape))

def DeepHandConvBlock(kernel_size, stride, filters, in_channels, padding=0):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=filters)
        )

def DeepHandResConvSequence(stride, filters1, filters2,
                          padding1=1, padding2=0, padding3=0,
                          first_in_channels=0):
    if first_in_channels == 0:
        first_in_channels = filters1
    return nn.Sequential(
        # added padding = 1 to make shapes fit when joining
        # with left module
        DeepHandConvBlock(kernel_size=1, stride=stride, filters=filters1,
                        in_channels=first_in_channels, padding=padding1),
        nn.ReLU(),
        DeepHandConvBlock(kernel_size=3, stride=1, filters=filters1,
                        in_channels=filters1, padding=padding2),
        nn.ReLU(),
        DeepHandConvBlock(kernel_size=1, stride=1, filters=filters2,
                        in_channels=filters1, padding=padding3)
    )

class DeepHandResBlockIDSkip(nn.Module):
    def __init__(self, filters1, filters2,
                 padding_right1=1, padding_right2=0, padding_right3=0):
        super(DeepHandResBlockIDSkip, self).__init__()
        self.right_res = DeepHandResConvSequence(stride=1,
                                               filters1=filters1,
                                               filters2=filters2,
                                               padding1=padding_right1,
                                               padding2=padding_right2,
                                               padding3=padding_right3,
                                               first_in_channels=
                                               filters2)
        self.relu = nn.ReLU()

    def forward(self, input):
        left_res = input
        right_res = self.right_res(input)
        # element-wise sum
        out = left_res + right_res
        out = self.relu(out)
        return out

class DeepHandResBlockConv(nn.Module):
    def __init__(self, stride, filters1, filters2, first_in_channels=0,
                 padding_left=0, padding_right1=0, padding_right2=0,
                 padding_right3=0):
        super(DeepHandResBlockConv, self).__init__()
        if first_in_channels == 0:
            first_in_channels = filters1
        self.left_res = DeepHandConvBlock(kernel_size=1, stride=stride,
                                        filters=filters2,
                                        padding=padding_left,
                                        in_channels=first_in_channels)
        self.right_res = DeepHandResConvSequence(stride=stride,
                                               filters1=filters1,
                                               filters2=filters2,
                                               padding1=padding_right1,
                                               padding2=padding_right2,
                                               padding3=padding_right3,
                                               first_in_channels=
                                               first_in_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        left_res = self.left_res(input)
        right_res = self.right_res(input)
        # element-wise sum
        out = left_res + right_res
        out = self.relu(out)
        return out

def make_bilinear_weights(size, num_channels):
    ''' Make a 2D bilinear kernel suitable for upsampling
    Stack the bilinear kernel for application to tensor '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, 1, size, size)
    for i in range(num_channels):
        w[i, 0] = filt
    return w


class SoftmaxLogProbability2D(torch.nn.Module):
    def __init__(self):
        super(SoftmaxLogProbability2D, self).__init__()

    def forward(self, x):
        orig_shape = x.data.shape
        seq_x = []
        for channel_ix in range(orig_shape[1]):
            softmax_ = F.softmax(x[:, channel_ix, :, :].contiguous()
                                 .view((orig_shape[0], orig_shape[2] * orig_shape[3])), dim=1)\
                .view((orig_shape[0], orig_shape[2], orig_shape[3]))
            seq_x.append(softmax_.log())
        x = torch.stack(seq_x, dim=1)
        return x

def cudafy(object):
    return object.cuda()

class DeepHand(nn.Module):

    joint_ixs = []
    VERBOSE = False
    WEIGHT_LOSS_INTERMED1 = 0.5
    WEIGHT_LOSS_INTERMED2 = 0.5
    WEIGHT_LOSS_INTERMED3 = 0.5
    WEIGHT_LOSS_MAIN = 1

    def __init__(self, joint_ixs, use_cuda=True):
        super(DeepHand, self).__init__()
        self.joint_ixs = joint_ixs
        self.conv1 = DeepHandConvBlock(kernel_size=7, stride=1, filters=64,
                                     in_channels=4, padding=3)
        if use_cuda:
            self.conv1 = cudafy(self.conv1)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if use_cuda:
            self.mp1 = cudafy(self.mp1)
        self.res2a = DeepHandResBlockConv(stride=1, filters1=64, filters2=256,
                                        padding_right1=1)
        if use_cuda:
            self.res2a = cudafy(self.res2a)
        self.res2b = DeepHandResBlockIDSkip(filters1=64, filters2=256)
        if use_cuda:
            self.res2b = cudafy(self.res2b)
        self.res2c = DeepHandResBlockIDSkip(filters1=64, filters2=256)
        if use_cuda:
            self.res2c = cudafy(self.res2c)
        self.res3a = DeepHandResBlockConv(stride=2, filters1=128, filters2=512,
                                        padding_right3=1,
                                        first_in_channels=256)
        if use_cuda:
            self.res3a = cudafy(self.res3a)
        self.interm_loss1 = DeepHandConvBlock(kernel_size=3, stride=1,
                                              filters=1, in_channels=512,
                                              padding=1)
        if use_cuda:
            self.interm_loss1 = cudafy(self.interm_loss1)
        self.interm_loss1_deconv = nn.Upsample(scale_factor=4, mode='bilinear')
        if use_cuda:
            self.interm_loss1_deconv = cudafy(self.interm_loss1_deconv)
        self.interm_loss1_softmax = SoftmaxLogProbability2D()
        if use_cuda:
            self.interm_loss1_softmax = cudafy(self.interm_loss1_softmax)
        self.res3b = DeepHandResBlockIDSkip(filters1=128, filters2=512)
        if use_cuda:
            self.res3b = cudafy(self.res3b)
        self.res3c = DeepHandResBlockIDSkip(filters1=128, filters2=512)
        if use_cuda:
            self.res3c = cudafy(self.res3c)
        self.res4a = DeepHandResBlockConv(stride=2, filters1=256, filters2=1024,
                                        padding_right3=1,
                                        first_in_channels=512)
        if use_cuda:
            self.res4a = cudafy(self.res4a)
        self.interm_loss2 = DeepHandConvBlock(kernel_size=3, stride=1,
                                            filters=1, in_channels=1024,
                                            padding=1)
        if use_cuda:
            self.interm_loss2 = cudafy(self.interm_loss2)
        self.interm_loss2_deconv = nn.Upsample(scale_factor=8, mode='bilinear')
        if use_cuda:
            self.interm_loss2_deconv = cudafy(self.interm_loss2_deconv)
        self.interm_loss2_softmax = SoftmaxLogProbability2D()
        if use_cuda:
            self.interm_loss2_softmax = cudafy(self.interm_loss2_softmax)
        self.res4b = DeepHandResBlockIDSkip(filters1=256, filters2=1024)
        if use_cuda:
            self.res4b = cudafy(self.res4b)
        self.res4c = DeepHandResBlockIDSkip(filters1=256, filters2=1024)
        if use_cuda:
            self.res4c = cudafy(self.res4c)
        self.res4d = DeepHandResBlockIDSkip(filters1=256, filters2=1024)
        if use_cuda:
            self.res4d = cudafy(self.res4d)
        self.conv4e = DeepHandConvBlock(kernel_size=3, stride=1, filters=512,
                                     in_channels=1024, padding=1)
        if use_cuda:
            self.conv4e = cudafy(self.conv4e)
        self.interm_loss3 = DeepHandConvBlock(kernel_size=3, stride=1,
                                            filters=1, in_channels=512,
                                            padding=1)
        if use_cuda:
            self.interm_loss3 = cudafy(self.interm_loss3)
        self.interm_loss3_deconv = nn.Upsample(scale_factor=8, mode='bilinear')
        if use_cuda:
            self.interm_loss3_deconv = cudafy(self.interm_loss3_deconv)
        self.interm_loss3_softmax = SoftmaxLogProbability2D()
        if use_cuda:
            self.interm_loss3_softmax = cudafy(self.interm_loss3_softmax)
        self.conv4f = DeepHandConvBlock(kernel_size=3, stride=1, filters=256,
                                      in_channels=512, padding=1)
        if use_cuda:
            self.conv4f = cudafy(self.conv4f)
        NUM_HEATMAPS = len(self.joint_ixs)
        self.main_loss_conv = DeepHandConvBlock(kernel_size=3, stride=1,
                                              filters=NUM_HEATMAPS, in_channels=256,
                                              padding=1)
        if use_cuda:
            self.main_loss_conv = cudafy(self.main_loss_conv)
        self.main_loss_deconv = nn.Upsample(scale_factor=8, mode='bilinear')
        if use_cuda:
            self.main_loss_deconv = cudafy(self.main_loss_deconv)
        self.softmax_final = SoftmaxLogProbability2D()
        if use_cuda:
            self.softmax_final = cudafy(self.softmax_final)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mp1(out)
        out = self.res2a(out)
        out = self.res2b(out)
        out = self.res2c(out)
        res3aout = self.res3a(out)
        out = self.res3b(res3aout)
        out = self.res3c(out)
        res4aout = self.res4a(out)
        out = self.res4b(res4aout)
        out = self.res4c(out)
        out = self.res4d(out)
        conv4eout = self.conv4e(out)
        out = self.conv4f(conv4eout)
        out = self.main_loss_conv(out)
        out = self.main_loss_deconv(out)
        # main loss
        out_main = self.softmax_final(out)
        #intermediate losses
        # intermed 1
        #out_intermed1 = self.interm_loss1(res3aout)
        #out_intermed1 = self.interm_loss1_deconv(out_intermed1)
        #out_intermed1 = self.interm_loss1_softmax(out_intermed1)
        # intermed 2
        #out_intermed2 = self.interm_loss2(res4aout)
        #out_intermed2 = self.interm_loss2_deconv(out_intermed2)
        #out_intermed2 = self.interm_loss2_softmax(out_intermed2)
        # intermed 3
        #out_intermed3 = self.interm_loss3(conv4eout)
        #out_intermed3 = self.interm_loss3_deconv(out_intermed3)
        #out_intermed3 = self.interm_loss3_softmax(out_intermed3)
        # return net
        #return out_intermed1, out_intermed2, out_intermed3, out_main
        return out_main
