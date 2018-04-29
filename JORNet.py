import HALNet as HALNet
from HALNet import HALNet as HALNet_class
import torch.nn as nn
from magic import cudafy

class JORNet(HALNet_class):
    innerprod1_size = 256 * 16 * 16
    crop_res = (128, 128)
    #innerprod1_size = 65536

    def __init__(self, params_dict):
        super(JORNet, self).__init__(params_dict)

        self.main_loss_conv = cudafy(HALNet.HALNetConvBlock(
                kernel_size=3, stride=1, filters=21, in_channels=256, padding=1),
            self.use_cuda)
        self.main_loss_deconv1 = cudafy(nn.Upsample(size=self.crop_res, mode='bilinear'), self.use_cuda)
        #self.main_loss_deconv2 = cudafy(nn.Upsample(scale_factor=1, mode='bilinear'),
        #                                self.use_cuda)
        if self.cross_entropy:
            self.softmax_final = cudafy(HALNet.
                                        SoftmaxLogProbability2D(), self.use_cuda)
        self.innerproduct1 = cudafy(
            nn.Linear(in_features=self.innerprod1_size, out_features=200), self.use_cuda)
        self.innerproduct2 = cudafy(
            nn.Linear(in_features=200, out_features=self.num_joints * 3), self.use_cuda)
        self.softmax_final = cudafy(HALNet.SoftmaxLogProbability2D(), self.use_cuda)

    def forward_main_loss(self, conv4fout):
        # main loss
        out = self.main_loss_conv(conv4fout)
        out = self.main_loss_deconv1(out)
        out_main_heatmaps = out
        # main loss
        if self.cross_entropy:
            out_main_heatmaps = self.softmax_final(out)
        # joints output
        # this view is necessary to make layers agree
        conv4fout = conv4fout.view(-1, self.innerprod1_size)
        out_joints1 = self.innerproduct1(conv4fout)
        out_joints2 = self.innerproduct2(out_joints1)
        out_main_joints = out_joints2
        return out_main_heatmaps, out_main_joints

    def forward(self, x):
        # get subhalnet outputs (common to JORNet)
        out_intermed1, out_intermed2, out_intermed3, conv4fout =\
            super(JORNet, self).forward_subnet(x)
        # get main outputs (heatmaps and joints position)
        out_main_heatmap, out_main_joints = self.forward_main_loss(conv4fout)
        return out_intermed1, out_intermed2, out_intermed3,\
               out_main_heatmap, out_main_joints
