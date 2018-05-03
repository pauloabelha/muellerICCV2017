import HALNet as HALNet
from HALNet import HALNet as HALNet_class
import torch.nn as nn
from magic import cudafy

class HALNet_prior(HALNet_class):
    innerprod1_size = 256 * 16 * 16
    crop_res = (128, 128)
    #innerprod1_size = 65536

    def map_out_to_loss(self, innerprod1_size):
        return cudafy(nn.Linear(in_features=innerprod1_size, out_features=200), self.use_cuda)

    def map_out_conv(self, in_channels):
        return cudafy(HALNet.HALNetConvBlock(
            kernel_size=3, stride=1, filters=21, in_channels=in_channels, padding=1),
            self.use_cuda)

    def __init__(self, params_dict):
        super(HALNet_prior, self).__init__(params_dict)

        self.main_loss_conv = cudafy(HALNet.HALNetConvBlock(
                kernel_size=3, stride=1, filters=21, in_channels=256, padding=1),
            self.use_cuda)
        self.main_loss_deconv1 = cudafy(nn.Upsample(size=self.crop_res, mode='bilinear'), self.use_cuda)
        if self.cross_entropy:
            self.softmax_final = cudafy(HALNet.
                                        SoftmaxLogProbability2D(), self.use_cuda)

        self.softmax_final = cudafy(HALNet.SoftmaxLogProbability2D(), self.use_cuda)

    def forward_loss(self, out):
        # main loss
        out = self.map_out_conv(out.shape[1])(out)
        out = self.main_loss_deconv1(out)
        out_main_heatmaps = out
        # main loss
        if self.cross_entropy:
            out_main_heatmaps = self.softmax_final(out)
        # prior output
        out_prior = self.map_prior_conv(out)
        return out_main_heatmaps, out_prior

    def forward(self, x):
        res3aout, res4aout, conv4eout, conv4fout = self.forward_common_net(x)
        # losses
        out_intermed_hm1, out_intermed_j1 = self.forward_loss(res3aout)
        out_intermed_hm2, out_intermed_j2 = self.forward_loss(res4aout)
        out_intermed_hm3, out_intermed_j3 = self.forward_loss(conv4eout)
        out_intermed_hm_main, out_intermed_j_main = self.forward_loss(conv4fout)
        return out_intermed_hm1, out_intermed_hm2, out_intermed_hm3, out_intermed_hm_main,\
               out_intermed_j1, out_intermed_j2, out_intermed_j3, out_intermed_j_main
