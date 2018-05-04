import HALNet as HALNet
from HALNet import HALNet as HALNet_class
import torch.nn as nn
from magic import cudafy
import torch.nn.functional as F
import torch


class SoftmaxLogProbability1D(torch.nn.Module):
    def __init__(self):
        super(SoftmaxLogProbability1D, self).__init__()

    def forward(self, x):
        orig_shape = x.data.shape
        seq_x = []
        for channel_ix in range(orig_shape[1]):
            softmax_ = F.softmax(x[:, channel_ix, :].contiguous()
                                 .view((orig_shape[0], orig_shape[2])), dim=1) \
                .view((orig_shape[0], orig_shape[2]))
            seq_x.append(softmax_.log())
        x = torch.stack(seq_x, dim=1)
        return x

class HALNet_prior(HALNet_class):
    prior_size = (210, 300)



    def __init__(self, params_dict):
        super(HALNet_prior, self).__init__(params_dict)

        if self.cross_entropy:
            self.softmax_final = cudafy(HALNet.
                                        SoftmaxLogProbability2D(), self.use_cuda)

        self.softmax_final = cudafy(HALNet.SoftmaxLogProbability2D(), self.use_cuda)

        self.map_out_conv_prior = cudafy(HALNet.HALNetConvBlock(
            kernel_size=3, stride=1, filters=1, in_channels=256, padding=1),
            self.use_cuda)
        self.main_loss_deconv_prior1 = cudafy(nn.Upsample(size=self.prior_size, mode='bilinear'), self.use_cuda)
        self.softmax_1d = cudafy(SoftmaxLogProbability1D(), self.use_cuda)

    def forward_prior(self, conv4fout):
        out_prior = self.map_out_conv_prior(conv4fout)
        out_prior = self.main_loss_deconv_prior1(out_prior)
        out_prior = out_prior.view((out_prior.shape[0], out_prior.shape[2], out_prior.shape[3]))
        if self.cross_entropy:
            out_prior = cudafy(self.softmax_1d(out_prior))
        return out_prior

    def forward(self, x):
        # get subhalnet outputs (common to JORNet)
        out_intermed1, out_intermed2, out_intermed3, conv4fout = self.forward_subnet(x)
        # out to main loss of halnet
        out_main = self.forward_main_loss(conv4fout)
        # out to prior
        out_prior = self.forward_prior(conv4fout)
        return out_intermed1, out_intermed2, out_intermed3, out_main, out_prior