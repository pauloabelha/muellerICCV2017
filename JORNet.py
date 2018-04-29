import HALNet
import torch.nn as nn

class JORNet(HALNet):

    def __init__(self, params_dict):
        super(JORNet, self).__init__(params_dict)

        innerproduct_required_size = 256 * 30 * 40

        self.main_loss_conv = super(JORNet, self).cudafy(
            super(JORNet, self).HALNetConvBlock(
                kernel_size=3, stride=1, filters=64, in_channels=256, padding=1),
            self.use_cuda)
        self.main_loss_deconv1 = super(JORNet, self).cudafy(
            nn.Upsample(scale_factor=8), self.use_cuda)
        self.main_loss_deconv2 = super(JORNet, self).cudafy(
            nn.Upsample(scale_factor=8, mode='bilinear'), self.use_cuda)
        if self.cross_entropy:
            self.softmax_final = super(JORNet, self).cudafy(
                super(JORNet, self).SoftmaxLogProbability2D(), self.use_cuda)
        self.innerproduct1 = super(JORNet, self).cudafy(
            nn.Linear(in_features=self.innerproduct_required_size, out_features=200))
        self.innerproduct2 = super(JORNet, self).cudafy(
            nn.Linear(in_features=200, out_features=self.num_joints * 3))

    def forward_main_loss(self, conv4fout):
        # main loss
        out = self.main_loss_conv(conv4fout)
        out = self.main_loss_deconv1(out)
        out = self.main_loss_deconv2(out)
        out = self.main_loss_deconv1(out)
        out_main_heatmaps = out
        # main loss
        if self.cross_entropy:
            out_main_heatmaps = super(JORNet, self).softmax_final(out)
        # joints output
        # this view is necessary to make layers agree
        conv4fout = conv4fout.view(-1, self.innerproduct_required_size)
        out_joints1 = self.innerproduct1(conv4fout)
        out_joints2 = self.innerproduct2(out_joints1)
        out_main_joints = out_joints2
        return out_main_heatmaps, out_main_joints

    def forward(self, x):
        # get subhalnet outputs (common to JORNet)
        out_intermed1, out_intermed2, out_intermed3, conv4fout =\
            super(JORNet, self).forward_subnet(x)
        # get main outputs (heatmaps and joints position)
        out_main_heatmap, out_main_joints = self.forward_main_loss(x)
        return out_intermed1, out_intermed2, out_intermed3,\
               out_main_heatmap, out_main_joints
