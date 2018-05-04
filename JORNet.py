import HALNet as HALNet
from HALNet import HALNet as HALNet_class
import torch.nn as nn
from magic import cudafy

class JORNet(HALNet_class):
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
        self.innerproduct1_joint1 = cudafy(
            nn.Linear(in_features=524288, out_features=200), self.use_cuda)
        self.innerproduct2_joint1 = cudafy(
            nn.Linear(in_features=200, out_features=self.num_joints * 3), self.use_cuda)

        self.innerproduct1_joint2 = cudafy(
            nn.Linear(in_features=262144, out_features=200), self.use_cuda)
        self.innerproduct2_joint2 = cudafy(
            nn.Linear(in_features=200, out_features=self.num_joints * 3), self.use_cuda)

        self.innerproduct1_joint3 = cudafy(
            nn.Linear(in_features=131072, out_features=200), self.use_cuda)
        self.innerproduct2_joint3 = cudafy(
            nn.Linear(in_features=200, out_features=self.num_joints * 3), self.use_cuda)

        self.innerproduct1_joint_main = cudafy(
            nn.Linear(in_features=65536, out_features=200), self.use_cuda)
        self.innerproduct2_join_main = cudafy(
            nn.Linear(in_features=200, out_features=self.num_joints * 3), self.use_cuda)

    def forward(self, x):
        out_intermed_hm1, out_intermed_hm2, out_intermed_hm3, conv4fout, \
        res3aout, res4aout, conv4eout = self.forward_subnet(x)
        out_intermed_hm_main = self.forward_main_loss(conv4fout)

        innerprod1_size = res3aout.shape[1] * res3aout.shape[2] * res3aout.shape[3]
        out_intermed_j1 = res3aout.view(-1, innerprod1_size)
        out_intermed_j1 = self.innerproduct1_joint1(out_intermed_j1)
        out_intermed_j1 = self.innerproduct2_joint1(out_intermed_j1)

        innerprod1_size = res4aout.shape[1] * res4aout.shape[2] * res4aout.shape[3]
        out_intermed_j2 = res4aout.view(-1, innerprod1_size)
        out_intermed_j2 = self.innerproduct1_joint2(out_intermed_j2)
        out_intermed_j2 = self.innerproduct2_joint2(out_intermed_j2)

        innerprod1_size = conv4eout.shape[1] * conv4eout.shape[2] * conv4eout.shape[3]
        out_intermed_j3 = conv4eout.view(-1, innerprod1_size)
        out_intermed_j3 = self.innerproduct1_joint3(out_intermed_j3)
        out_intermed_j3 = self.innerproduct2_joint3(out_intermed_j3)

        innerprod1_size = conv4fout.shape[1] * conv4fout.shape[2] * conv4fout.shape[3]
        out_intermed_j_main = conv4fout.view(-1, innerprod1_size)
        out_intermed_j_main = self.innerproduct1_joint_main(out_intermed_j_main)
        out_intermed_j_main = self.innerproduct2_join_main(out_intermed_j_main)

        return out_intermed_hm1, out_intermed_hm2, out_intermed_hm3, out_intermed_hm_main,\
               out_intermed_j1, out_intermed_j2, out_intermed_j3, out_intermed_j_main
