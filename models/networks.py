import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
#from modules import ConvOffset3d
import numpy as np

from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
from libs.conv_offset2D import ConvOffset2D, th_batch_map_offsets, th_generate_grid
from deform_conv.modules import ConvOffset2d
import config as cfg

from ConvLSTM import *

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type = 'instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'batch3d':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance3d':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

#####################
# Content Encoder
#####################
def content_E(input_nc, output_nc, content_nc, latent_nc, ngf, which_model_E, norm='batch', use_dropout=False,gpu_ids=[],init_type='normal'):
    netCE = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type = norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_E == 'resnet_6blocks':
        netCE = ContentEncoder(input_nc, output_nc, ngf, norm_layer = norm_layer, use_dropout = use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_E == 'resnet_3blocks':
        netCE = ContentEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer,use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    elif which_model_E == 'resnet_1blocks':
        netCE = ContentEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer,use_dropout=use_dropout, n_blocks=1, gpu_ids=gpu_ids)
    elif which_model_E == 'unet_64':
        netCE = UnetEncoder(input_nc, output_nc, 4, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_E == 'unet_128':
        netCE = UnetEncoder(input_nc, output_nc, 5, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_E == 'unet_128_G':
        netCE = UnetGenerator(input_nc, output_nc, content_nc, latent_nc, 7, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_E == 'unet_256_G':
        netCE = UnetGenerator(input_nc, output_nc, content_nc, latent_nc, 8, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_E)
    if len(gpu_ids) > 0:
        #netCE = torch.nn.DataParallel(netCE.cuda(), device_ids=gpu_ids)
        netCE.cuda(device_id=gpu_ids[0])
    #netCE.apply(weights_init)
    init_weights(netCE, init_type=init_type)
    return netCE

#####################
# Content Encoder
#####################
def content_E_3D(input_nc, output_nc, ngf, which_model_E, norm='batch3d', use_dropout=False,gpu_ids=[],init_type='normal'):
    netCE = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type = norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_E == 'unet_128_3D':
        netCE = UnetEncoder3D_128(input_nc, output_nc, 7, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_E)
    if len(gpu_ids) > 0:
        #netCE = torch.nn.DataParallel(netCE.cuda(), device_ids=gpu_ids)
        netCE.cuda(device_id=gpu_ids[0])
    #netCE.apply(weights_init)
    init_weights(netCE, init_type=init_type)
    return netCE


def seq_E(shape,input_nc, ngf, norm='batch', use_dropout=False,gpu_ids=[],init_type='normal'):
    netSE = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type = norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netSE = SeqEncoder(shape,input_nc,3,ngf,norm_layer=norm_layer,use_dropout=use_dropout,gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        #netCE = torch.nn.DataParallel(netCE.cuda(), device_ids=gpu_ids)
        netSE.cuda(device_id=gpu_ids[0])
    #netSE.apply(weights_init)
    init_weights(netSE, init_type=init_type)
    return netSE

#####################
# Motion Encoder
#####################
def motion_E(input_nc, output_nc, ngf, which_model_E, norm='batch', use_dropout=False,gpu_ids=[],init_type='normal'):
    netME = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type = norm)
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_E == 'resnet_6blocks':
        netME = MotionEncoder(input_nc, output_nc,8, ngf, norm_layer=norm_layer,use_dropout=use_dropout,n_blocks=6,gpu_ids=gpu_ids)
    elif which_model_E == 'resnet_3blocks':
        netME = MotionEncoder(input_nc, output_nc,7, ngf, norm_layer=norm_layer,use_dropout=use_dropout,n_blocks=3,gpu_ids=gpu_ids)
    elif which_model_E == 'resnet_1blocks':
        netME = MotionEncoder(input_nc, output_nc,7, ngf, norm_layer=norm_layer,use_dropout=use_dropout,n_blocks=1,gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_E)
    if len(gpu_ids) > 0:
        #netME = torch.nn.DataParallel(netME.cuda(), device_ids=gpu_ids)
        netME.cuda(device_id=gpu_ids[0])
    #netME.apply(weights_init)
    init_weights(netME, init_type=init_type)
    return netME

#####################
# Motion Predictor
#####################
def motion_P(shape,hist_len,input_nc, output_nc, ngf, norm='batch', use_dropout=False,gpu_ids=[],init_type='normal'):
    netMP = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type = norm)
    if use_gpu:
        assert(torch.cuda.is_available())

    netMP=SeqPredictor(shape,hist_len,input_nc,output_nc,ngf,norm_layer=norm_layer,use_dropout=use_dropout,gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        #netMP = torch.nn.DataParallel(netMP.cuda(), device_ids=gpu_ids)
        netMP.cuda(device_id=gpu_ids[0])
    #netMP.apply(weights_init)
    init_weights(netMP, init_type=init_type)
    return netMP

#####################
# Offsets Predictor
#####################
def offsets_P(shape,hist_len,input_nc, output_nc, ngf, norm='batch', use_dropout=False,gpu_ids=[],init_type='normal',relu='tanh',groups=1):
    netOP = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type = norm)
    if use_gpu:
        assert(torch.cuda.is_available())

    netOP = OffsetsPredictor(shape,hist_len,input_nc,output_nc,ngf,norm_layer=norm_layer,use_dropout=use_dropout,gpu_ids=gpu_ids,relu=relu,groups=groups)

    if len(gpu_ids) > 0:
        #netMP = torch.nn.DataParallel(netMP.cuda(), device_ids=gpu_ids)
        netOP.cuda(device_id=gpu_ids[0])
    init_weights(netOP, init_type=init_type)
    return netOP

#####################
# Overall Generator
#####################
def define_G(shape,hist_len,input_nc,latent_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False,gpu_ids=[],init_type='normal'):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type = norm)
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netG == 'Sequential':
        netG=SeqDecoder(shape,hist_len,input_nc,output_nc,ngf,norm_layer=norm_layer,use_dropout=use_dropout,gpu_ids=gpu_ids)
    elif which_model_netG == 'Standalone':
        netG=Decoder(input_nc,output_nc,ngf,norm_layer=norm_layer,use_dropout=use_dropout,gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_64':
        netG = UnetDecoder(input_nc, latent_nc, output_nc, 4, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetDecoder(input_nc, latent_nc,output_nc, 5, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' %
                                  which_model_netG)
    if len(gpu_ids) > 0:
        #netG = torch.nn.DataParallel(netG.cuda(), device_ids=gpu_ids)
        netG.cuda(device_id=gpu_ids[0])
    #netG.apply(weights_init)
    init_weights(netG, init_type=init_type)
    return netG

#####################
# Overall Generator
#####################
def define_G_3D(shape,hist_len,input_nc,latent_nc, output_nc, ngf, which_model_netG, norm='batch3d', use_dropout=False,gpu_ids=[],init_type='normal'):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type = norm)
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netG == 'unet_128_3D_0':
        netG = UnetDecoder3D_128(input_nc, 0,output_nc, 7, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128_3D':
        netG = UnetDecoder3D_128(input_nc, latent_nc,output_nc, 7, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' %
                                  which_model_netG)
    if len(gpu_ids) > 0:
        #netG = torch.nn.DataParallel(netG.cuda(), device_ids=gpu_ids)
        netG.cuda(device_id=gpu_ids[0])
    #netG.apply(weights_init)
    init_weights(netG, init_type=init_type)
    return netG

#####################
# Overall Descrimenator
#####################
def define_D(isize, input_nc,feat_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[],init_type='normal'):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(isize, input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(isize, input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'DCGAN':
        netD = DCGAN_D(isize, input_nc, ndf, 0, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'Multi_D':
        netD = Multi_D(isize, input_nc, feat_nc, ndf, 0, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        #netD = torch.nn.DataParallel(netD.cuda(), device_ids=gpu_ids)
        netD.cuda(device_id=gpu_ids[0])
    #netD.apply(weights_init)
    init_weights(netD, init_type=init_type)
    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
################################################################################
# Classes
################################################################################
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):

        #print input.size()
        target_tensor = self.get_target_tensor(input, target_is_real)
        #print target_tensor.size()
        return self.loss(input, target_tensor)

class GDLLoss(nn.Module):
    def __init__(self,input_nc, tensor=torch.FloatTensor):
        super(GDLLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.alpha = 0.5
        self.Tensor = tensor
        self.dx_filter = nn.Conv2d(input_nc,input_nc,kernel_size=(1,2),stride=1,padding=(0,0),bias=False,groups=input_nc).cuda()
        self.dy_filter = nn.Conv2d(input_nc,input_nc,kernel_size=(2,1),stride=1,padding=(0,0),bias=False,groups=input_nc).cuda()
        wx_tensor = self.Tensor(self.dx_filter.weight.size())
        wy_tensor = self.Tensor(self.dy_filter.weight.size())
        wx_tensor[:,:,:,0] = -1
        wx_tensor[:,:,:,1] = 1
        wy_tensor[:,:,0,:] = 1
        wy_tensor[:,:,1,:] = -1
        self.dx_filter.weight.data.copy_(wx_tensor)
        self.dy_filter.weight.data.copy_(wy_tensor)
        for param in self.dx_filter.parameters():
            param.requires_grad = False
        for param in self.dy_filter.parameters():
            param.requires_grad = False

    def __call__(self,gen,gt):
        gen_dx = torch.abs(self.dx_filter(gen))
        gen_dy = torch.abs(self.dy_filter(gen))
        gt_dx = torch.abs(self.dx_filter(gt))
        gt_dy = torch.abs(self.dy_filter(gt))
        #grad_diff_x = torch.abs(gt_dx - gen_dx)
        #grad_diff_y = torch.abs(gt_dy - gen_dy)
        return self.loss(gen_dx,gt_dx)*self.alpha + self.loss(gen_dy,gt_dy)*self.alpha

class TripLoss(nn.Module):
    def __init__(self,p=2):
        super(TripLoss, self).__init__()
        if p == 2:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()
    def __call__(self,a,p,n):
        p = Variable(p.data)
        n = Variable(n.data)
        dp = self.loss(a,p)
        dn = self.loss(a,n)

        return -torch.log(dn/(dn+dp+1e-20))
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ContentEncoder(nn.Module):
    def __init__(self,input_nc,output_nc,ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False,n_blocks=6,gpu_ids=[],padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ContentEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        #model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf, kernel_size=3,padding=0,bias=use_bias),norm_layer(ngf),nn.ReLU(True)]
        n_downsampling = 4
        # 64 --> 8, 32-->4
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1)]
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2,padding=0,dilation=1,bias=use_bias),
                      norm_layer(ngf * mult *2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf*mult,ngf*mult,padding_type=padding_type,norm_layer=norm_layer,use_dropout=use_dropout,use_bias=use_bias)]

        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf*mult,output_nc,kernel_size=3,padding=0)]
        model += [nn.ReLU(True)]
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    # output size osize = (isize/2**n_downsampling)
    # output batchSize*output_nc*osize*osize
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data,torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        output = output.unsqueeze(1) # [bs, oc, s0, s1] -> [bs,1,oc,s0,s1]
        return output

# Defines the seqEncoder
# https://github.com/jcjohnson/fast-neural-style/
class SeqEncoder(nn.Module):
    def __init__(self,shape,input_nc,filter_size,ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False,gpu_ids=[],padding_type='reflect'):
        super(SeqEncoder, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.conv_lstm = CLSTM(shape, input_nc, filter_size, ngf,1,use_bias)
        model = [nn.ReflectionPad2d(1), nn.Conv2d(ngf, ngf, kernel_size=3,padding=0,bias=use_bias),norm_layer(ngf),nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    # output size osize = (isize/2**n_downsampling)
    # output batchSize*output_nc*osize*osize
    def forward(self,input, hidden_state):
        convlstm_out = self.conv_lstm(input, hidden_state)
        if (self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            #convlstm_out = nn.parallel.data_parallel(self.conv_lstm, input, hidden_state, self.gpu_ids)
            output = nn.parallel.data_parallel(self.model, convlstm_out[1][-1], self.gpu_ids)
        else:
            #convlstm_out = self.conv_lstm(input, hidden_state)
            output = self.model(convlstm_out[1][-1])
        #print 'conv_lstm output shape: ', convlstm_out[1].size()
        return convlstm_out[0], output

    def init_hidden(self,batch_size):
        return self.conv_lstm.init_hidden(batch_size)

# Motion Encoder
class MotionEncoder(nn.Module):
    def __init__(self,input_nc,output_nc,num_downs=7,ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False,n_blocks=3,gpu_ids=[],padding_type='reflect'):
        assert(n_blocks >= 0)
        super(MotionEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_nc, ngf, kernel_size=4,stride=2,padding=1,bias=use_bias),norm_layer(ngf),nn.LeakyReLU(0.2,True)]

        n_downsampling = num_downs - 3
        # 128-->4
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
                                stride=2,padding=1,bias=use_bias),
                      norm_layer(ngf * mult *2),
                      nn.LeakyReLU(0.2,True)]

        mult = 2**n_downsampling
        #for i in range(n_blocks):
        #   model += [ResnetBlock(ngf*mult,ngf*mult,padding_type=padding_type,norm_layer=norm_layer,use_dropout=use_dropout,use_bias=use_bias)]

        model += [nn.Conv2d(ngf*mult,output_nc,kernel_size=4,bias=use_bias),norm_layer(output_nc),nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data,torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        output = output#.unsqueeze(1) # [bs, oc, s0, s1] -> [bs,1,oc,s0,s1]
        return output


# CNN decoder
# input: encoder batchSize*(input_nc*2)*osize*osize
# output: 3*isize*isize
class Decoder(nn.Module):
    def __init__(self,input_nc,output_nc=3,ngf=16,norm_layer=nn.BatchNorm2d,use_dropout=False,gpu_ids=[],padding_type='reflect'):
        super(Decoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #8-->64, 4-->32, output seq_len, batchsize, ngf*8, H, W
        n_upsampling = 3
        mult = 2**n_upsampling
        model = []
        #model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf*mult, kernel_size=3,padding=0,bias=use_bias),norm_layer(ngf*mult),nn.ReLU(True)]
        for i in range(n_upsampling):
            #model += [nn.ReflectionPad2d(1)]
            mult = 2**(n_upsampling - i)  #8, 4, 2
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3,stride=2,
                                         padding=1,output_padding=1,
                                         bias=use_bias,dilation=1),
                      norm_layer(int(ngf * mult /2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7,padding = 0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self,input):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        return outputdisplay_id

# CNN + LSTM + TCNN decoder
# input: encoder batchSize*hist_len*(input_nc*2)*osize*osize
# output: 3*isize*isize
class SeqDecoder(nn.Module):
    def __init__(self,shape,seq_len,input_nc,output_nc=3,ngf=16,norm_layer=nn.BatchNorm2d,use_dropout=False,gpu_ids=[],padding_type='reflect'):
        super(SeqDecoder, self).__init__()
        self.seq_len = seq_len
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.shape = shape
        self.filter_size = 3
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # ConvLSTM
        filter_size=3
        nlayers=1
        #If using this format, then we need to transpose in CLSTM
        #input = Variable(torch.rand(batch_size,seq_len,inp_chans,shape[0],shape[1])).cuda()
        n_upsampling = 3
        mult = 2**n_upsampling
        self.conv_lstm = CLSTM(shape, input_nc, filter_size, ngf*mult,nlayers,use_bias)

        #8-->64, 4-->32, output seq_len, batchsize, ngf*8, H, W
        model = []
        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)  #8, 4, 2
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3,stride=2,
                                         padding=1,output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult /2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3,padding = 0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self,input, hidden_state):
        convlstm_out = self.conv_lstm(input, hidden_state)
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, convlstm_out[1][-1], self.gpu_ids)
        else:
            output = self.model(convlstm_out[1][-1])
        return convlstm_out[0], output

    def init_hidden(self,batch_size):
        return self.conv_lstm.init_hidden(batch_size)

# CNN + LSTM + TCNN Predictor
# input: encoder batchSize*hist_len*(input_nc*2)*osize*osize
# output: 3*isize*isize
class SeqPredictor(nn.Module):
    def __init__(self,shape,seq_len,input_nc,output_nc,ngf=16,norm_layer=nn.BatchNorm2d,use_dropout=False,gpu_ids=[],padding_type='reflect'):
        super(SeqPredictor, self).__init__()
        self.seq_len = seq_len
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.shape = shape
        self.filter_size = 3
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # ConvLSTM
        filter_size=3
        nlayers=2
        #If using this format, then we need to transpose in CLSTM
        #input = Variable(torch.rand(batch_size,seq_len,inp_chans,shape[0],shape[1])).cuda()
        n_upsampling = 2
        mult = 2**n_upsampling
        #print 'shape: ',shape,' ,input_nc: ', input_nc, ' ,ngf: ', ngf*mult
        self.conv_lstm = CLSTM(shape, input_nc, filter_size, ngf*mult,nlayers,use_bias)

        #8-->64, 4-->32, output seq_len, batchsize, ngf*8, H, W
        model = []
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf*4, output_nc, kernel_size=3,padding = 0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self,input,hidden_state):
        convlstm_out = self.conv_lstm(input, hidden_state)
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            #convlstm_out = nn.parallel.data_parallel(self.conv_lstm, input, hidden_state, self.gpu_ids)
            output = nn.parallel.data_parallel(self.model, convlstm_out[1][-1], self.gpu_ids)
        else:
            #convlstm_out = self.conv_lstm(input, hidden_state)
            output = self.model(convlstm_out[1][-1])

        output = output.unsqueeze(1)
        return convlstm_out[0], output #[bs,1, oc, s0, s1]

    def init_hidden(self,batch_size):
        return self.conv_lstm.init_hidden(batch_size)

# CNN + LSTM + TCNN OffsetsPredictor
# input: encoder torch.cat(feature_maps,offsets)
# output: new_offsets
class OffsetsPredictor(nn.Module):
    def __init__(self, shape, seq_len,input_nc,output_nc,ngf=16,norm_layer=nn.BatchNorm2d,use_dropout=False,gpu_ids=[],relu='leakyrelu',groups = 1,padding_type='reflect'):
        super(OffsetsPredictor, self).__init__()
        self.seq_len = seq_len
        self.input_nc = input_nc
        self.output_nc = output_nc
        #assert(input_nc / output_nc == groups)
        #self.seperate_groups = input_nc/output_nc 
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.shape = shape
        self.filter_size = 3
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self._grid_param = None

        if relu == 'relu':
            self.relu = nn.ReLU(True)
        elif relu == 'leakyrelu':
            self.relu = nn.LeakyReLU(0.2,True)
        elif relu == 'tanh':
            self.relu = nn.Tanh()
        # ConvLSTM
        filter_size=3
        nlayers = 1
        self.groups = groups
        #If using this format, then we need to transpose in CLSTM
        '''offset_nc = self.groups*2*filter_size*filter_size
        self.offset_nc = offset_nc
        self.shape = shape
        mult = 4
        #print 'shape: ',shape,' ,input_nc: ', input_nc, ' ,ngf: ', ngf*mult
        self.conv_lstm = CLSTM(shape, input_nc+offset_nc, filter_size, offset_nc, nlayers,use_bias)
        #self.conv_lstm = CLSTM(shape, input_nc*3, filter_size, output_nc*2, nlayers,use_bias)
        self.conv_1 = nn.Sequential(nn.Conv2d(
            offset_nc,
            offset_nc,
            kernel_size=(filter_size, filter_size),
            stride=(1, 1),
            padding=(1, 1),
            bias=False),nn.Tanh())
        self.conv_offset_1 = ConvOffset2d(input_nc,
            input_nc, (filter_size, filter_size),
            stride=1,
            padding=1,
            num_deformable_groups=num_deformable_groups)
        self.norm_1 = norm_layer(input_nc)
        '''


        #8-->64, 4-->32, output seq_len, batchsize, ngf*8, H, W
        offset_nc = self.groups*2

        self.offset_nc = offset_nc
        self.shape = shape
        #print 'shape: ',shape,' ,input_nc: ', input_nc, ' ,ngf: ', ngf*mult
        self.embedding = nn.Sequential(nn.Conv2d(offset_nc,ngf, kernel_size = 1, padding=0),norm_layer(ngf),nn.LeakyReLU(0.2,True))
        self.conv_lstm = CLSTM(shape, input_nc+ngf, filter_size, ngf, nlayers,use_bias)
        model = [nn.Conv2d(ngf, offset_nc, kernel_size=1,padding = 0)]
        self.model = nn.Sequential(*model)
        #self.enhance = nn.Sequential(nn.Conv2d(input_nc, input_nc, kernel_size=3,stride=1,padding = 1,bias=False),norm_layer(input_nc))
        #self.norml = norm_layer(input_nc)
    def forward(self,input,pre_offset,hidden_state):
        #input = Variable(input.data)
        embedded = self.embedding(pre_offset)
        convlstm_out = self.conv_lstm(torch.cat([input.unsqueeze(1),embedded.unsqueeze(1)],2), hidden_state)
        offset = self.model(convlstm_out[1][-1])
        #print offset.size()
        #offset = convlstm_out[1][-1]

        if self.input_nc == self.output_nc:
            offset_r = offset.repeat(1,self.input_nc/self.groups,1,1)
            #output = self.relu(self.norml(self._transform(self, input, offset_r)))
            output = self._transform(self, input, offset_r)
        else:
            assert(self.input_nc/self.output_nc == self.groups)
            #inputs = torch.split(input,self.output_nc,dim = 1) #output_channels*groups
            offs = torch.split(offset,2,dim = 1) # 2*groups
            offsets = [off.repeat(1,self.output_nc,1,1) for off in offs]
            offset_r = torch.cat(offsets,1)
            output_ = self._transform(self, input, offset_r)
            outputs = torch.split(output_,self.output_nc,1)
            output = Variable(torch.zeros(outputs[0].size()).cuda())
            for out in outputs:
                output += out
            output = output/self.groups
                # shared offset with groups*2 channels, target has nc*2 channels, repeat n=(nc/groups) times (nc/groups)
                #output = self.relu(self.conv_offset_1(input,offset))
        return convlstm_out[0], offset, output #[bs,1, oc, s0, s1]

    def init_hidden(self,batch_size):
        return self.conv_lstm.init_hidden(batch_size)

    def init_offset(self,batch_size):
        return Variable(torch.zeros(batch_size,self.offset_nc,self.shape[0],self.shape[1])).cuda()

    @staticmethod
    def _transform(self, x, offsets):
        x_shape = x.size()
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        x = self._to_bc_h_w(x, x_shape)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        return x_offset


    @staticmethod
    def _get_grid(self, x):
        batch_size, input_size = x.size(0), (x.size(1), x.size(2))
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_size, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_size, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_size, dtype, cuda)
        return self._grid

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x


#Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(in_dim, out_dim, padding_type,norm_layer,use_dropout,use_bias)
        self.downsample = None
        if in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_dim),
            )
        self.relu = nn.LeakyReLU(0.2,True)

    def build_conv_block(self,in_dim,out_dim,padding_type,norm_layer,use_dropout,use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=p,bias=use_bias),norm_layer(out_dim),nn.LeakyReLU(0.2,True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(out_dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(residual + self.conv_block(x))
        return out

#Define a resnet bottleneck
class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, padding_type, norm_layer, use_dropout, use_bias):
        super(Bottleneck, self).__init__()
        self.conv_block = self.build_conv_block(in_dim, out_dim, padding_type,norm_layer,use_dropout,use_bias)
        self.downsample = None
        if in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_dim),
            )
        self.relu = nn.ReLU(True)

    def build_conv_block(self,in_dim, out_dim, padding_type,norm_layer,use_dropout,use_bias):
        conv_block = [nn.Conv2d(in_dim,out_dim/4,kernel_size=1,bias=use_bias),norm_layer(out_dim/4),nn.ReLU(True)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(out_dim/4,out_dim/4,kernel_size=3,padding=p,bias=use_bias),norm_layer(out_dim/4),nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.Conv2d(out_dim/4,out_dim,kernel_size=1,bias=use_bias),norm_layer(out_dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(residual + self.conv_block(x))
        return out

# Unet Encoder
class UnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False,gpu_ids=[]):
        super(UnetEncoder,self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        assert(num_downs > 2)
        self.num_downs = num_downs
        self.gpu_ids = gpu_ids
        #################################################### Encoder ##################################################################
        self.conv_1 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf), nn.LeakyReLU(0.2,True)) #2 128-->64, 256-->128
        self.conv_2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*2),nn.LeakyReLU(0.2,True))  #4 64-->32, 128-->64
        self.conv_3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*4),nn.LeakyReLU(0.2,True)) #8 32-->16, 64-->32
        self.conv_4 = nn.Sequential(nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*8),nn.LeakyReLU(0.2,True)) #16 16-->8, 32-->16
        self.conv_5 = nn.Sequential(nn.Conv2d(ngf*8, output_nc, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(output_nc),nn.LeakyReLU(0.2,True))  #32 8-->4, 16-->8


    def forward(self, input):
        encs = []
        out = input
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            out = nn.parallel.data_parallel(self.conv_1, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.conv_2, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.conv_3, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.conv_4, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.conv_5, out, self.gpu_ids)
            encs += [out]
        else:
            out = self.conv_1(out)
            encs += [out]
            out = self.conv_2(out)
            encs += [out]
            out = self.conv_3(out)
            encs += [out]
            out = self.conv_4(out)
            encs += [out]
            out = self.conv_5(out)
            encs += [out]
        output = encs
        return output

class UnetDecoder(nn.Module):
    def __init__(self, input_nc, latent_nc, output_nc, num_downs, ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False,gpu_ids=[]):
        super(UnetDecoder,self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        assert(num_downs > 2)
        self.num_downs = num_downs
        self.gpu_ids = gpu_ids

        self.deconv_4 = nn.Sequential(nn.ConvTranspose2d(input_nc+latent_nc, ngf*4, kernel_size=4, stride=2, padding=1), norm_layer(ngf*4), nn.LeakyReLU(0.2,True))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*2, kernel_size=4, stride=2, padding=1), norm_layer(ngf*2), nn.LeakyReLU(0.2,True))
        deconv_2 = [nn.ConvTranspose2d(ngf*4, ngf, kernel_size=4, stride=2,padding=1), norm_layer(ngf), nn.LeakyReLU(0.2,True)]
        self.deconv_2 = nn.Sequential(*deconv_2)
        deconv_1 = [nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1), nn.Tanh()]
        self.deconv_1 = nn.Sequential(*deconv_1)
    def forward(self, inputs,latent):
        if len(self.gpu_ids) > 1 and isinstance(inputs[0].data, torch.cuda.FloatTensor):
            out = nn.parallel.data_parallel(self.deconv_4, torch.cat([inputs[-1],latent],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_3, torch.cat([inputs[-2],out],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_2, torch.cat([inputs[-3],out],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_1, torch.cat([inputs[-4],out],1),self.gpu_ids)
        else:
            out = self.deconv_4(torch.cat([inputs[-1],latent],1))
            out = self.deconv_3(torch.cat([inputs[-2],out],1))
            out = self.deconv_2(torch.cat([inputs[-3],out],1))
            out = self.deconv_1(torch.cat([inputs[-4],out],1))

        return out

# Unet Generatpr
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, content_nc,latent_nc, num_downs, ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False,gpu_ids=[]):
        super(UnetGenerator,self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        assert(num_downs > 2)
        self.num_downs = num_downs
        self.gpu_ids = gpu_ids

        #################################################### Encoder ##################################################################
        self.conv_1 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf),nn.LeakyReLU(0.2,True)) #2 128-->64, 256-->128
        self.conv_2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*2),nn.LeakyReLU(0.2,True))  #4 64-->32, 128-->64
        self.conv_3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*4),nn.LeakyReLU(0.2,True)) #8 32-->16, 64-->32
        self.conv_4 = nn.Sequential(nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*8),nn.LeakyReLU(0.2,True)) #16 16-->8, 32-->16
        self.conv_5 = nn.Sequential(nn.Conv2d(ngf*8, content_nc, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(content_nc),nn.LeakyReLU(0.2,True))  #32 8-->4, 16-->8
        '''if num_downs == 7: #128
            self.conv_6 = nn.Sequential(nn.Conv2d(ngf*8, content_nc, kernel_size=4, bias=use_bias),norm_layer(content_nc),nn.Tanh()) #128 4-->1
        elif num_downs == 8: #256
            self.conv_6 = nn.Sequential(nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*8),nn.LeakyReLU(0.2,True)) #256 4-->1
            self.conv_7 = nn.Sequential(nn.Conv2d(ngf*8, content_nc, kernel_size=4, bias=use_bias),norm_layer(content_nc),nn.Tanh()) #256 4-->1
        else:
            raise NotImplementedError('Generator Unet encoder model number of layers [%d] is not supported' % num_downs)
        '''

        ################################################### Decoder ###################################################################
        '''if num_downs == 7: #128
            self.deconv_6 = nn.Sequential(nn.ConvTranspose2d(content_nc+latent_nc, ngf*8, kernel_size=4, bias=use_bias),norm_layer(ngf*8),nn.LeakyReLU(0.2,True)) #128 1-->4
        elif num_downs == 8: #256
            self.deconv_7 = nn.Sequential(nn.ConvTranspose2d(content_nc+latent_nc, ngf*8, kernel_size=4, bias=use_bias),norm_layer(ngf*8),nn.LeakyReLU(0.2,True)) #256 1-->4
            self.deconv_6 = nn.Sequential(nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=4,stride=2,padding=1,bias=use_bias),norm_layer(ngf*8),nn.LeakyReLU(0.2,True)) #256 4-->8
        else:
            raise NotImplementedError('Generator Unet decoder model number of layers [%d] is not supported' % num_downs)
        '''
        self.deconv_5 = nn.Sequential(nn.ConvTranspose2d(content_nc, ngf*8, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*8),nn.LeakyReLU(0.2,True)) #2 4-->8, 8-->16
        self.deconv_4 = nn.Sequential(nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*4),nn.LeakyReLU(0.2,True))  #4 8-->16, 16-->32
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(ngf*4*2, ngf*2, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf*2),nn.LeakyReLU(0.2,True)) #8 16-->32, 32-->64
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(ngf*2*2, ngf, kernel_size=4, stride=2,padding=1,bias=use_bias),norm_layer(ngf),nn.LeakyReLU(0.2,True)) #16 32-->64, 64-->128
        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2,padding=1,bias=use_bias),nn.Tanh())  #32 64-->128, 128-->256
        #self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2,padding=1,bias=use_bias),nn.Tanh())  #32 64-->128, 128-->256
        ################################################## Fusion layers ###########################################################
        fusion_conv_1 = [nn.LeakyReLU(0.2,True),nn.Conv2d(input_nc*2, ngf*2, kernel_size=3, stride=1,padding=1,bias=use_bias),norm_layer(ngf*2),nn.LeakyReLU(0.2,True)]
        fusion_conv_1 += [nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1,padding=1,bias=use_bias),norm_layer(ngf),nn.LeakyReLU(0.2,True)]
        fusion_conv_1 += [nn.Conv2d(ngf,output_nc, kernel_size=3, stride=1,padding=1,bias=use_bias),nn.Tanh()]
        self.fusion = nn.Sequential(*fusion_conv_1)
        self.tanh = nn.Tanh()
    def forward(self, input, latent, layer_idx, fusion):
        encs = []
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            ####################### Encoder forward #################################
            enc_1 = nn.parallel.data_parallel(self.conv_1, input, self.gpu_ids)
            encs += [encs_1]
            enc_2 = nn.parallel.data_parallel(self.conv_2, enc_1, self.gpu_ids)
            encs += [enc_2]
            enc_3 = nn.parallel.data_parallel(self.conv_3, enc_2, self.gpu_ids)
            encs += [enc_3]
            enc_4 = nn.parallel.data_parallel(self.conv_4, enc_3, self.gpu_ids)
            encs += [enc_4]
            enc_5 = nn.parallel.data_parallel(self.conv_5, enc_4, self.gpu_ids)
            encs += [enc_5]
            '''enc_6 = nn.parallel.data_parallel(self.conv_6, enc_5, self.gpu_ids)
            encs += [enc_6]
            if self.num_downs == 8:
                enc_7 = nn.parallel.data_parallel(self.conv_7, enc_6, self.gpu_ids)
                encs += [enc_7]
            '''
            ####################### DCN forward #################################

            '''if len(latent) == 2:
                enc_2 = latent[0]
            elif len(latent) == 3:
                enc_2 = latent[0]
                enc_4 = latent[1]
            '''
            for i in xrange(len(layer_idx)):
                encs[layer_idx[i]] = latent[i]
            ####################### Decoder forward #################################
            '''if self.num_downs == 8:
                dec_7 = nn.parallel.data_parallel(self.deconv_7, torch.cat([enc_7,latent[-1]],1), self.gpu_ids)
                dec_6 = nn.parallel.data_parallel(self.deconv_6, torch.cat([dec_7,enc_6],1), self.gpu_ids)
            elif self.num_downs==7:
                dec_6 = nn.parallel.data_parallel(self.deconv_6, torch.cat([enc_6,latent[-1]],1), self.gpu_ids)
            '''
            dec_5 = nn.parallel.data_parallel(self.deconv_5, encs[4], self.gpu_ids)
            dec_4 = nn.parallel.data_parallel(self.deconv_4, torch.cat([dec_5,encs[3]],1), self.gpu_ids)
            dec_3 = nn.parallel.data_parallel(self.deconv_3, torch.cat([dec_4,encs[2]],1), self.gpu_ids)
            dec_2 = nn.parallel.data_parallel(self.deconv_2, torch.cat([dec_3,encs[1]],1), self.gpu_ids)
            dec_1 = nn.parallel.data_parallel(self.deconv_1, torch.cat([dec_2,encs[0]],1), self.gpu_ids)

        else:
            ####################### Encoder forward #################################
            enc_1 = self.conv_1(input)
            encs += [enc_1]
            enc_2 = self.conv_2(enc_1)
            encs += [enc_2]
            enc_3 = self.conv_3(enc_2)
            encs += [enc_3]
            enc_4 = self.conv_4(enc_3)
            encs += [enc_4]
            enc_5 = self.conv_5(enc_4)
            encs += [enc_5]
            '''enc_6 = self.conv_6(enc_5)
            encs += [enc_6]
            if self.num_downs == 8:
                enc_7 = self.conv_7(enc_6)
                encs += [enc_7]
            '''
            ####################### DCN forward #################################
            '''
            if len(latent) == 2:
                enc_2 = latent[0]
            elif len(latent) == 3:
                #print 'changing'
                enc_2 = latent[0]
                enc_4 = latent[1]
            '''
            for i in xrange(len(layer_idx)):
                encs[layer_idx[i]] = latent[i]
            ####################### Decoder forward #################################
            '''if self.num_downs == 8:
                dec_7 = self.deconv_7(torch.cat([enc_7,latent[-1]],1))
                dec_6 = self.deconv_6(torch.cat([dec_7,enc_6],1))
            else:
                dec_6 = self.deconv_6(torch.cat([enc_6,latent[-1]],1))
            '''
            dec_5 = self.deconv_5(encs[4])
            dec_4 = self.deconv_4(torch.cat([dec_5,encs[3]],1))
            dec_3 = self.deconv_3(torch.cat([dec_4,encs[2]],1))
            dec_2 = self.deconv_2(torch.cat([dec_3,encs[1]],1))
            dec_1 = self.deconv_1(torch.cat([dec_2,encs[0]],1))
            #dec_1 = self.deconv_1(dec_2)


            if len(fusion) > 0:
                out = self.fusion(torch.cat([fusion[0],dec_1],1))
            else:
                out = dec_1
        return encs, out



# Unet Encoder Decoder 3D
class UnetEncoder3D_128(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=32,norm_layer=nn.BatchNorm3d,use_dropout=False,gpu_ids=[]):
        super(UnetEncoder3D_128,self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        assert(num_downs > 2)
        self.num_downs = num_downs
        self.gpu_ids = gpu_ids
        self.leakyrelu = nn.LeakyReLU(0.2,True)
        self.relu = nn.ReLU(True)
        self.conv_1_1 = nn.Conv3d(input_nc, ngf, kernel_size=(4,4,4), stride=(2,2,2),padding=1,bias=use_bias)
        self.norm_1 = norm_layer(ngf) #32
        self.conv_2_1 = nn.Conv3d(ngf, ngf*2, kernel_size=(2,4,4), stride=(1,2,2),padding=(0,1,1),bias=use_bias)
        #self.conv_2_1_off = nn.Conv3d(ngf, 1*3*2*4*4, kernel_size=(2,4,4), stride=(1,2,2),padding=(0,1,1),bias=use_bias)
        #self.conv_2_1 = ConvOffset3d(ngf,ngf*2,kernel_size=(2,4,4),stride=(1,2,2),padding=(0,1,1), channel_per_group=ngf)
        self.norm_2 = norm_layer(ngf*2) #64
        self.conv_3_1 = nn.Conv3d(ngf*2, ngf*4, kernel_size=(2,4,4), stride=(1,2,2),padding=(0,1,1),bias=use_bias)
        #self.conv_3_1_off = nn.Conv3d(ngf*2, 1*3*2*4*4, kernel_size=(2,4,4), stride=(1,2,2),padding=(0,1,1),bias=use_bias)
        #self.conv_3_1 = ConvOffset3d(ngf*2,ngf*4,kernel_size=(2,4,4),stride=(1,2,2),padding=(0,1,1), channel_per_group=ngf*2)
        self.norm_3 = norm_layer(ngf*4) # 128
        self.conv_4_1 = nn.Conv3d(ngf*4, ngf*8, kernel_size=(1,4,4), stride=(1,2,2),padding=(0,1,1),bias=use_bias)
        #self.conv_4_1_off = nn.Conv3d(ngf*4, 1*3*1*4*4, kernel_size=(1,4,4), stride=(1,2,2),padding=(0,1,1),bias=use_bias)
        #self.conv_4_1 = ConvOffset3d(ngf*4,ngf*8,kernel_size=(1,4,4),stride=(1,2,2),padding=(0,1,1), channel_per_group=ngf*4)
        self.norm_4 = norm_layer(ngf*8) # 256

        #self.conv_5_1 = nn.Conv3d(ngf*8, ngf*16, kernel_size=(1,4,4), stride=(1,2,2),padding=(0,1,1),bias=use_bias)
        self.conv_5_1_off = nn.Conv3d(ngf*8, 8//2*3*1*4*4, kernel_size=(1,4,4), stride=(1,2,2),padding=(0,1,1),bias=use_bias)
        self.conv_5_1 = ConvOffset3d(ngf*8,ngf*16,kernel_size=(1,4,4),stride=(1,2,2),padding=(0,1,1), channel_per_group=ngf*2)
        self.norm_5 = norm_layer(ngf*16) # 512
        self.conv_6_1 = nn.Conv3d(ngf*16, output_nc, kernel_size=(1,4,4), stride=(1,1,1),padding=0,bias=use_bias)
        self.norm_6 = norm_layer(output_nc) # output_nc

    def forward(self, input):
        encs = []
        out = input
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            out = nn.parallel.data_parallel(self.conv_1_1, out, self.gpu_ids)
            #out = nn.parallel.data_parallel(self.norm_1, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.conv_2_1, out, self.gpu_ids)
            #off = nn.parallel.data_parallel(self.conv_2_1_off, out, self.gpu_ids)
            #out = nn.parallel.data_parallel(self.conv_2_1, out,off, self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_2, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.conv_3_1, out, self.gpu_ids)
            #off = nn.parallel.data_parallel(self.conv_3_1_off, out, self.gpu_ids)
            #out = nn.parallel.data_parallel(self.conv_3_1, out,off, self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_3, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.conv_4_1, out, self.gpu_ids)
            #off = nn.parallel.data_parallel(self.conv_4_1_off, out, self.gpu_ids)
            #out = nn.parallel.data_parallel(self.conv_4_1, out,off, self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_4, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
            #out = nn.parallel.data_parallel(self.conv_5_1, out, self.gpu_ids)
            off = nn.parallel.data_parallel(self.conv_5_1_off, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.conv_5_1, out,off, self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_5, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.conv_6_1, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_6, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
        else:
            out = self.conv_1_1(out)
            #out = self.norm_1(out)
            out = self.leakyrelu(out)
            encs += [out]
            #off = self.conv_2_1_off(out)
            #out = self.conv_2_1(out,off)
            out = self.conv_2_1(out)
            out = self.norm_2(out)
            out = self.leakyrelu(out)
            encs += [out]
            #off = self.conv_3_1_off(out)
            #out = self.conv_3_1(out,off)
            out = self.conv_3_1(out)
            out = self.norm_3(out)
            out = self.leakyrelu(out)
            encs += [out]
            #off = self.conv_4_1_off(out)
            #out = self.conv_4_1(out,off)
            out = self.conv_4_1(out)
            out = self.norm_4(out)
            out = self.leakyrelu(out)
            encs += [out]
            off = self.conv_5_1_off(out)
            out = self.conv_5_1(out,off)
            #out = self.conv_5_1(out)
            out = self.norm_5(out)
            out = self.leakyrelu(out)
            encs += [out]
            out = self.conv_6_1(out)
            out = self.norm_6(out)
            out = self.leakyrelu(out)
            encs += [out]
        return encs

class UnetDecoder3D_128(nn.Module):
    def __init__(self, input_nc, latent_nc, output_nc, num_downs, ngf=64,norm_layer=nn.BatchNorm3d,use_dropout=False,gpu_ids=[]):
        super(UnetDecoder3D_128,self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        assert(num_downs > 2)
        self.num_downs = num_downs
        self.latent_nc = latent_nc
        self.gpu_ids = gpu_ids
        self.relu = nn.LeakyReLU(0.2,True)#nn.ReLU(True)
        self.tanh = nn.Tanh()
        #mult = 2**(num_downs - 2)
        #self.conv = nn.Conv2d(latent_nc, input_nc, kernel_size=3, stride=1,padding=1)
        self.deconv_6_1 = nn.ConvTranspose3d(input_nc + latent_nc, ngf*16 , kernel_size=(1,4,4), stride=1,padding=0)
        self.norm_6 = norm_layer(ngf*16) # 512
        self.deconv_5_1 = nn.ConvTranspose3d(ngf*32, ngf*8 , kernel_size=(1,4,4), stride=(1,2,2),padding=(0,1,1))
        self.norm_5 = norm_layer(ngf*8) # 256
        self.deconv_4_1 = nn.ConvTranspose3d(ngf*16, ngf*4 , kernel_size=(1,4,4), stride=(1,2,2),padding=(0,1,1))
        self.norm_4 = norm_layer(ngf*4) # 128
        self.deconv_3_1 = nn.ConvTranspose3d(ngf*8, ngf*2 , kernel_size=(2,4,4), stride=(1,2,2),padding=(0,1,1))
        self.norm_3 = norm_layer(ngf*2) # 64
        self.deconv_2_1 = nn.ConvTranspose3d(ngf*4, ngf , kernel_size=(2,4,4), stride=(1,2,2),padding=(0,1,1))
        self.norm_2 = norm_layer(ngf) # 32
        self.deconv_1_1 = nn.ConvTranspose3d(ngf*2, output_nc , kernel_size=(4,4,4), stride=(2,2,2),padding=1)
        self.norm_1 = norm_layer(output_nc) # output_nc
        #self.conv_0_1 = nn.Conv3d(ngf, output_nc, kernel_size = 3, stride = 1, padding = 1)
    def forward(self, inputs,latent):
        if len(self.gpu_ids) > 1 and isinstance(inputs[0].data, torch.cuda.FloatTensor):
            if self.latent_nc == 0:
                assert(latent == None)
                out = nn.parallel.data_parallel(self.deconv_6_1, inputs[-1],self.gpu_ids)
            else:
                out = nn.parallel.data_parallel(self.deconv_6_1, torch.cat([inputs[-1],latent],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_6, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.relu, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_5_1, torch.cat([inputs[-2],out],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_5, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.relu, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_4_1, torch.cat([inputs[-3],out],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_4, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.relu, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_3_1, torch.cat([inputs[-4],out],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_3, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.relu, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_2_1, torch.cat([inputs[-5],out],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_2, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.relu, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_1_1, torch.cat([inputs[-6],out],1),self.gpu_ids)
            #out = nn.parallel.data_parallel(self.norm_1, out,self.gpu_ids)
            #out = nn.parallel.data_parallel(self.relu, out,self.gpu_ids)
            #out = nn.parallel.data_parallel(self.conv_0_1, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.tanh, out,self.gpu_ids)
        else:
            if self.latent_nc == 0:
                assert(latent == None)
                out = self.deconv_6_1(inputs[-1])
            else:
                out = self.deconv_6_1(torch.cat([inputs[-1],latent],1))
            out = self.norm_6(out)
            out = self.relu(out)
            out = self.deconv_5_1(torch.cat([inputs[-2],out],1))
            out = self.norm_5(out)
            out = self.relu(out)
            out = self.deconv_4_1(torch.cat([inputs[-3],out],1))
            out = self.norm_4(out)
            out = self.relu(out)
            out = self.deconv_3_1(torch.cat([inputs[-4],out],1))
            out = self.norm_3(out)
            out = self.relu(out)
            out = self.deconv_2_1(torch.cat([inputs[-5],out],1))
            out = self.norm_2(out)
            out = self.relu(out)
            out = self.deconv_1_1(torch.cat([inputs[-6],out],1))
            #out = self.norm_1(out)
            #out = self.relu(out)
            #out = self.conv_0_1(out)
            out = self.tanh(out)
        return out

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator3D(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator3D, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock3D(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock3D, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, isize, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=2, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        output = output.view(output.size(0),-1)
        return output

class DCGAN_D(nn.Module):
    def __init__(self, isize, input_nc, ndf, n_layers=0, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(DCGAN_D, self).__init__()
        self.gpu_ids = gpu_ids
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(input_nc, ndf),
                        nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))

        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2


        # state size. K x 4 x 4
        #cndf = cndf * 2
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=True))
        main.add_module('final.{0}.sigmoid'.format(input_nc),
                        nn.Sigmoid())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 0:
            f_out = nn.parallel.data_parallel(self.main, input, self.gpu_ids)
        else:
            f_out = self.main(input)
        output = f_out.view(-1,1)
        return output


## Discriminator with multi inputs
class Multi_D(nn.Module):
    def __init__(self, isize, input_nc, feat_nc, ndf, n_layers=0, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(Multi_D, self).__init__()
        self.gpu_ids = gpu_ids
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        fsize = isize / 16 # 128 --> 8


        embedded = nn.Sequential()
        main = nn.Sequential()
        # input is nc x isize x isize
        embedded.add_module('initial.conv.{0}-{1}'.format(input_nc, ndf),
                        nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False))
        embedded.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))

        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_layers):
            embedded.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            embedded.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            embedded.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > fsize:
            in_feat = cndf
            out_feat = cndf * 2
            embedded.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            embedded.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            embedded.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        cndf += feat_nc
        while csize > 4:
            in_feat = cndf
            out_feat = cndf
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf
            csize = csize / 2

        # state size. K x 4 x 4
        #cndf = cndf * 2
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=True))
        main.add_module('final.{0}.sigmoid'.format(input_nc),
                        nn.Sigmoid())

        self.embedded = embedded
        self.main = main

    def forward(self, input, feature):
        if isinstance(input.data, torch.cuda.FloatTensor) and len(self.gpu_ids) > 0:
            embedded = nn.parallel.data_parallel(self.embedded,input,self.gpu_ids)
            output = nn.parallel.data_parallel(self.main, torch.cat([embedded,feature],1), self.gpu_ids)
        else:
            embedded = self.embedded(input)
            output = self.main(torch.cat([embedded,feature],1))
        output = output.view(-1,1)
        return output
