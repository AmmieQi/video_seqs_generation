import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np

from ConvLSTM import *

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
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
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type = 'instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
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
def content_E(input_nc, output_nc, ngf, which_model_E, norm='batch', use_dropout=False,gpu_ids=[],init_type='normal'):
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
        netCE = UnetEncoder(input_nc, output_nc, 3, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_E == 'unet_128':
        netCE = UnetEncoder(input_nc, output_nc, 4, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
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
        netME = MotionEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer,use_dropout=use_dropout,n_blocks=6,gpu_ids=gpu_ids)
    elif which_model_E == 'resnet_3blocks':
        netME = MotionEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer,use_dropout=use_dropout,n_blocks=3,gpu_ids=gpu_ids)
    elif which_model_E == 'resnet_1blocks':
        netME = MotionEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer,use_dropout=use_dropout,n_blocks=1,gpu_ids=gpu_ids)
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
        netG = UnetDecoder(input_nc, latent_nc, output_nc, 3, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetDecoder(input_nc, latent_nc,output_nc, 4, ngf, norm_layer=norm_layer,use_dropout=use_dropout, gpu_ids=gpu_ids)
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
    def __init__(self,input_nc,output_nc,ngf=64,norm_layer=nn.BatchNorm2d,use_dropout=False,n_blocks=3,gpu_ids=[],padding_type='reflect'):
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

        model = [nn.ReflectionPad2d(2), nn.Conv2d(input_nc, ngf, kernel_size=5,padding=0,bias=use_bias),norm_layer(ngf),nn.LeakyReLU(0.2,True)]
        n_downsampling = 4
        # 64 --> 8, 32-->4
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2,padding=1,dilation=1,bias=use_bias),
                      norm_layer(ngf * mult *2),
                      nn.LeakyReLU(0.2,True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf*mult,ngf*mult,padding_type=padding_type,norm_layer=norm_layer,use_dropout=use_dropout,use_bias=use_bias)]
            #for j in range(n_blocks):
            #model += [Bottleneck(ngf*mult, ngf*mult,padding_type=padding_type,norm_layer=norm_layer,use_dropout=use_dropout,use_bias=use_bias)]
            #model += [Bottleneck(ngf*mult, ngf*mult*2,padding_type=padding_type,norm_layer=norm_layer,use_dropout=use_dropout,use_bias=use_bias)]
            #model += [nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)]

        #mult = 2**n_downsampling
        model += [nn.ReflectionPad2d(2)]
        model += [nn.Conv2d(ngf*mult,output_nc,kernel_size=5,padding=0)]
        model += [nn.LeakyReLU(0.2,True)]

        self.model = nn.Sequential(*model)

    # output size osize = (isize/2**n_downsampling)
    # output batchSize*output_nc*osize*osize
    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data,torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        output = output.unsqueeze(1) # [bs, oc, s0, s1] -> [bs,1,oc,s0,s1]
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
            #convlstm_out = nn.parallel.data_parallel(self.conv_lstm, input, hidden_state, self.gpu_ids)
            output = nn.parallel.data_parallel(self.model, convlstm_out[1][-1], self.gpu_ids)
        else:
            #convlstm_out = self.conv_lstm(input, hidden_state)
            output = self.model(convlstm_out[1][-1])
        #print 'conv_lstm output shape: ', convlstm_out[1].size()
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
        nlayers=1
        #If using this format, then we need to transpose in CLSTM
        #input = Variable(torch.rand(batch_size,seq_len,inp_chans,shape[0],shape[1])).cuda()
        n_upsampling = 2
        mult = 2**n_upsampling
        #print 'shape: ',shape,' ,input_nc: ', input_nc, ' ,ngf: ', ngf*mult
        self.conv_lstm = ResCLSTM(shape, input_nc, filter_size, ngf*mult,nlayers,use_bias)

        #8-->64, 4-->32, output seq_len, batchsize, ngf*8, H, W
        model = []
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf*4, output_nc, kernel_size=3,padding = 0)]
        #model += [nn.Tanh()]

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

# Unet Encoder Decoder
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
        self.leakyrelu = nn.LeakyReLU(0.2,True)
        self.relu = nn.ReLU(True)
        self.conv_1_1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1,bias=use_bias)
        self.norm_1 = norm_layer(ngf)
        self.conv_2_1 = nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1,bias=use_bias)
        self.norm_2 = norm_layer(ngf*2)
        self.conv_3_1 = nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1,bias=use_bias)
        self.norm_3 = norm_layer(ngf*4)
        self.conv_4_1 = nn.Conv2d(ngf*4, output_nc, kernel_size=4, stride=2,padding=1,bias=use_bias)

    def forward(self, input):
        encs = []
        out = input
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            out = nn.parallel.data_parallel(self.con_1_1, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_1, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.con_2_1, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_2, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.con_3_1, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_3, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
            out = nn.parallel.data_parallel(self.conv_4_1, out, self.gpu_ids)
            out = nn.parallel.data_parallel(self.leakyrelu, out, self.gpu_ids)
            encs += [out]
        else:

            out = self.conv_1_1(out)
            out = self.norm_1(out)
            out = self.leakyrelu(out)
            encs += [out]
            out = self.conv_2_1(out)
            out = self.norm_2(out)
            out = self.leakyrelu(out)
            encs += [out]
            out = self.conv_3_1(out)
            out = self.norm_3(out)
            out = self.leakyrelu(out)
            encs += [out]
            out = self.conv_4_1(out)
            out = self.leakyrelu(out)
            encs += [out]
        return encs

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
        self.relu = nn.LeakyReLU(0.2,True)#nn.ReLU(True)
        self.tanh = nn.Tanh()
        mult = 2**(num_downs - 2)
        #self.conv = nn.Conv2d(latent_nc, input_nc, kernel_size=3, stride=1,padding=1)

        self.deconv_4_1 = nn.ConvTranspose2d(input_nc + latent_nc, ngf*4 , kernel_size=4, stride=2,padding=1)
        self.norm_4 = norm_layer(ngf*4)
        self.deconv_3_1 = nn.ConvTranspose2d(ngf*8, ngf*2 , kernel_size=4, stride=2,padding=1)
        self.norm_3 = norm_layer(ngf*2)
        self.deconv_2_1 = nn.ConvTranspose2d(ngf*4, ngf , kernel_size=4, stride=2,padding=1)
        self.norm_2 = norm_layer(ngf)
        self.deconv_1_1 = nn.ConvTranspose2d(ngf*2, output_nc , kernel_size=4, stride=2,padding=1)
        #self.norm_4 = norm_layer(ngf*4)
    def forward(self, inputs,latent):
        if len(self.gpu_ids) > 1 and isinstance(inputs[0].data, torch.cuda.FloatTensor):
            #lout = nn.parallel.data_parallel(self.conv, latent, self.gpu_ids)
            #lout = nn.parallel.data_parallel(self.relu, lout,self.gpu_ids)
            #input = inputs[-1]+lout
            out = nn.parallel.data_parallel(self.deconv_4_1, torch.cat([inputs[-1],latent],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_4, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.relu, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_3_1, torch.cat([inputs[-2],out],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_3, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.relu, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_2_1, torch.cat([inputs[-3],out],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.norm_2, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.relu, out,self.gpu_ids)
            out = nn.parallel.data_parallel(self.deconv_1_1, torch.cat([inputs[-4],out],1),self.gpu_ids)
            out = nn.parallel.data_parallel(self.tanh, out,self.gpu_ids)
        else:
            #lout = self.relu(self.conv(latent))
            #input = inputs[-1]+lout
            out = self.deconv_4_1(torch.cat([inputs[-1],latent],1))
            out = self.norm_4(out)
            out = self.relu(out)
            out = self.deconv_3_1(torch.cat([inputs[-2],out],1))
            out = self.norm_3(out)
            out = self.relu(out)
            out = self.deconv_2_1(torch.cat([inputs[-3],out],1))
            out = self.norm_2(out)
            out = self.relu(out)
            out = self.deconv_1_1(torch.cat([inputs[-4],out],1))
            out = self.tanh(out)
        return out

'''
# Defines the Unet Encoder.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        #assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:

            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x, l):
        if self.outermost:
            return self.model(x)
        elif self.innermost:
            #innderEnc = x
            #latentEnc = l
            enc = torch.cat([x,l],1)
            return torch.cat([self.model(enc), x], 1)
        else:
            return torch.cat([self.model(x), x], 1)
'''

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
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw-1, stride=1, padding=padw)]

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
        fsize = isize / 8 # 128 --> 16


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
