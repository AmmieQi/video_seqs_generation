import numpy as np
import random
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class MultiModel(BaseModel):
    def name(self):
        return 'MultiModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        seq_len = opt.seq_len
        self.seq_len = opt.seq_len
        self.pre_len = opt.pre_len
        low_shape = (opt.fineSize/32, opt.fineSize/32)
        high_shape = (opt.fineSize/4, opt.fineSize/4)
        #self.input_seq = self.Tensor(nb, seq_len, opt.input_nc, size, size)
        self.use_cycle = opt.use_cycle
        self.input_nc = opt.input_nc
        # Content_Encoder, Motion_Encoder, Motion_Predictor, Overall_Generator, Overall_D
        # opt.input_nc = 3
        # opt.output_nc = 3
        #self.input_nc = opt.input_nc * opt.seq_len

        self.content_nc = 8
        self.netCE = networks.content_E(opt.input_nc*self.seq_len, opt.output_nc, self.content_nc, opt.latent_nc,
                                                  opt.ngf, opt.which_model_E, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
        self.netOPL = networks.offsets_P(low_shape, seq_len, self.content_nc, self.content_nc, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type,relu='leakyrelu',groups = 1)
        self.netOPH = networks.offsets_P(high_shape, seq_len, opt.ngf*2, opt.ngf*2, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type,relu='leakyrelu',groups=1)

        self.netME = networks.content_E(opt.input_nc*self.seq_len,self.content_nc, opt.output_nc,opt.latent_nc,
                                                opt.ngf, 'unet_128', opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
        self.netMP = networks.motion_P(low_shape, seq_len, opt.latent_nc,
                                            opt.latent_nc, opt.ngf, opt.norm, not opt.no_dropout, self.gpu_ids, opt.init_type)
        self.netG = networks.define_G(low_shape, seq_len, 32,opt.latent_nc,
                                            opt.output_nc, opt.ngf,opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,opt.init_type)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.fineSize, opt.output_nc, 256,opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids,opt.init_type)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netCE, 'CE', which_epoch)
            self.load_network(self.netOPH, 'OPH', which_epoch)
            self.load_network(self.netOPL, 'OPL', which_epoch)
            self.load_network(self.netME, 'ME', which_epoch)
            self.load_network(self.netMP, 'MP', which_epoch)
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionPixel = torch.nn.MSELoss()
            self.criterionPre   = torch.nn.L1Loss()
            self.criterionFeat = torch.nn.MSELoss()
            self.criteriondiff = torch.nn.L1Loss()
            self.criterionSim = torch.nn.MSELoss()
            self.criterionFlow = torch.nn.MSELoss()
            #self.criterionTrip = torch.nn.TripletMarginLoss(margin=1, p=2)
            self.criterionTrip = networks.TripLoss(p=2)
            self.criterionCoh = networks.GDLLoss(2, tensor=self.Tensor)
            self.criterionCoh_h = networks.GDLLoss(2, tensor=self.Tensor)
            self.criterionGDL = networks.GDLLoss(opt.input_nc, tensor=self.Tensor)
            # initialize optimizers
            self.optimizer_CE = torch.optim.Adam(self.netCE.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_OPH = torch.optim.Adam(self.netOPH.parameters(), lr = opt.lr/20, betas = (opt.beta1, 0.999))
            self.optimizer_OPL = torch.optim.Adam(self.netOPL.parameters(), lr = opt.lr/20, betas = (opt.beta1, 0.999))
            self.optimizer_ME = torch.optim.Adam(self.netME.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_MP = torch.optim.Adam(self.netMP.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_CE)
            self.optimizers.append(self.optimizer_OPH)
            self.optimizers.append(self.optimizer_OPL)
            self.optimizers.append(self.optimizer_ME)
            self.optimizers.append(self.optimizer_MP)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))


        print('---------- Networks initialized -------------')
        print('------------- Content Encoder ---------------')
        networks.print_network(self.netCE)
        print('------------ Offsets Predictor Low --------------')
        networks.print_network(self.netOPL)
        print('------------ Offsets Predictor High --------------')
        networks.print_network(self.netOPH)
        print('-------------- Motion Encoder ---------------')
        networks.print_network(self.netME)
        print('------------- Motion Predictor --------------')
        networks.print_network(self.netMP)
        print('------------ Generator/Decoder --------------')
        networks.print_network(self.netG)
        if self.isTrain:
            print('-------------- Discriminator ---------------')
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self,input):
        # X: batchsize, seq_len_x, inchains, size(0), size(1)
        # Y: batchsize, seq_len_y, inchains, size(0), size(1)
        input_X = input[0]
        input_Y = input[1]
        self.image_paths   = input[2]
        #self.input_X.resize_(input_X.size()).copy_(input_X)
        #self.input_Y.resize_(input_Y.size()).copy_(input_Y)
        if len(self.gpu_ids) > 0:
            self.input_X = [x.clone().cuda() for x in input_X]
            self.input_Y = [y.clone().cuda() for y in input_Y]
        else:
            self.input_X = [x.clone() for x in input_X]
            self.input_Y = [y.clone() for y in input_Y]
        #self.image_paths = input['X_paths']

    def forward_G(self):
        self.real_X = [Variable(x) for x in self.input_X]
        self.real_Y = [Variable(y) for y in self.input_Y]
        self.seq_len = len(self.real_X)
        self.pre_len = len(self.real_Y)

    def forward_P(self):
        self.real_X = [Variable(x) for x in self.input_X]
        self.real_Y = [Variable(y) for y in self.input_Y]

    def test(self):
      self.real_X = [Variable(x,volatile=True) for x in self.input_X]
      self.real_Y = [Variable(y,volatile=True) for y in self.input_Y]
      self.seq_len = len(self.real_X)
      self.pre_len = len(self.real_Y)
      self.forward()
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        #one = Variable(torch.FloatTensor(pred_real.size(0),1).fill_(1).cuda(),requires_grad=False)
        loss_D_real = self.criterionGAN(pred_real, True) #True
        # Fake
        pred_fake = netD.forward(fake.detach())
        #zero = Variable(torch.FloatTensor(pred_fake.size(0),1).fill_(0).cuda(),requires_grad=False)
        loss_D_fake = self.criterionGAN(pred_fake, False) #False
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake = self.fake_pool.query(self.fakes)
        self.real = Variable(torch.cat(self.input_Y, 0))
        #self.real = Variable(self.input_Y[self.T])
        self.loss_D = self.backward_D_basic(self.netD, self.real, fake)

    def backward_G(self):
        #lambda_idt = self.opt.identity
        lambda_fea = self.opt.lambda_fea
        lambda_pix = self.opt.lambda_pix
        lambda_dif = self.opt.lambda_dif
        lambda_pre = self.opt.lambda_pre
        lambda_gan = self.opt.lambda_gan
        lambda_tra = 10
        lambda_gdl = 5
        #for p in self.netME.parameters(): # reset requires_grad
        #    p.requires_grad = True # they are set to False below in netG update
        # forward
        
        self.X_T = torch.cat(self.real_X,1)    # bs, len*nc, s0, s1
        
        batch_size = self.real_X[0].size(0)

        self.T = random.randint(0,self.pre_len-1)       
        target = min(self.T+1,self.pre_len-1)
        self.encs_xs = self.netCE.forward(self.X_T,[],[],[])
        ####################################################### flow prediction ####################################################################
        hidden_state_OPL = self.netOPL.init_hidden(batch_size)
        #hidden_state_OPH = self.netOPH.init_hidden(batch_size)

        ln = 4
        hn = 1
        FL = self.encs_xs[0][ln] #[bs,,nc,s0,s1] high level feature
        #HL = self.encs_xs[0][hn] # low level feature 
     
        init_offsets_l = self.netOPL.init_offset(batch_size)#Variable(torch.zeros(batch_size,1,FL.size(1)*2,FL.size(2),FL.size(3))).cuda()
        #init_offsets_h = self.netOPH.init_offset(batch_size)# Variable(torch.zeros(batch_size,1,HL.size(1)*2,HL.size(2),HL.size(3))).cuda()
        #------------------------------------------- encode offset--------------------------------------------
        hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(FL,init_offsets_l,hidden_state_OPL)
        #hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(HL,init_offsets_h,hidden_state_OPH)

        ##------------------------------------------prediction------------------------------------------##
        target_l = Variable(self.Tensor(pred_offsets_l.size()).zero_())
        target_l = target_l.squeeze(1)       
        hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(FL,pred_offsets_l,hidden_state_OPL)               
        self.loss_flow_coh_1 = self.criterionCoh(pred_offsets_l,target_l)
        self.low_feats = [SFL]

        '''target_h = Variable(self.Tensor(pred_offsets_h.size()).zero_())
        target_h = target_h.squeeze(1)
        hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(HL,pred_offsets_h,hidden_state_OPH)
        self.loss_flow_coh_h = self.criterionCoh(pred_offsets_h,target_h)   
        self.high_feats =[SHL]
        '''
        self.offsets_lx = [pred_offsets_l.data[:,0,...].unsqueeze(1)]
        self.offsets_ly = [pred_offsets_l.data[:,1,...].unsqueeze(1)]
        
        for t in range(1,self.pre_len):
            hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(FL,pred_offsets_l,hidden_state_OPL)           
            self.loss_flow_coh_1 += self.criterionCoh(pred_offsets_l,target_l)           
            self.low_feats += [SFL]

            '''hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(HL,pred_offsets_h,hidden_state_OPH)
            self.loss_flow_coh_h += self.criterionCoh_h(pred_offsets_h,target_h)
            self.high_feats +=[SHL]
            '''
            self.offsets_lx += [pred_offsets_l.data[:,0,...].unsqueeze(1)]
            self.offsets_ly += [pred_offsets_l.data[:,1,...].unsqueeze(1)]

        self.loss_flow_coh_1  = self.loss_flow_coh_1*0.01
        #self.loss_flow_coh_h  = self.loss_flow_coh_h*0.01
        ####################################################### pixel prediction ####################################################################

        #latent = [self.high_feats[0],self.low_feats[0]] # flow 1,2
        #layer_idx = [hn,ln]
        latent = [self.low_feats[0]] # flow 1
        layer_idx = [ln]
        enc_xt, fake = self.netCE.forward(self.X_T,latent,layer_idx,[])
        self.fakes = [fake]
        
        #self.loss_gan = self.criterionGAN(self.netD.forward(self.fakes[0]), True)*lambda_gan
        self.loss_pix = self.criterionPixel(self.fakes[0], self.real_Y[0])
        self.loss_gdl = self.criterionGDL(self.fakes[0], self.real_Y[0])

        for t in range(1,self.pre_len):
            #latent = [self.high_feats[t],self.low_feats[t]]         
            latent = [self.low_feats[t]] # flow 1 
            enc_xt, fake = self.netCE.forward(self.X_T,latent,layer_idx,[])            
            self.fakes += [fake]
            #self.loss_gan += self.criterionGAN(self.netD.forward(self.fakes[t]), True)*lambda_gan
            self.loss_pix += self.criterionPixel(self.fakes[t],self.real_Y[t])
            self.loss_gdl += self.criterionGDL(self.fakes[t], self.real_Y[t])


        ####################################################### loss backward ####################################################################
        self.loss_pix = self.loss_pix*lambda_pix
        self.loss_gdl = self.loss_gdl*lambda_gdl   
        self.loss_G = self.loss_pix +self.loss_flow_coh_1 +self.loss_gdl+ self.loss_flow_coh_h#+self.loss_flow_trip_h + self.loss_flow_trip_l#+ self.loss_flow_trip_y#+self.loss_sim #+ self.loss_gan + self.loss_pix + self.loss_gdl
        self.loss_G.backward()


        self.loss_copy_last = self.criterionPixel(self.real_X[-1],self.real_Y[0])*10


    def forward(self):               
        
        self.X_T = torch.cat(self.real_X,1)    # bs, len*nc, s0, s1
        
        batch_size = self.real_X[0].size(0)              
        self.encs_xs = self.netCE.forward(self.X_T,[],[],[])        
        ####################################################### flow prediction ####################################################################
        hidden_state_OPL = self.netOPL.init_hidden(batch_size)
        #hidden_state_OPH = self.netOPH.init_hidden(batch_size)
        # global_feature_maps
        #initialization of the Offsets Predictor
        #self.enc_x0, dec_x0 = self.netCE.forward(self.real_X[0],[Variable(self.latent_y[0].data)])
        ln = 0
        hn = 1
        FL = self.encs_xs[0][ln] #[bs,,nc,s0,s1] high level feature
        #HL = self.encs_xs[0][hn] # low level feature 
        #HL = self.X_T       
        init_offsets_l = self.netOPL.init_offset(batch_size)#Variable(torch.zeros(batch_size,1,FL.size(1)*2,FL.size(2),FL.size(3))).cuda()
        #init_offsets_h = self.netOPH.init_offset(batch_size)# Variable(torch.zeros(batch_size,1,HL.size(1)*2,HL.size(2),HL.size(3))).cuda()
        #------------------------------------------- encode --------------------------------------------
        hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(FL,init_offsets_l,hidden_state_OPL)
        #hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(HL,init_offsets_h,hidden_state_OPH)
               
        ##------------------------------------------prediction------------------------------------------##         
        hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(FL,pred_offsets_l,hidden_state_OPL)                       
        self.low_feats = [SFL]       
        #hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(HL,pred_offsets_h,hidden_state_OPH)                
        #self.high_feats =[SHL]       

        for t in range(1,self.pre_len):
            hidden_state_OPL, pred_offsets_l, SFL = self.netOPL.forward(FL,pred_offsets_l,hidden_state_OPL)                                         
            self.low_feats += [SFL]
            #hidden_state_OPH, pred_offsets_h, SHL = self.netOPH.forward(HL,pred_offsets_h,hidden_state_OPH)                       
            #self.high_feats +=[SHL]
        
        ####################################################### pixel prediction ####################################################################       
        #latent = [self.high_feats[0],self.low_feats[0]] # flow 1,2      
        #layer_idx = [hn,ln]   
        latent = [self.low_feats[0]] # flow 1      
        layer_idx = [ln]      
        enc_xt, fake = self.netCE.forward(self.X_T,latent,layer_idx,[])
        self.fakes = [fake]

        for t in range(1,self.pre_len):
            #latent = [self.high_feats[t],self.low_feats[t]]       
            latent = [self.low_feats[t]] # flow 1       
            enc_xt, fake = self.netCE.forward(self.X_T,latent,layer_idx,[])            
            self.fakes += [fake]

    def optimize_generator(self):
        # forward_G
        self.forward_G()
        #Encoder
        self.optimizer_CE.zero_grad()
        self.optimizer_OPH.zero_grad()
        self.optimizer_OPL.zero_grad()
        self.optimizer_ME.zero_grad()
        self.optimizer_MP.zero_grad()
        # Decoder
        self.optimizer_G.zero_grad()


        if self.use_cycle:
            self.backward_cycle()
        else:
            self.backward_G()

        self.optimizer_CE.step()
        self.optimizer_OPH.step()
        self.optimizer_OPL.step()
        self.optimizer_ME.step()
        self.optimizer_MP.step()

        self.optimizer_G.step()

        # D

    def optimize_discriminator(self):
        #for p in netD.parameters(): # reset requires_grad
        #    p.requires_grad = True # they are set to False below in netG update
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def optimize_parameters(self):
        self.optimize_generator()
        #self.optimize_discriminator()

    def get_current_errors(self):
        #GAN = self.loss_gan.data[0]
        PIX = self.loss_pix.data[0]
        GDL = self.loss_gdl.data[0]
        #DES = self.loss_D.data[0]
        COH_L = self.loss_flow_coh_1.data[0]*100
        COH_H = 0#self.loss_flow_coh_h.data[0]*100
        CLST = self.loss_copy_last.data[0]
        return OrderedDict([('Pixel', PIX),('CopyLast',CLST),('COH_0',COH_L),('COH_1',COH_H),('GDL',GDL)])
        

    def get_current_visuals(self):
        images = []
        for i in xrange(self.seq_len):
            name = 'frame_'+str(i)
            image = util.tensor2im(self.real_X[i].data)
            images += [(name,image)]

        for j in xrange(self.pre_len):
            real_name = 'frame_'+str(self.seq_len+j)
            real_image = util.tensor2im(self.real_Y[j].data)
            fake_name = 'fake_'+str(self.seq_len+j)
            fake_image = util.tensor2im(self.fakes[j].data)
            images += [(real_name, real_image),(fake_name,fake_image)]
            offset_x = util.tensor2im(self.offsets_lx[j])
            offset_y = util.tensor2im(self.offsets_ly[j])
            images += [('offset_x_%s'%(j+self.seq_len), offset_x)]
            images += [('offset_y_%s'%(j+self.seq_len), offset_y)]
                 
        return OrderedDict(images)
        #return OrderedDict([('real_Began', real_Y_0), ('Pred_Began', fake_Y_0), ('real_End', real_Y_E),
        #                    ('Pred_End', fake_Y_E)])

    def save(self, label):
        self.save_network(self.netCE, 'CE', label, self.gpu_ids)
        self.save_network(self.netOPH, 'OPH', label, self.gpu_ids)
        self.save_network(self.netOPL, 'OPL', label, self.gpu_ids)
        self.save_network(self.netME, 'ME', label, self.gpu_ids)
        self.save_network(self.netMP, 'MP', label, self.gpu_ids)
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def get_all_current_visuals(self):
        images = []
        for i in xrange(self.seq_len):
            name = 'gt_%04d'%i
            image = util.tensor2im(self.real_X[i].data)
            images += [(name,image)]

        for i in xrange(self.pre_len):
            name = 'gt_%04d'%(i+self.seq_len)
            image = util.tensor2im(self.real_Y[i].data)
            images += [(name,image)]

        for i in xrange(self.seq_len):
            name = 'pred_%04d'%i
            image = util.tensor2im(self.real_X[i].data)
            images += [(name,image)]

        for j in xrange(self.pre_len):
            name = 'pred_%04d'%(j+self.seq_len)
            image = util.tensor2im(self.fakes[j].data)
            images += [(name,image)]

        return OrderedDict(images)
    '''def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_CE.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_SE.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_ME.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_MP.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr'''
