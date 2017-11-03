from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=800, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=51, help='# of iter to linearly decay learning rate to zero')
 	self.parser.add_argument('--d_step', type=int, default=5, help='# of train discriminator')
 	self.parser.add_argument('--p_step', type=int, default=1, help='# of train predictor')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_fea', type=float, default=1.0, help='weight for feature loss ')
        self.parser.add_argument('--lambda_pix', type=float, default=10.0, help='weight for pixel loss ')
        self.parser.add_argument('--lambda_dif', type=float, default=1.0, help='weight for pixel loss ')
        self.parser.add_argument('--lambda_gan', type=float, default=0.05, help='GAN loss')
        self.parser.add_argument('--lambda_pre', type=float, default=1.0, help='prediction loss')
        self.parser.add_argument('--pool_size', type=int, default=200, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
	self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=30, help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain = True
