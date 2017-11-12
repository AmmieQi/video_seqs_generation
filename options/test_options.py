from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--lambda_fea', type=float, default=1.0, help='weight for feature loss ')
        self.parser.add_argument('--lambda_pix', type=float, default=10.0, help='weight for pixel loss ')
        self.parser.add_argument('--lambda_dif', type=float, default=1.0, help='weight for pixel loss ')
        self.parser.add_argument('--lambda_gan', type=float, default=0.05, help='GAN loss')
        self.parser.add_argument('--lambda_pre', type=float, default=1.0, help='prediction loss')
        self.isTrain = False
