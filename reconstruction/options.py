import os
import argparse


class Options:
    def __init__(self, isTrain):
        self.project_name = None
        self.dataset = None
        self.fold = None
        self.result_dir = None
        self.isTrain = isTrain
        self.model = dict()
        self.train = dict()
        self.test = dict()
        self.transform = dict()
        self.post = dict()
        self.gpu = None
        # self.tags = None
        # self.notes = None

        self.data_name = {'rsna': 'RSNA', 'vin': 'VinDr-CXR', 'brain': 'Brain Tumor', 'lag': 'LAG', 'isic': 'ISIC2018',
                          'c16': 'Camelyon16', 'brats': 'BraTS2021'}
        self.epochs = {'rsna': 250, 'vin': 250, 'brain': 600, 'lag': 250, 'brats': 250,
                       'oct': 250, 'colon': 250}
        self.in_c = {'c16': 3, 'colon': 1}

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        parser.add_argument("-d", '--dataset', type=str, default='rsna',
                            help='rsna, vin, brain, lag, brats, oct, colon, isic')
        parser.add_argument("-g", '--gpu', type=int, default=6, help='select gpu devices')
        parser.add_argument("-p", '--project-name', type=str, default="MedIAnomaly", required=False,
                            help='Name of the current project. eg, MedIAnomaly')
        parser.add_argument("-n", '--notes', type=str, default="default", required=False,
                            help='Notes of the current experiment. e.g., ae-architecture')
        parser.add_argument("-f", '--fold', type=str, default='0', help='0-4, five fold cross validation')
        parser.add_argument("-m", '--model-name', type=str, default='ae', help='ae, aeu, memae')
        parser.add_argument('--input-size', type=int, default=64, help='input size of the image')

        # Parameters only for reconstruction model
        parser.add_argument('--base-width', type=int, default=16,
                            help='Base channels of CNN layers. Please do not modify this value, and adjust the '
                                 'expansion instead.')
        parser.add_argument('--expansion', type=int, default=1, help='expansion of the base channels.')
        parser.add_argument('--hidden-num', type=int, default=1024, help='Hidden size of the bottleneck')
        parser.add_argument("-ls", '--latent-size', type=int, default=16,
                            help='latent size of the reconstruction model')
        parser.add_argument('--en-depth', type=int, default=1, help='Depth of each encoder block')
        parser.add_argument('--de-depth', type=int, default=1, help='Depth of each decoder block')

        parser.add_argument('--train-epochs', type=int, default=250, help='number of training epochs')
        parser.add_argument('--train-eval-freq', type=int, default=25, help='epoch to evaluate')
        parser.add_argument('-bs', '--train-batch-size', type=int, default=64, help='batch size')
        parser.add_argument('--train-lr', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--train-weight-decay', type=float, default=0, help='weight decay')
        parser.add_argument('--train-seed', type=int, default=None, help='random seed')

        parser.add_argument("-save", '--test-save-flag', action='store_true')
        parser.add_argument("--heaviside", action='store_true')
        parser.add_argument('--test-model-path', type=str, default=None, help='model path to test')

        parser.add_argument('--epsilon', type=float, default=0, help='quasibinarizer (by Shouhei Hanaoka) epsilon. zero means no binarizing')
        parser.add_argument('--firing_rate_cost_weight', type=float, default=0, help='quasibinarizer (by Shouhei Hanaoka) neuron firing rate penarizing factor. zero means no penalty')
        parser.add_argument('--perceptual_loss_weight', type=float, default=1, help='perceptual loss weight. zero means no perceptual loss')

        parser.add_argument('--wf', type=int, default=4, help='number of filters in the first layer is 2**wf')
        parser.add_argument('--latent_size_with_noise', type=int, default=4096, help='latent size for noised models')
        parser.add_argument('--noise', type=float, default=0.0, help='denoising AE noise level (per image standard deviation)')
        # v31: default True (this was the behaviour actually used in v30); disable with --no-using_identity_connection
        parser.add_argument('--using_identity_connection', action=argparse.BooleanOptionalAction, default=True,
                            help='residual (identity) connection inside the U-Net conv blocks')
        parser.add_argument('--not_use_log_var', action='store_true')
        parser.add_argument('--use_KL_divergence', action='store_true')
        parser.add_argument('--rho', type=float, default=0.05)
        parser.add_argument('--attention_gate', action='store_true')
        # v31
        parser.add_argument('--top_mixer', type=str, default='attn', choices=['attn', 'fc'],
                            help='bottom bottleneck mixer: attn (token transformer on 8x8 grid) or fc (legacy dense layers)')
        parser.add_argument('--top_attn_depth', type=int, default=2, help='number of attention blocks before/after the bottom QB')
        parser.add_argument('--norm_type', type=str, default='group', choices=['group', 'batch'],
                            help='normalisation around the bottom QB and in the attention gates (group: per-sample; batch: legacy)')
        parser.add_argument('--test_batch_size', type=int, default=64,
                            help='batch size of the evaluation loop (v31; was 1). Results are batch-independent with GroupNorm/eval-mode BN')
        parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers (0 = load in the main process)')
        parser.add_argument('--perceptual_bf16', action=argparse.BooleanOptionalAction, default=True,
                            help='run the VGG19 perceptual loss in bf16 autocast during training (evaluation stays fp32)')
        parser.add_argument('--grad_clip', type=float, default=1.0,
                            help='clip the global gradient norm to this value (0 = no clipping; the norm is logged either way)')
        parser.add_argument('--ldp_samples', type=int, default=8,
                            help='number of independent noise draws averaged for the LDP (noisy) readout at test time (1 = single draw only)')
        parser.add_argument('--full_eval', action='store_true',
                            help='also run range coding, png residual coding, one-class/few-shot classifiers and t-SNE (slow)')

        args = parser.parse_args()

        self.gpu = args.gpu
        self.dataset = args.dataset
        self.project_name = args.project_name
        self.fold = args.fold
        self.result_dir = os.path.expanduser("~") + f'/Experiment/MedIAnomaly/{self.dataset}'

        self.model['name'] = args.model_name
        self.model['in_c'] = self.in_c.setdefault(self.dataset, 1)
        self.model['input_size'] = args.input_size

        # added by Shouhei Hanaoka
        self.model['epsilon'] = args.epsilon
        self.model['firing_rate_cost_weight'] = args.firing_rate_cost_weight
        self.model['perceptual_loss_weight'] = args.perceptual_loss_weight
        self.model['heaviside'] = args.heaviside
        self.model['wf'] = args.wf
        self.model['latent_size_with_noise'] = args.latent_size_with_noise

        # Parameters only for reconstruction model
        self.model['base_width'] = args.base_width
        self.model['expansion'] = args.expansion
        self.model['hidden_num'] = args.hidden_num
        self.model['ls'] = args.latent_size
        self.model['en_depth'] = args.en_depth
        self.model['de_depth'] = args.de_depth

        self.model['using_identity_connection'] = args.using_identity_connection
        self.model['not_use_log_var'] = args.not_use_log_var
        self.model['use_KL_divergence'] = args.use_KL_divergence
        self.model['rho'] = args.rho
        self.model['attention_gate'] = args.attention_gate
        self.model['top_mixer'] = args.top_mixer
        self.model['top_attn_depth'] = args.top_attn_depth
        self.model['norm_type'] = args.norm_type

        # --- training params --- #
        self.train['save_dir'] = '{}/{}/fold_{}'.format(self.result_dir, self.model['name'], self.fold)
        self.train['epochs'] = self.epochs.setdefault(self.dataset, 250)
        self.train['eval_freq'] = args.train_eval_freq
        self.train['batch_size'] = args.train_batch_size
        self.train['lr'] = args.train_lr
        self.train['weight_decay'] = args.train_weight_decay
        self.train['seed'] = args.train_seed

        self.train['noise_level'] = args.noise

        # --- test parameters --- #
        self.test['save_flag'] = args.test_save_flag
        self.test['full_eval'] = args.full_eval
        self.test['ldp_samples'] = args.ldp_samples
        self.train['grad_clip'] = args.grad_clip
        self.test['batch_size'] = args.test_batch_size
        self.train['num_workers'] = args.num_workers
        self.train['perceptual_bf16'] = args.perceptual_bf16
        self.test['save_dir'] = '{:s}/test_results'.format(self.train['save_dir'])
        if not args.test_model_path:
            self.test['model_path'] = '{:s}/checkpoints/model.pth'.format(self.train['save_dir'])

    def save_options(self):
        if not os.path.exists(self.train['save_dir']):
            os.makedirs(self.train['save_dir'], exist_ok=True)
            os.makedirs(os.path.join(self.train['save_dir'], 'test_results'), exist_ok=True)
            os.makedirs(os.path.join(self.train['save_dir'], 'checkpoints'), exist_ok=True)

        filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        file = open(filename, 'w')
        groups = ['model', 'train', 'test', 'transform']

        file.write("# ---------- Options ---------- #")
        file.write('\ndataset: {:s}\n'.format(self.dataset))
        file.write('isTrain: {}\n'.format(self.isTrain))
        for group, options in self.__dict__.items():
            if group not in groups:
                continue
            file.write('\n\n-------- {:s} --------\n'.format(group))
            if group == 'transform':
                for name, val in options.items():
                    if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                        file.write("{:s}:\n".format(name))
                        for t_val in val.transforms:
                            file.write("\t{:s}\n".format(t_val.__class__.__name__))
            else:
                for name, val in options.items():
                    file.write("{:s} = {:s}\n".format(name, repr(val)))
        file.close()
