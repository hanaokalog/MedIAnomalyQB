import random
import os
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
import torch.nn as nn

from networks.ae import AE
from networks.ae_qb import AE_QB
from networks.aeu_qb import AEU_QB
from networks.mem_ae import MemAE
from networks.aeu import AEU
from networks.vae import VAE
from networks.ganomaly import Ganomaly
from networks.constrained_ae import ConstrainedAE
from networks.unet import UNet

from dataloaders.data_utils import get_transform, get_data_path
from dataloaders.dataload import MedAD, BraTSAD, OCT2017, ColonAD, ISIC2018, CpChildA, Camelyon16AD
import wandb
from thop import profile
from utils.losses import *


class BaseWorker:
    def __init__(self, opt):
        self.logger = None
        self.opt = opt
        self.seed = None
        self.train_set = None
        self.test_set = None
        self.train_loader = None
        self.test_loader = None
        self.scheduler = None
        self.optimizer = None

        self.net = None
        self.criterion = None

    def set_gpu_device(self):
        torch.cuda.set_device(self.opt.gpu)
        print("=> Set GPU device: {}".format(self.opt.gpu))

    def set_seed(self):
        self.seed = self.opt.train['seed']
        if self.seed is None:
            self.seed = np.random.randint(1, 999999)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def set_network_loss(self):
        if self.opt.model['name'] in ['ae', 'ceae', 'ae-ssim', 'ae-l1', 'ae-perceptual']:
            self.net = AE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                          base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                          mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                          en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            if self.opt.model['name'] == 'ae-ssim':
                self.criterion = SSIMLoss()
            elif self.opt.model['name'] == 'ae-l1':
                self.criterion = L1Loss()
            elif self.opt.model['name'] == 'ae-perceptual':
                self.criterion = RelativePerceptualL1Loss()
            else:
                self.criterion = AELoss()
        elif self.opt.model['name'] in ['ae-qb']:
            self.net = AE_QB(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                          base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                          mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                          en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"],
                          epsilon=self.opt.model['epsilon'])
            self.criterion = AE_QBLoss(firing_rate_cost_weight=self.opt.model['firing_rate_cost_weight'])
        elif self.opt.model['name'] in ['aeu-qb']:
            self.net = AEU_QB(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                          base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                          mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                          en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"],
                          epsilon=self.opt.model['epsilon'])
            self.criterion = AEU_QBLoss(firing_rate_cost_weight=self.opt.model['firing_rate_cost_weight'])
        elif self.opt.model['name'] in ['aeu-perceptual-qb']:
            self.net = AEU_QB(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                          base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                          mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                          en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"],
                          epsilon=self.opt.model['epsilon'])
            self.criterion = AEU_Perceptual_QBLoss(
                firing_rate_cost_weight=self.opt.model['firing_rate_cost_weight'],
                perceptual_loss_weight=self.opt.model['perceptual_loss_weight']
            )
        elif self.opt.model['name'] == 'ae-spatial':
            self.net = AE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                          base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                          mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                          en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"],
                          spatial=True)
            self.criterion = AELoss()
        elif self.opt.model['name'] == 'ae-grad':
            self.net = AE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                          base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                          mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                          en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            self.criterion = AELoss(grad_score=True)
        elif self.opt.model['name'] == 'constrained-ae':
            self.net = ConstrainedAE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                                     base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                                     mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                                     en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            self.criterion = ConstrainedAELoss()
        elif self.opt.model['name'] == 'memae':
            self.net = MemAE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                             base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                             mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                             en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            self.criterion = MemAELoss()
        elif self.opt.model['name'] == 'aeu':
            self.net = AEU(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                           base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                           mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                           en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            self.criterion = AEULoss()
        elif 'vae' in self.opt.model['name']:
            self.net = VAE(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                           base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                           mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                           en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            if self.opt.model['name'] == 'vae':
                self.criterion = VAELoss()
            elif self.opt.model['name'] == 'vae-elbo':
                self.criterion = VAELoss(grad='elbo')
            elif self.opt.model['name'] == 'vae-kl':
                self.criterion = VAELoss(grad='kl')
            elif self.opt.model['name'] == 'vae-rec':
                self.criterion = VAELoss(grad='rec')
            elif self.opt.model['name'] == 'vae-combi':
                self.criterion = VAELoss(grad='combi')
            else:
                raise Exception("Invalid VAE model: {}".format(self.opt.model['name']))
        elif self.opt.model['name'] == 'ganomaly':
            self.net = Ganomaly(input_size=self.opt.model['input_size'], in_planes=self.opt.model['in_c'],
                                base_width=self.opt.model['base_width'], expansion=self.opt.model['expansion'],
                                mid_num=self.opt.model['hidden_num'], latent_size=self.opt.model['ls'],
                                en_num_layers=self.opt.model["en_depth"], de_num_layers=self.opt.model["de_depth"])
            self.criterion = GANomalyLoss()
        elif self.opt.model['name'] == 'dae':
            self.net = UNet(in_channels=self.opt.model['in_c'], n_classes=self.opt.model['in_c'])
            self.criterion = AELoss()
        else:
            raise NotImplementedError("Unexpected model name: {}".format(self.opt.model['name']))
        self.net = self.net.cuda()

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.train['lr'],
                                          weight_decay=self.opt.train['weight_decay'])
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                             T_max=self.opt.train['epochs'],
        #                                                             eta_min=5e-5)

    def set_dataloader(self, test=False):
        data_path = get_data_path(dataset=self.opt.dataset)
        train_transform = get_transform(self.opt)
        test_transform = get_transform(self.opt)

        context_encoding = True if self.opt.model["name"] == "ceae" else False

        if self.opt.dataset in ['rsna', 'vin', 'brain', 'lag']:
            if not test:
                self.train_set = MedAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                       transform=train_transform, mode='train', context_encoding=context_encoding)
            self.test_set = MedAD(main_path=data_path, img_size=self.opt.model['input_size'], transform=test_transform,
                                  mode='test')
        elif self.opt.dataset == 'brats':
            if not test:
                self.train_set = BraTSAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                         transform=train_transform, mode='train', context_encoding=context_encoding)
            self.test_set = BraTSAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test')
        elif self.opt.dataset == 'c16':
            if not test:
                self.train_set = Camelyon16AD(main_path=data_path, img_size=self.opt.model['input_size'],
                                              transform=train_transform, mode='train', n_channel=self.opt.model["in_c"],
                                              context_encoding=context_encoding)
            self.test_set = Camelyon16AD(main_path=data_path, img_size=self.opt.model['input_size'],
                                         transform=test_transform, mode='test', n_channel=self.opt.model["in_c"])
        elif self.opt.dataset == "isic":
            if not test:
                self.train_set = ISIC2018(main_path=data_path, img_size=self.opt.model['input_size'],
                                          transform=train_transform, mode='train', context_encoding=context_encoding,
                                          n_channel=self.opt.model["in_c"])
            self.test_set = ISIC2018(main_path=data_path, img_size=self.opt.model['input_size'],
                                     transform=test_transform, mode='test',
                                     n_channel=self.opt.model["in_c"])
        elif self.opt.dataset == "oct":
            if not test:
                self.train_set = OCT2017(main_path=data_path, img_size=self.opt.model['input_size'],
                                         transform=train_transform, mode='train')
            self.test_set = OCT2017(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test')
        elif self.opt.dataset == "colon":
            if not test:
                self.train_set = ColonAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                         transform=train_transform, mode='train', n_channel=self.opt.model["in_c"])
            self.test_set = ColonAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test', n_channel=self.opt.model["in_c"])
        elif self.opt.dataset == 'cpchild':
            if not test:
                self.train_set = CpChildA(main_path=data_path, img_size=self.opt.model['input_size'],
                                          transform=train_transform, mode='train')
            self.test_set = CpChildA(main_path=data_path, img_size=self.opt.model['input_size'],
                                     transform=test_transform, mode='test')
        else:
            raise Exception("Invalid dataset: {}".format(self.opt.dataset))

        if not test:
            self.train_loader = DataLoader(self.train_set, batch_size=self.opt.train['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)

    def set_logging(self, test=False):
        example_in = torch.zeros((1, self.opt.model["in_c"],
                                  self.opt.model['input_size'], self.opt.model['input_size'])).cuda()

        # flops, params = profile(self.net, inputs=(example_in,))
        flops, params = profile(copy.deepcopy(self.net), inputs=(example_in,))
        flops, params = round(flops * 1e-6, 4), round(params * 1e-6, 4)  # 1e6 = M
        flops, params = str(flops) + "M", str(params) + "M"

        exp_configs = {"dataset": self.opt.dataset,
                       "model": self.opt.model["name"],
                       "in_channels": self.opt.model["in_c"],
                       "input_size": self.opt.model['input_size'],

                       "base_width": self.opt.model['base_width'],
                       "expansion": self.opt.model['expansion'],
                       "mid_num": self.opt.model['hidden_num'],
                       "latent_size": self.opt.model['ls'],
                       "en_num_layers": self.opt.model["en_depth"],
                       "de_num_layers": self.opt.model["de_depth"],

                       "epsilon": self.opt.model['epsilon'],
                       "firing_rate_cost_weight": self.opt.model['firing_rate_cost_weight'],
                       "perceptual_loss_weight": self.opt.model['perceptual_loss_weight'],

                       "epochs": self.opt.train["epochs"],
                       "batch_size": self.opt.train["batch_size"],
                       "lr": self.opt.train["lr"],
                       "weight_decay": self.opt.train["weight_decay"],
                       "seed": self.seed,

                       "num_params": params,
                       "FLOPs": flops}

        if not test:
            self.logger = wandb.init(project=self.opt.project_name, config=exp_configs)
        print("============= Configurations =============")
        for key, values in exp_configs.items():
            print(key + ":" + str(values))
        print()

    def close_network_grad(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def enable_network_grad(self):
        for param in self.net.parameters():
            param.requires_grad = True

    def set_test_loader(self):
        """ Use for only evaluation"""
        data_path = get_data_path(dataset=self.opt.dataset)
        test_transform = get_transform(self.opt)
        if self.opt.dataset in ['rsna', 'vin', 'brain', 'lag']:
            self.test_set = MedAD(main_path=data_path, img_size=self.opt.model['input_size'], transform=test_transform,
                                  mode='test')
        elif self.opt.dataset == 'brats':
            self.test_set = BraTSAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test')
        elif self.opt.dataset == 'c16':
            self.test_set = Camelyon16AD(main_path=data_path, img_size=self.opt.model['input_size'],
                                         transform=test_transform, mode='test', n_channel=self.opt.model["in_c"])
        elif self.opt.dataset == "oct":
            self.test_set = OCT2017(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test')
        elif self.opt.dataset == "colon":
            self.test_set = ColonAD(main_path=data_path, img_size=self.opt.model['input_size'],
                                    transform=test_transform, mode='test', n_channel=self.opt.model["in_c"])
        elif self.opt.dataset == "isic":
            self.test_set = ISIC2018(main_path=data_path, img_size=self.opt.model['input_size'],
                                     transform=test_transform, mode='test')
        elif self.opt.dataset == 'cpchild':
            self.test_set = CpChildA(main_path=data_path, img_size=self.opt.model['input_size'],
                                     transform=test_transform, mode='test')
        else:
            raise Exception("Invalid dataset: {}".format(self.opt.dataset))
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        print("=> Set test dataset: {} | Input size: {} | Batch size: {}".format(self.opt.dataset,
                                                                                 self.opt.model['input_size'], 1))

    def save_checkpoint(self):
        torch.save(self.net.state_dict(), os.path.join(self.opt.train['save_dir'], "checkpoints", "model.pt"))

    def load_checkpoint(self):
        model_path = os.path.join(self.opt.train['save_dir'], "checkpoints", "model.pt")
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:{}".format(self.opt.gpu))))
        print("=> Load model from {}".format(model_path))

    def run_eval(self):
        results = self.evaluate()
        metrics_save_path = os.path.join(self.opt.train['save_dir'], "metrics.txt")
        with open(metrics_save_path, "w") as f:
            for key, value in results.items():
                f.write(str(key) + ": " + str(value) + "\n")
                print(key + ": {:.4f}".format(value))

    def evaluate(self) -> dict:
        pass
