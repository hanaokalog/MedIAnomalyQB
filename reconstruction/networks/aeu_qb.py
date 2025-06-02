import torch
import torch.nn as nn
from networks.aeu import AEU
from networks.base_units.blocks import BasicBlock, BottleNeck, SpatialBottleNeck, BottleNeckWithQuasiBinarize

class AEU_QB(AEU):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1, spatial=False, epsilon=0.):
        super(AEU_QB, self).__init__(input_size, in_planes, base_width, expansion, mid_num, latent_size, en_num_layers,
                                     de_num_layers)

        self.bottle_neck = BottleNeckWithQuasiBinarize(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                        latent_size=latent_size, epsilon=epsilon)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        z, de4 = bottle_out['z'], bottle_out['out']
        firing_rate = bottle_out['firing_rate']
        real_firing_rate = bottle_out['real_firing_rate']

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat, log_var = self.de_block4(de1).chunk(2, 1)

        return {'x_hat': x_hat, 'log_var': log_var, 'z': z, 'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3], 'firing_rate': firing_rate, 'real_firing_rate': real_firing_rate}
