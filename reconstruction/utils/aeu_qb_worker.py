import time
import torch
import os
import sys
from sklearn import metrics
from utils.util import compute_best_dice
import numpy as np
import scipy.ndimage

import gzip

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from utils.ae_worker import AEWorker
from utils.aeu_worker import AEUWorker
from utils.util import AverageMeter

import utils.compressor
import utils.fewshot_classifiers

import wandb

from utils.blob_noise_gpu import make_noise_like_gpu



def make_noise_like(x, sigma = 1.0):
    
    shape = x.size()
    
    assert(shape[3] == shape[2])
    sz = shape[2]
    d = shape[1]
    num = shape[0]

    res = np.zeros((num, d, sz, sz))

    for i in range(num):

        mother_std = x[i,:,:,:].std().detach().cpu().numpy()

        z1 = np.random.randn(sz,sz,d)
        z2 = np.random.randn(sz,sz,1)

        rad1 = np.random.rand()*16+1
        rad2 = np.random.rand()*16+1

        for dd in range(d):
            z1[:,:,dd] = scipy.ndimage.gaussian_filter(z1[:,:,dd], rad1) 
        z2 = scipy.ndimage.gaussian_filter(z2, rad2)

        z1 /= z1.std()
        z2 /= z2.std()

        z2 = np.repeat(z2, d, axis=2)
        z2 -= np.random.rand()*2
        z2 = np.where(z2>0, z2, 0)
        z2 = z2 ** (np.random.rand()*2.0+0.01)

        z = z1 * z2

        res[i,:,:,:] = z.transpose((2,0,1)) * mother_std * sigma

    return torch.from_numpy(res.astype(np.float32)).clone()



class AEU_QBWorker(AEUWorker):
    def __init__(self, opt):
        super(AEU_QBWorker, self).__init__(opt)
        self.pixel_metric = True if self.opt.dataset == "brats" else False
        self.firing_rate_cost_weight = self.opt.model['firing_rate_cost_weight']
        self.auc_fewshot_validation = 0
        self.auc_fewshot_test = 0
        self.method_best_fewshot = 0
        self.epoch_best_fewshot = 0

        # fewshot validation system
        self.fsct = None

        # v31: GPU blob-noise generator (seeded in train_epoch from self.seed)
        self.noise_gen = None

    @staticmethod
    def heaviside_budget(fire_count, n, prefix=""):
        """Upper bounds (bits) on the information carried by the Heaviside readout, from firing frequencies."""
        if fire_count is None or n == 0:
            return {}
        p = (fire_count / n).clamp(0.0, 1.0)

        def h2(q):
            q = q.clamp(1e-12, 1 - 1e-12)
            return -(q * torch.log2(q) + (1 - q) * torch.log2(1 - q))

        n_ch = p.numel()
        p_bar = p.mean()
        sum_h2 = float(h2(p).sum())
        return {
            f"{prefix}real_firing_rate": float(p_bar),                         # mean Heaviside firing rate p_bar
            f"{prefix}heaviside_budget_bits": sum_h2,                          # sum_i h2(p_i)  (tighter)
            f"{prefix}heaviside_budget_bits_jensen": float(n_ch * h2(p_bar)),  # N h2(p_bar)
            f"{prefix}heaviside_budget_per_channel": sum_h2 / n_ch,
            f"{prefix}dead_channel_fraction": float(((p == 0) | (p == 1)).double().mean()),  # constant channels carry 0 bits
        }

    def _test_batch_size(self):
        return self.opt.test.get('batch_size', 1)

    def train_epoch(self, force_firing=False, firing_cost_multiplier=1.0, shortcut_multiplier=1.0, noise_level = 0.0, epoch=0):
        self.net.train()
        losses = AverageMeter()
        losses_recon = AverageMeter()
        losses_logvar = AverageMeter()
        losses_firing = AverageMeter()
        losses_perceptual = AverageMeter()
        firing_rates = AverageMeter()
        real_firing_rates = AverageMeter()
        
#        with torch.autograd.set_detect_anomaly(True):
        if 1:
        
            if hasattr(self.criterion, 'perceptual_loss'):
                self.criterion.perceptual_loss.train_amp = self.opt.train.get('perceptual_bf16', False)

            for idx_batch, data_batch in enumerate(self.train_loader):
                img = data_batch['img'].cuda(non_blocking=True)

                if 0 < noise_level:
                    # v31: batched GPU implementation (the CPU/scipy version took ~0.9 s per batch of 128)
                    if self.noise_gen is None:
                        self.noise_gen = torch.Generator(device=img.device)
                        self.noise_gen.manual_seed(int(self.seed) if self.seed is not None else 0)
                    img_noised = img + make_noise_like_gpu(img, noise_level, generator=self.noise_gen)
                else:
                    img_noised = img

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    net_out = self.net(img_noised, shortcut_multiplier=shortcut_multiplier)
                
                net_out["x_hat"] = net_out["x_hat"].float()
                net_out["log_var"] = net_out["log_var"].float()

                if idx_batch == 0 and epoch%5==1:
                    if self.logger is not None:
                        if(img.shape[1] == 1):
                            img_noised1 = img_noised[0,0,:,:]
                            img_denoised1 = net_out["x_hat"][0,0,:,:]
                            img_logvar1 = net_out["log_var"][0,0,:,:]
                            img_noised1 = (img_noised1 - img_noised1.min()) / (img_noised1.max() - img_noised1.min())
                            img_denoised1 = (img_denoised1 - img_denoised1.min()) / (img_denoised1.max() - img_denoised1.min())
                            img_logvar1 = (img_logvar1 - img_logvar1.min()) / (img_logvar1.max() - img_logvar1.min())
                            self.logger.log(step=epoch, data={f'imgs_train/Ep{epoch}_noised': wandb.Image(img_noised1.T[:,:,np.newaxis], caption=f'noised_Ep{epoch}', mode="L")})
                            self.logger.log(step=epoch, data={f'imgs_train/Ep{epoch}_denoised': wandb.Image(img_denoised1.T[:,:,np.newaxis], caption=f'denoised_Ep{epoch}', mode="L")})
                            self.logger.log(step=epoch, data={f'imgs_train/Ep{epoch}_logvar': wandb.Image(img_logvar1.T[:,:,np.newaxis], caption=f'logvar_Ep{epoch}', mode="L")})
                        else:
                            img_noised1 = img_noised[0,:,:,:]
                            img_denoised1 = net_out["x_hat"][0,:,:,:]
                            img_logvar1 = net_out["log_var"][0,:,:,:]
                            img_noised1 = (img_noised1 - img_noised1.min()) / (img_noised1.max() - img_noised1.min())
                            img_denoised1 = (img_denoised1 - img_denoised1.min()) / (img_denoised1.max() - img_denoised1.min())
                            img_logvar1 = (img_logvar1 - img_logvar1.min()) / (img_logvar1.max() - img_logvar1.min())
                            self.logger.log(step=epoch, data={f'imgs_train/Ep{epoch}_noised': wandb.Image(img_noised1.permute((0,1,2)), caption=f'noised_Ep{epoch}', mode="RGB")})
                            self.logger.log(step=epoch, data={f'imgs_train/Ep{epoch}_denoised': wandb.Image(img_denoised1.permute((0,1,2)), caption=f'denoised_Ep{epoch}', mode="RGB")})
                            self.logger.log(step=epoch, data={f'imgs_train/Ep{epoch}_logvar': wandb.Image(img_logvar1.permute((0,1,2)), caption=f'logvar_Ep{epoch}', mode="RGB")})

                # v31: meters hold detached GPU tensors (no autograd chain across the epoch, no per-step sync)
                firing_rates.update(net_out["firing_rate"].detach().mean(), img.size(0))
                real_firing_rates.update(net_out["real_firing_rate"].detach().mean(), img.size(0))

                loss_etc = self.criterion(img, net_out, force_firing=force_firing, firing_cost_multiplier=firing_cost_multiplier)
                loss = loss_etc['loss'].float()
                losses_recon.update(loss_etc['recon_loss'].detach().mean(), img.size(0))
                losses_logvar.update(loss_etc['log_var'].detach().mean(), img.size(0))
                losses_firing.update(loss_etc['firing_loss'].detach().mean(), img.size(0))
                if 'perceptual_loss' in loss_etc:
                    losses_perceptual.update(loss_etc['perceptual_loss'].detach().mean(), img.size(0))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                losses.update(loss.detach(), img.size(0))

            print("expected_firing_rate: {:.4f}, real_firing_rate: {:,.4f}, loss_recon: {:.4f}, loss_firing: {:.4f}, loss_perceptual: {:.4f}".format(
                    firing_rates.avg, 
                    real_firing_rates.avg,
                    losses_recon.avg, 
                    losses_firing.avg,
                    losses_perceptual.avg
            ))
        _f = lambda v: float(v)
        return (_f(losses.avg), _f(losses_recon.avg), _f(losses_logvar.avg), _f(losses_firing.avg),
                _f(losses_perceptual.avg), _f(firing_rates.avg), _f(real_firing_rates.avg))


    def evaluate(self, epoch='test'):
        self.net.eval()
        self.close_network_grad()

        # v31: range coding, png residual coding, one-class / few-shot classifiers and t-SNE are slow and
        #      are only run with --full_eval.
        full_eval = self.opt.test.get('full_eval', False)

        if full_eval:
            # calculate training_firing_rates from training dataset

            # pass 1
            firing_count = None
            count = 0
            losses_recon = []
            losses_perceptual = []
            for idx_batch, data_batch in enumerate(self.train_loader):
                # binary latent
                self.net.using_heaviside = True
                self.net.adding_noise_in_test = False

                img = data_batch['img']
                img = img.cuda()

                net_out = self.net(img)

                # count firings
                firing = net_out["z"]
                firing_partial_count = torch.sum(firing, dim=0, keepdim=True)
                if firing_count is None:
                    firing_count = torch.zeros_like(firing_partial_count)
                firing_count += firing_partial_count

                loss_etc = self.criterion(img, net_out, all_scores=True, force_firing=False, firing_cost_multiplier=1.0)
                losses_recon.append(loss_etc['recon_losses'])
                losses_perceptual.append(loss_etc['perceptual_losses'])

                count += firing.shape[0]

            training_firing_rates = (firing_count / count).flatten()

            # pass 2
            encoded_lengths = []
            encoded_diff_lengths = []
            for idx_batch, data_batch in enumerate(self.train_loader):
                self.net.using_heaviside = True
                self.net.adding_noise_in_test = False

                img = data_batch['img']
                img = img.cuda()

                net_out = self.net(img)

                firing = net_out["z"]

                for i in range(firing.shape[0]):
                    encoded = utils.compressor.encode(
                        firing[i, :].cpu().detach().numpy(),
                        training_firing_rates.cpu().detach().numpy()
                    )
                    encoded_lengths.append(len(encoded) * sys.getsizeof(encoded[0]))

                diffs = (img - net_out["x_hat"]).detach().cpu().numpy()
                for i in range(diffs.shape[0]):
                    encoded_diff_lengths.append(utils.compressor.encoded_length_residual((((np.clip(np.squeeze(np.transpose((diffs[i,:,:,:]-.5)*2.0, (1,2,0))), -3.0, 3.0))+3.0)/6.0*255).astype('uint8')))

            train_losses_recon = torch.cat(losses_recon, dim=0).cpu().detach().numpy()
            train_losses_perceptual = torch.cat(losses_perceptual, dim=0).cpu().detach().numpy()
            train_encoded_lengths = np.array(encoded_lengths)
            train_encoded_diff_lengths = np.array(encoded_diff_lengths)

            train_metafeatures = np.stack((train_losses_recon, train_losses_perceptual, train_encoded_lengths, train_encoded_diff_lengths), axis=1)
            oneclassmodel = make_pipeline(StandardScaler(), OneClassSVM())
            oneclassmodel.fit(train_metafeatures)

        # test

        test_imgs, test_imgs_hat, test_score_maps, test_names, test_labels, test_masks = [], [], [], [], [], []
        test_firing_rates = []
        test_real_firing_rates = []
        # v32: per-channel Heaviside firing counts (sigma(h) > 1/2, i.e. before noise) for the information budget
        fire_count_all, fire_count_normal, n_all, n_normal = None, None, 0, 0
        test_recon_losses = []
        test_perceptual_losses = []
        test_imgs_hat_for_compression = []
        test_imgs_diff_for_compression = []
        test_imgs_hat_for_LDP = []

        test_l2_score_maps = []

        # v31: metrics for the Heaviside (exactly 1 bit / channel) and noisy (LDP) inference modes
        test_score_maps_hv = []
        test_perceptual_losses_hv = []
        test_perceptual_losses_ldp = []
        # v32: LDP readout averaged over K independent noise draws (K = --ldp_samples)
        ldp_samples = int(self.opt.test.get('ldp_samples', 1))
        test_perceptual_losses_ldp_avg = []
        test_score_maps_ldp_avg = []

        test_repts = []
        test_repts_binary = []
        for idx_batch, data_batch in enumerate(self.test_loader):
            # v31: batched (test_batch_size); every per-sample quantity below is computed per sample
            img, label, name = data_batch['img'], data_batch['label'], data_batch['name']
            img = img.cuda(non_blocking=True)
            img.requires_grad = self.grad_flag  # Will be True for gradient-based methods

            # vanilla settings
            self.net.using_heaviside = False
            self.net.adding_noise_in_test = False

            net_out = self.net(img)

            test_firing_rates += net_out['firing_rate'].cpu().detach().numpy().tolist()
            test_real_firing_rates += net_out['real_firing_rate'].cpu().detach().numpy().tolist()

            fired = (net_out['unnoised_z'].detach() > 0.5).to(torch.float64)           # B x N
            is_normal = (label.view(-1) == 0).to(fired.device)
            if fire_count_all is None:
                fire_count_all = torch.zeros(fired.shape[1], dtype=torch.float64, device=fired.device)
                fire_count_normal = torch.zeros_like(fire_count_all)
            fire_count_all += fired.sum(dim=0)
            fire_count_normal += fired[is_normal].sum(dim=0)
            n_all += fired.shape[0]
            n_normal += int(is_normal.sum())

            lossset = self.criterion(img, net_out, all_scores=True, force_firing=False)
            test_score_maps.append(lossset['anomaly_score_maps'].cpu().detach())  # Nx1xHxW
            test_l2_score_maps.append(lossset['l2_anomaly_score_maps'].cpu().detach())  # Nx1xHxW

            test_recon_losses += lossset["recon_losses"].cpu().detach().numpy().tolist()
            test_perceptual_losses += lossset["perceptual_losses"].cpu().detach().numpy().tolist()

            test_labels.extend(label.view(-1).tolist())
            if self.pixel_metric:
                test_masks.append(data_batch['mask'])

            test_names.extend([[n] for n in name])      # keep the old [[name], ...] layout for visualize_2d
            test_imgs.append(img.cpu())
            test_imgs_hat.append(net_out['x_hat'].cpu())
            if full_eval:
                test_repts.append(net_out['z'].cpu().detach().numpy())

            # Heaviside inference (binary latent, <= 1 bit per channel)
            self.net.using_heaviside = True
            self.net.adding_noise_in_test = False
            net_out_hv = self.net(img)
            lossset_hv = self.criterion(img, net_out_hv, all_scores=True, force_firing=False)
            test_score_maps_hv.append(lossset_hv['anomaly_score_maps'].cpu().detach())
            test_perceptual_losses_hv += lossset_hv["perceptual_losses"].cpu().detach().numpy().tolist()
            test_imgs_hat_for_compression.append(net_out_hv['x_hat'].cpu())
            test_imgs_diff_for_compression.append(net_out_hv['x_hat'].cpu() - img.cpu())
            if full_eval:
                test_repts_binary.append(net_out_hv['z'].cpu().detach().numpy())

            # noisy (local differential privacy) inference
            self.net.using_heaviside = False
            self.net.adding_noise_in_test = True
            net_out_ldp = self.net(img)
            lossset_ldp = self.criterion(img, net_out_ldp, all_scores=True, force_firing=False)
            test_perceptual_losses_ldp += lossset_ldp["perceptual_losses"].cpu().detach().numpy().tolist()
            test_imgs_hat_for_LDP.append(net_out_ldp['x_hat'].cpu())

            if ldp_samples > 1:
                acc_perc = lossset_ldp["perceptual_losses"].detach().view(-1).clone()
                acc_map = lossset_ldp['anomaly_score_maps'].detach().clone()
                for _ in range(ldp_samples - 1):
                    net_out_k = self.net(img)                       # fresh Laplace noise in every QB layer
                    lossset_k = self.criterion(img, net_out_k, all_scores=True, force_firing=False)
                    acc_perc += lossset_k["perceptual_losses"].detach().view(-1)
                    acc_map += lossset_k['anomaly_score_maps'].detach()
                test_perceptual_losses_ldp_avg += (acc_perc / ldp_samples).cpu().numpy().tolist()
                test_score_maps_ldp_avg.append((acc_map / ldp_samples).cpu())

            self.net.using_heaviside = False
            self.net.adding_noise_in_test = False

        test_imgs = torch.cat(test_imgs, dim=0)
        test_imgs_hat = torch.cat(test_imgs_hat, dim=0)
        test_imgs_hat_for_compression = torch.cat(test_imgs_hat_for_compression, dim=0)
        test_imgs_hat_for_LDP = torch.cat(test_imgs_hat_for_LDP, dim=0)
        test_score_maps = torch.cat(test_score_maps, dim=0)  # Nx1xHxW
        test_score_maps = np.clip(test_score_maps, -1.0e+3, +1.0e+3)
        test_scores = torch.mean(test_score_maps, dim=[1, 2, 3]).cpu().detach().numpy()  # N
        test_l2_score_maps = torch.cat(test_l2_score_maps, dim=0)  # Nx1xHxW
        test_l2_scores = torch.mean(test_l2_score_maps, dim=[1, 2, 3]).cpu().detach().numpy()  # N
        test_l2_scores = np.clip(test_l2_scores, -1.0e+8, +1.0e+8)
        test_score_maps_hv = np.clip(torch.cat(test_score_maps_hv, dim=0), -1.0e+3, +1.0e+3)

        test_scores_firing = np.array(test_firing_rates) * self.firing_rate_cost_weight
        test_scores_real_firing = np.array(test_real_firing_rates) * self.firing_rate_cost_weight

        test_image_derived_losses = test_scores - test_scores_firing
        test_recon_losses = np.array(test_recon_losses)
        test_perceptual_losses = np.array(test_perceptual_losses)
        test_perceptual_losses_hv = np.array(test_perceptual_losses_hv)
        test_perceptual_losses_ldp = np.array(test_perceptual_losses_ldp)

        # image-level metrics
        test_labels = np.array(test_labels)

        results = {'AUC': metrics.roc_auc_score(test_labels, test_scores),
                   'AP': metrics.average_precision_score(test_labels, test_scores),
                   'AUC_firing': metrics.roc_auc_score(test_labels, test_scores_firing),
                   'AP_firing': metrics.average_precision_score(test_labels, test_scores_firing),
                   'AUC_real_firing': metrics.roc_auc_score(test_labels, test_scores_real_firing),
                   'AP_real_firing': metrics.average_precision_score(test_labels, test_scores_real_firing),
                   'AUC_image_derived': metrics.roc_auc_score(test_labels, test_image_derived_losses),
                   'AP_image_derived': metrics.average_precision_score(test_labels, test_image_derived_losses),
                   'AUC_perceptual': metrics.roc_auc_score(test_labels, test_perceptual_losses),
                   'AUC_perceptual_heaviside': metrics.roc_auc_score(test_labels, test_perceptual_losses_hv),
                   'AUC_perceptual_ldp': metrics.roc_auc_score(test_labels, test_perceptual_losses_ldp),
                   'AUC_recon': metrics.roc_auc_score(test_labels, test_recon_losses),
                   'AP_l2': metrics.average_precision_score(test_labels, test_l2_scores),
                   'AUC_l2': metrics.roc_auc_score(test_labels, test_l2_scores)
        }
        if ldp_samples > 1:
            test_perceptual_losses_ldp_avg = np.array(test_perceptual_losses_ldp_avg)
            results.update({'AUC_perceptual_ldp_avg': metrics.roc_auc_score(test_labels, test_perceptual_losses_ldp_avg),
                            'AP_perceptual_ldp_avg': metrics.average_precision_score(test_labels, test_perceptual_losses_ldp_avg)})
        # pixel-level metrics
        if self.pixel_metric:
            test_masks = torch.cat(test_masks, dim=0).unsqueeze(1)  # NxHxW -> Nx1xHxW
            masks_flat = test_masks.numpy().reshape(-1)
            pix_ap = metrics.average_precision_score(masks_flat, test_score_maps.cpu().numpy().reshape(-1))
            pix_auc = metrics.roc_auc_score(masks_flat, test_score_maps.cpu().numpy().reshape(-1))
            best_dice, best_thresh = compute_best_dice(test_score_maps.cpu().numpy(), test_masks.numpy())
            results.update({'PixAUC': pix_auc, 'PixAP': pix_ap, 'BestDice': best_dice, 'BestThresh': best_thresh})
            # l2-only (w/o log_var, firing rate)
            pix_ap_l2 = metrics.average_precision_score(masks_flat, test_l2_score_maps.cpu().numpy().reshape(-1))
            pix_auc_l2 = metrics.roc_auc_score(masks_flat, test_l2_score_maps.cpu().numpy().reshape(-1))
            best_dice_l2, best_thresh_l2 = compute_best_dice(test_l2_score_maps.cpu().numpy(), test_masks.numpy())
            results.update({'PixAUC_l2': pix_auc_l2, 'PixAP_l2': pix_ap_l2, 'BestDice_l2': best_dice_l2, 'BestThresh_l2': best_thresh_l2})
            # Heaviside inference
            pix_ap_hv = metrics.average_precision_score(masks_flat, test_score_maps_hv.cpu().numpy().reshape(-1))
            best_dice_hv, _ = compute_best_dice(test_score_maps_hv.cpu().numpy(), test_masks.numpy())
            results.update({'PixAP_heaviside': pix_ap_hv, 'BestDice_heaviside': best_dice_hv})
            if ldp_samples > 1:
                maps_ldp = np.clip(torch.cat(test_score_maps_ldp_avg, dim=0), -1.0e+3, +1.0e+3).cpu().numpy()
                results.update({'PixAP_ldp_avg': metrics.average_precision_score(masks_flat, maps_ldp.reshape(-1)),
                                'BestDice_ldp_avg': compute_best_dice(maps_ldp, test_masks.numpy())[0]})
        else:
            test_masks = None

        # others
        # v32: Heaviside information budget on the test set (Appendix A): I(X; X_hat) <= H(Z) <= sum_i h2(p_i) <= N h2(p_bar)
        results.update(self.heaviside_budget(fire_count_all, n_all, prefix=""))
        if n_normal > 0:
            results.update(self.heaviside_budget(fire_count_normal, n_normal, prefix="normal_"))

        results.update({"normal_score": np.mean(test_scores[np.where(test_labels == 0)]),
                        "abnormal_score": np.mean(test_scores[np.where(test_labels == 1)])})

        # reconstruction results (first 4 and last 4 test images)
        test_imgs_ = torch.cat((test_imgs[0:4], test_imgs[-5:-1]), dim=0)
        test_imgs_hat_ = torch.cat((test_imgs_hat[0:4], test_imgs_hat[-5:-1]), dim=0)
        test_imgs_hat_for_compression_ = torch.cat((test_imgs_hat_for_compression[0:4],
                                                    test_imgs_hat_for_compression[-5:-1]), dim=0)
        test_imgs_hat_for_LDP_ = torch.cat((test_imgs_hat_for_LDP[0:4], test_imgs_hat_for_LDP[-5:-1]), dim=0)

        if self.pixel_metric:
            test_imgs_abnormal_score_map_ = torch.cat((test_score_maps[0:4,:,:,:], test_score_maps[-5:-1,:,:,:]), dim=0)
            test_imgs_mask_ = torch.cat((test_masks[0:4,:,:,:], test_masks[-5:-1,:,:,:]), dim=0)
            if test_imgs_.shape[1] == 3:
                test_imgs_abnormal_score_map_ = test_imgs_abnormal_score_map_.repeat(1,3,1,1,1)
                test_imgs_mask_ = test_imgs_mask_.repeat(1,3,1,1,1)
            img = torch.stack((test_imgs_, test_imgs_hat_, test_imgs_-test_imgs_hat_, test_imgs_abnormal_score_map_, test_imgs_mask_, test_imgs_hat_for_compression_, test_imgs_hat_for_LDP_), dim=4)
        else:
            img = torch.stack((test_imgs_, test_imgs_hat_, test_imgs_-test_imgs_hat_, test_imgs_hat_for_compression_, test_imgs_hat_for_LDP_), dim=4)
        img = torch.permute(img, (4,2,0,3,1))
        img = img.reshape((img.shape[0]*img.shape[1], img.shape[2]*img.shape[3], img.shape[4]))
        if(img.shape[2] == 1):
            img = img.reshape((img.shape[0], img.shape[1]))
        img = torch.nan_to_num(img, nan=0.0, posinf=+10.0, neginf=-10.0)
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img)) * 255.0
        img = img.to(torch.uint8)
        plt.imsave(os.path.join(self.opt.train['save_dir'], f'imgs_Ep{epoch}.png'), img.numpy(), cmap='gray')
        if self.logger is not None:
            if(len(img.shape) == 2):
                self.logger.log(step=epoch, data={f'imgs/Ep{epoch}': wandb.Image(img[:,:,np.newaxis].numpy(), caption=f'imgs_Ep{epoch}', mode="L")})
            else:
                assert(img.shape[2] == 3)
                self.logger.log(step=epoch, data={f'imgs/Ep{epoch}': wandb.Image(img.numpy(), caption=f'imgs_Ep{epoch}', mode="RGB")})

        if full_eval:
            test_repts = np.concatenate(test_repts, axis=0)  # Nxd
            test_repts_binary = np.concatenate(test_repts_binary, axis=0)  # Nxd

            # latent expression compression with arithmetic coding
            encoded_length = []
            for i in range(test_repts_binary.shape[0]):
                encoded = utils.compressor.encode(test_repts_binary[i, :], training_firing_rates.cpu().detach().numpy())
                encoded_length.append(len(encoded)* sys.getsizeof(encoded[0]))
            encoded_length = np.array(encoded_length)

            # ... and png compression of residual information (diff)
            diffs = torch.cat(test_imgs_diff_for_compression, dim=0).detach().cpu().numpy() # NxCxHxW
            encoded_diff_length = []
            for i in range(diffs.shape[0]):
                encoded_diff_length.append(utils.compressor.encoded_length_residual((((np.clip(np.squeeze(np.transpose((diffs[i,:,:,:]-.5)*2.0, (1,2,0))), -3.0, 3.0))+3.0)/6.0*255).astype('uint8')))
            encoded_diff_length = np.array(encoded_diff_length)
            total_encoded_length = encoded_length + encoded_diff_length

            results.update({'auc_encoded_length': metrics.roc_auc_score(test_labels, encoded_length),
                            'ap_encoded_length': metrics.average_precision_score(test_labels, encoded_length),
                            'auc_png_encoded_length': metrics.roc_auc_score(test_labels, encoded_diff_length),
                            'ap_png_encoded_length': metrics.average_precision_score(test_labels, encoded_diff_length),
                            'auc_total_encoded_length': metrics.roc_auc_score(test_labels, total_encoded_length),
                            'ap_total_encoded_length': metrics.average_precision_score(test_labels, total_encoded_length),
                            'average_range_encoded_length': np.mean(encoded_length),
                            'average_png_encoded_length': np.mean(encoded_diff_length)})

            # one-class / few-shot classifiers on meta-features (NOTE: the few-shot tester uses test labels)
            test_metafeatures = np.stack((test_recon_losses, test_perceptual_losses, encoded_length, encoded_diff_length), axis=1)
            np.save(os.path.join(self.opt.train['save_dir'], 'train_metafeatures.npy'), train_metafeatures)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_metafeatures.npy'), test_metafeatures)
            if self.fsct is None:
                self.fsct = utils.fewshot_classifiers.FewshotClassifierTester(20, np.zeros(train_metafeatures.shape[0]), test_labels, 42)
            best_avg_rank, peeked_best_score, best_model_desc, test_score_with_the_best = self.fsct.do_validation(train_metafeatures, test_metafeatures, f"{epoch=}_")
            results.update({'best_avg_rank': best_avg_rank, 'auc_best_fewshot': test_score_with_the_best,
                            'auc_best_peeked': peeked_best_score, 'best_model_desc': best_model_desc})

            # rept tsne
            test_tsne = TSNE(n_components=2).fit_transform(test_repts)  # Nx2
            normal_tsne = test_tsne[np.where(test_labels == 0)]
            abnormal_tsne = test_tsne[np.where(test_labels == 1)]
            plt.rcParams.update({'font.size': 14})
            plt.scatter(normal_tsne[:, 0], normal_tsne[:, 1], color='b', label="Normal", s=2)
            plt.scatter(abnormal_tsne[:, 0], abnormal_tsne[:, 1], color='r', label="Abnormal", s=2)
            plt.xticks([])
            plt.yticks([])
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.opt.train['save_dir'], f'tsne_Ep{epoch}.pdf'))
            plt.close()

        if self.opt.test['save_flag']:
            self.visualize_2d(test_imgs, test_imgs_hat, test_score_maps, test_names, test_labels, test_masks)

            np.save(os.path.join(self.opt.train['save_dir'], 'test_labels.npy'), test_labels)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_scores_firing.npy'), test_scores_firing)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_scores_real_firing.npy'), test_scores_real_firing)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_perceptual_losses.npy'), test_perceptual_losses)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_perceptual_losses_heaviside.npy'), test_perceptual_losses_hv)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_recon_losses.npy'), test_recon_losses)
            if full_eval:
                np.save(os.path.join(self.opt.train['save_dir'], 'test_repts.npy'), test_repts)
                np.save(os.path.join(self.opt.train['save_dir'], 'encoded_length.npy'), encoded_length)

        self.enable_network_grad()
        return results

    def run_train(self):
        num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        t0 = time.time()

        for epoch in range(1, num_epochs + 1):

            firing_cost_multiplier = 0.0 if epoch<100.0 else 1.0 # np.minimum(epoch/100, 1.0)
            shortcut_multiplier = 1.0 # 0.0 if epoch<100.0 else 1.0

            train_loss, loss_recon, loss_logvar, loss_firing, loss_perceptual, firing_rate, real_firing_rate = \
                self.train_epoch(force_firing=True, firing_cost_multiplier=firing_cost_multiplier, shortcut_multiplier=shortcut_multiplier, noise_level = self.opt.train['noise_level'], epoch=epoch)
#            train_loss, loss_recon, loss_logvar, loss_firing, loss_perceptual, firing_rate, real_firing_rate = \
#                self.train_epoch(force_firing=True)
#            train_loss, loss_recon, loss_logvar, loss_firing, loss_perceptual, firing_rate, real_firing_rate = \
#                self.train_epoch(force_firing=False)
#            train_loss, loss_recon, loss_logvar, loss_firing, loss_perceptual, firing_rate, real_firing_rate = \
#                self.train_epoch(force_firing = (epoch < 100))

            self.logger.log(step=epoch, data={
                "train/loss": train_loss
                , "train/loss_recon": loss_recon
                , "train/loss_logvar": loss_logvar
                , "train/loss_firing": loss_firing
                , "train/loss_perceptual": loss_perceptual
                , "train/firing_rate": firing_rate
                , "train/real_firing_rate": real_firing_rate
            })
            # self.logger.log(step=epoch, data={"train/loss": train_loss, "train/lr": self.scheduler.get_last_lr()[0]})
            # self.scheduler.step()

            if epoch == 1 or epoch % self.opt.train['eval_freq'] == 0:
                eval_results = self.evaluate(epoch)

                t = time.time() - t0
                print("Epoch[{:3d}/{:3d}]  Time:{:.1f}s  loss:{:.5f}".format(epoch, num_epochs, t, train_loss),
                      end="  |  ")

                keys = list(eval_results.keys())
                for key in keys:
                    if(isinstance(eval_results[key], str)):
                        print(key+" : "+eval_results[key], end="  ")
                    else:
                        print(key+": {:.5f}".format(eval_results[key]), end="  ")
                    eval_results["val/"+key] = eval_results.pop(key)
                print()

                self.logger.log(step=epoch, data=eval_results)
                t0 = time.time()

        self.save_checkpoint()
        self.logger.finish()
