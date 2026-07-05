import time
import torch
import os
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

    def train_epoch(self, force_firing=False, firing_cost_multiplier=1.0, shortcut_multiplier=1.0, noise_level = 0.0, epoch=0):
        self.net.train()
        losses = AverageMeter()
        losses_recon = AverageMeter()
        losses_logvar = AverageMeter()
        losses_firing = AverageMeter()
        losses_perceptual = AverageMeter()
        firing_rates = AverageMeter()
        real_firing_rates = AverageMeter()
        
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img_noised = img.clone()

            img = img.cuda()

            if 0 < noise_level:
                img_noised += make_noise_like(img, noise_level)
                img_noised = img_noised.cuda()

            net_out = self.net(img_noised, shortcut_multiplier=shortcut_multiplier)

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

            firing_rates.update(net_out["firing_rate"].mean(), img.size(0))
            real_firing_rates.update(net_out["real_firing_rate"].mean(), img.size(0))
            loss_etc = self.criterion(img, net_out, force_firing=force_firing, firing_cost_multiplier=firing_cost_multiplier)
            loss = loss_etc['loss']
            losses_recon.update(loss_etc['recon_loss'].mean(), img.size(0))
            losses_logvar.update(loss_etc['log_var'].mean(), img.size(0))
            losses_firing.update(loss_etc['firing_loss'].mean(), img.size(0))
            if 'perceptual_loss' in loss_etc:
                losses_perceptual.update(loss_etc['perceptual_loss'].mean(), img.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), img.size(0))

        print("expected_firing_rate: {:.4f}, real_firing_rate: {:,.4f}, loss_recon: {:.4f}, loss_firing: {:.4f}, loss_perceptual: {:.4f}".format(
                firing_rates.avg, 
                real_firing_rates.avg,
                losses_recon.avg, 
                losses_firing.avg,
                losses_perceptual.avg
        ))
        return losses.avg, losses_recon.avg, losses_logvar.avg, losses_firing.avg, losses_perceptual.avg, firing_rates.avg, real_firing_rates.avg


    def evaluate(self, epoch='test'):
        self.net.eval()
        self.close_network_grad()

        # calculate training_firing_rates from training dataset

        # pass 1
        firing_count = None
        count = 0
        losses = []
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

            count += 1

        training_firing_rates = (firing_count / count).flatten()

        # pass 2
        encoded_lengths = []
        for idx_batch, data_batch in enumerate(self.train_loader):
            # binary latent
            self.net.using_heaviside = True
            self.net.adding_noise_in_test = False

            img = data_batch['img']
            img = img.cuda()

            net_out = self.net(img)

            firing = net_out["z"]

            # calculate lengths
            for i in range(firing.shape[0]):
                encoded = utils.compressor.encode(
                    firing[i, :].cpu().detach().numpy(), 
                    training_firing_rates.cpu().detach().numpy()
                )
                encoded_lengths.append(len(encoded))

        # make list
        train_losses_recon = torch.cat(losses_recon, dim=0).cpu().detach().numpy()
        train_losses_perceptual = torch.cat(losses_perceptual, dim=0).cpu().detach().numpy()
        train_encoded_lengths = np.array(encoded_lengths)



        # build an one-class SVM
        train_metafeatures = np.stack((train_losses_recon, train_losses_perceptual, train_encoded_lengths), axis=1)

        oneclassmodel = make_pipeline(StandardScaler(), OneClassSVM())

        oneclassmodel.fit(train_metafeatures)



        # test

        test_imgs, test_imgs_hat, test_scores, test_score_maps, test_names, test_labels, test_masks = \
            [], [], [], [], [], [], []
        test_firing_rates = []
        test_real_firing_rates = []
        test_recon_losses = []
        test_perceptual_losses = []
        test_firing_rates = []
        test_real_firing_rates = []
        test_imgs_hat_for_compression = []
        test_imgs_hat_for_LDP = []

        test_l2_score_maps = []
        test_l2_scores = []
        
        test_repts = []
        test_repts_binary = []
        # with torch.no_grad():
        for idx_batch, data_batch in enumerate(self.test_loader):
            # test batch_size=1
            img, label, name = data_batch['img'], data_batch['label'], data_batch['name']
            img = img.cuda()
            img.requires_grad = self.grad_flag  # Will be True for gradient-based methods
            
            # vanilla settings
            self.net.using_heaviside = False
            self.net.adding_noise_in_test = False
            
            net_out = self.net(img)

            test_firing_rate = net_out['firing_rate']
            test_real_firing_rate = net_out['real_firing_rate']
            test_firing_rates += test_firing_rate.cpu().detach().numpy().tolist()
            test_real_firing_rates += test_real_firing_rate.cpu().detach().numpy().tolist()

            # anomaly_score_map = self.criterion(img, net_out, anomaly_score=True, keepdim=True).detach().cpu()
            lossset = self.criterion(img, net_out, all_scores=True, force_firing=False)
            anomaly_score_map = lossset['anomaly_score_maps'].cpu().detach()  # Nx1xHxW
            test_score_maps.append(anomaly_score_map)
            l2_anomaly_score_map = lossset['l2_anomaly_score_maps'].cpu().detach()  # Nx1xHxW
            test_l2_score_maps.append(l2_anomaly_score_map)

            test_recon_losses += lossset["recon_losses"].cpu().detach().numpy().tolist()
            test_perceptual_losses += lossset["perceptual_losses"].cpu().detach().numpy().tolist()

            test_labels.append(label.item())
            if self.pixel_metric:
                mask = data_batch['mask']
                test_masks.append(mask)

            if 1: # self.opt.test['save_flag']:
                img_hat = net_out['x_hat']
                test_names.append(name)
                test_imgs.append(img.cpu())
                test_imgs_hat.append(img_hat.cpu())
                
                z = net_out['z']
                test_repts.append(z.cpu().detach().numpy())
            
            if 1:
                # outputs for image compression
                self.net.using_heaviside = True
                self.net.adding_noise_in_test = False
                
                net_out_for_compression = self.net(img)

                test_imgs_hat_for_compression.append(net_out_for_compression['x_hat'].cpu())
                test_repts_binary.append(net_out_for_compression['z'].cpu().detach().numpy())
                
                # outputs for local differential privacy output
                self.net.using_heaviside = False
                self.net.adding_noise_in_test = True
                
                net_out_for_LDP = self.net(img)

                test_imgs_hat_for_LDP.append(net_out_for_LDP['x_hat'].cpu())
                
                self.net.using_heaviside = False
                self.net.adding_noise_in_test = False



        test_score_maps = torch.cat(test_score_maps, dim=0)  # Nx1xHxW
        test_scores = torch.mean(test_score_maps, dim=[1, 2, 3]).cpu().detach().numpy()  # N

        test_l2_score_maps = torch.cat(test_l2_score_maps, dim=0)  # Nx1xHxW
        test_l2_scores = torch.mean(test_l2_score_maps, dim=[1, 2, 3]).cpu().detach().numpy()  # N

        test_scores_firing = np.array(test_firing_rates) * self.firing_rate_cost_weight
        test_scores_real_firing = np.array(test_real_firing_rates) * self.firing_rate_cost_weight

        test_image_derived_losses = test_scores - test_scores_firing
        test_recon_losses = np.array(test_recon_losses)
        test_perceptual_losses = np.array(test_perceptual_losses)

        # image-level metrics
        test_labels = np.array(test_labels)
        auc = metrics.roc_auc_score(test_labels, test_scores)
        ap = metrics.average_precision_score(test_labels, test_scores)
        ap_firing = metrics.average_precision_score(test_labels, test_scores_firing)
        ap_real_firing = metrics.average_precision_score(test_labels, test_scores_real_firing)
        ap_image_derived = metrics.average_precision_score(test_labels, test_image_derived_losses)
        ap_l2 = metrics.average_precision_score(test_labels, test_l2_scores)
        auc_firing = metrics.roc_auc_score(test_labels, test_scores_firing)
        auc_image_derived = metrics.roc_auc_score(test_labels, test_image_derived_losses)
        auc_l2 = metrics.roc_auc_score(test_labels, test_l2_scores)
        auc_real_firing = metrics.roc_auc_score(test_labels, test_scores_real_firing)
        auc_perceptual = metrics.roc_auc_score(test_labels, test_perceptual_losses)
        auc_recon = metrics.roc_auc_score(test_labels, test_recon_losses)
        results = {'AUC': auc, 
                    'AP': ap, 
                    'AUC_firing': auc_firing, 
                    'AP_firing': ap_firing, 
                    'AUC_real_firing': auc_real_firing,
                    'AP_real_firing': ap_real_firing,
                    'AUC_image_derived': auc_image_derived,
                    'AP_image_derived': ap_image_derived,
                    'AUC_perceptual': auc_perceptual,
                    'AUC_recon': auc_recon,
                    'AP_l2': ap_l2,
                    'AUC_l2': auc_l2
        }
        # pixel-level metrics
        if self.pixel_metric:
            test_masks = torch.cat(test_masks, dim=0).unsqueeze(1)  # NxHxW -> Nx1xHxW
            pix_ap = metrics.average_precision_score(test_masks.numpy().reshape(-1),
                                                     test_score_maps.cpu().numpy().reshape(-1))
            pix_auc = metrics.roc_auc_score(test_masks.numpy().reshape(-1),
                                            test_score_maps.cpu().numpy().reshape(-1))
            best_dice, best_thresh = compute_best_dice(test_score_maps.cpu().numpy(), test_masks.numpy())
            results.update({'PixAUC': pix_auc, 'PixAP': pix_ap, 'BestDice': best_dice, 'BestThresh': best_thresh})
            # l2-only (w/o log_var, firing rate)
            pix_ap_l2 = metrics.average_precision_score(test_masks.numpy().reshape(-1),
                                                     test_l2_score_maps.cpu().numpy().reshape(-1))
            pix_auc_l2 = metrics.roc_auc_score(test_masks.numpy().reshape(-1),
                                            test_l2_score_maps.cpu().numpy().reshape(-1))
            best_dice_l2, best_thresh_l2 = compute_best_dice(test_l2_score_maps.cpu().numpy(), test_masks.numpy())
            results.update({'PixAUC_l2': pix_auc_l2, 'PixAP_l2': pix_ap_l2, 'BestDice_l2': best_dice_l2, 'BestThresh_l2': best_thresh_l2})
        else:
            test_masks = None

        # others
        test_normal_score = np.mean(test_scores[np.where(test_labels == 0)])
        test_abnormal_score = np.mean(test_scores[np.where(test_labels == 1)])
        results.update({"normal_score": test_normal_score, "abnormal_score": test_abnormal_score})

        # latent representaions
        test_repts = np.concatenate(test_repts, axis=0)  # Nxd
        plt.imsave(os.path.join(self.opt.train['save_dir'], f'repts_Ep{epoch}.png'), test_repts[:,:])
        #if self.logger is not None:
        #    repts_img = np.stack((
        #        np.clip(test_repts[:,:]*2-1.0, 0., 1.), 
        #        np.clip(test_repts[:,:]*2-0.5, 0., 1.), 
        #        np.clip(test_repts[:,:]*2-0.0, 0., 1.)
        #    ), axis=2)
        #    self.logger.log(step=epoch, data={f'repts/Ep{epoch}': wandb.Image(repts_img, caption=f'repts_Ep{epoch}', mode='RGB')})

        # reconstruction results
        test_imgs_first = torch.cat(test_imgs, dim=0)[0:4,:,:,:]
        test_imgs_last = torch.cat(test_imgs, dim=0)[-5:-1,:,:,:]
        test_imgs_ = torch.cat((test_imgs_first, test_imgs_last), dim=0)

        test_imgs_first_hat = torch.cat(test_imgs_hat, dim=0)[0:4,:,:,:]
        test_imgs_last_hat = torch.cat(test_imgs_hat, dim=0)[-5:-1,:,:,:]
        test_imgs_hat_ = torch.cat((test_imgs_first_hat, test_imgs_last_hat), dim=0)

        if self.pixel_metric:
            test_imgs_first_abnormal_score_map = test_score_maps[0:4,:,:,:]
            test_imgs_last_abnormal_score_map = test_score_maps[-5:-1,:,:,:]
            test_imgs_abnormal_score_map_ = torch.cat((test_imgs_first_abnormal_score_map, test_imgs_last_abnormal_score_map), dim=0)

            test_imgs_first_mask = test_masks[0:4,:,:,:]
            test_imgs_last_mask = test_masks[-5:-1,:,:,:]
            test_imgs_mask_ = torch.cat((test_imgs_first_mask, test_imgs_last_mask), dim=0)

            if test_imgs_.shape[1] == 3:
                # color
                test_imgs_abnormal_score_map_ = test_imgs_abnormal_score_map_.repeat(1,3,1,1,1)
                test_imgs_mask_ = test_imgs_mask_.repeat(1,3,1,1,1)

        if 1:
            test_imgs_first_hat_for_compression =  torch.cat(test_imgs_hat_for_compression, dim=0)[0:4,:,:,:]
            test_imgs_last_hat_for_compression =  torch.cat(test_imgs_hat_for_compression, dim=0)[-5:-1,:,:,:]
            test_imgs_hat_for_compression_ = torch.cat((test_imgs_first_hat_for_compression, test_imgs_last_hat_for_compression), dim=0)

            test_imgs_first_hat_for_LDP =  torch.cat(test_imgs_hat_for_LDP, dim=0)[0:4,:,:,:]
            test_imgs_last_hat_for_LDP =  torch.cat(test_imgs_hat_for_LDP, dim=0)[-5:-1,:,:,:]
            test_imgs_hat_for_LDP_ = torch.cat((test_imgs_first_hat_for_LDP, test_imgs_last_hat_for_LDP), dim=0)

        if self.pixel_metric:
            img = torch.stack((test_imgs_, test_imgs_hat_, test_imgs_-test_imgs_hat_, test_imgs_abnormal_score_map_, test_imgs_mask_, test_imgs_hat_for_compression_, test_imgs_hat_for_LDP_), dim=4)
        else:
            img = torch.stack((test_imgs_, test_imgs_hat_, test_imgs_-test_imgs_hat_, test_imgs_hat_for_compression_, test_imgs_hat_for_LDP_), dim=4)
        img = torch.permute(img, (4,2,0,3,1))
        img = img.reshape((img.shape[0]*img.shape[1], img.shape[2]*img.shape[3], img.shape[4]))
        if(img.shape[2] == 1):
            img = img.reshape((img.shape[0], img.shape[1]))
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        plt.imsave(os.path.join(self.opt.train['save_dir'], f'imgs_Ep{epoch}.png'), img, cmap='gray')
        if self.logger is not None:
            if(len(img.shape) == 2):
                self.logger.log(step=epoch, data={f'imgs/Ep{epoch}': wandb.Image(img.T[:,:,np.newaxis], caption=f'imgs_Ep{epoch}', mode="L")})
            else:
                assert(img.shape[2] == 3)
                self.logger.log(step=epoch, data={f'imgs/Ep{epoch}': wandb.Image(img.permute((2,1,0)), caption=f'imgs_Ep{epoch}', mode="RGB")})

        test_repts_binary = np.concatenate(test_repts_binary, axis=0)  # Nxd
            
        # latent expression compression with arithmetic coding
        encoded_length = []
        for i in range(test_repts_binary.shape[0]):
            encoded = utils.compressor.encode(test_repts_binary[i, :], training_firing_rates.cpu().detach().numpy())
            encoded_length.append(len(encoded))
        
        encoded_length = np.array(encoded_length)
        
        auc_encoded_length = metrics.roc_auc_score(test_labels, encoded_length)
        ap_encoded_length = metrics.average_precision_score(test_labels, encoded_length)

        # seek the best model
        test_metafeatures = np.stack((test_recon_losses, test_perceptual_losses, encoded_length), axis=1)

        np.save(os.path.join(self.opt.train['save_dir'], 'train_metafeatures.npy'), train_metafeatures)
        np.save(os.path.join(self.opt.train['save_dir'], 'test_metafeatures.npy'), test_metafeatures)
        
        auc_fewshot, method_fewshot = utils.fewshot_classifiers.seek_best_classifier(train_metafeatures, test_metafeatures, test_labels)

        results.update({'auc_fewshot': auc_fewshot, 'method_fewshot': method_fewshot, 'auc_encoded_length': auc_encoded_length, 'ap_encoded_length': ap_encoded_length})



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
        # plt.title(self.opt.data_name[self.opt.dataset] + ' | OC-SVM Perf. 0.66/0.82')
        # plt.title('OC-SVM Perf. 0.48/0.52')
        plt.tight_layout()
        plt.savefig(os.path.join(self.opt.train['save_dir'], f'tsne_Ep{epoch}.pdf'))
        plt.close()

        if self.opt.test['save_flag']:
            test_imgs = torch.cat(test_imgs, dim=0)
            test_imgs_hat = torch.cat(test_imgs_hat, dim=0)
            self.visualize_2d(test_imgs, test_imgs_hat, test_score_maps, test_names, test_labels, test_masks)

            np.save(os.path.join(self.opt.train['save_dir'], 'test_labels.npy'), test_labels)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_repts.npy'), test_repts)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_scores_firing.npy'), test_scores_firing)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_scores_real_firing.npy'), test_scores_real_firing)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_perceptual_losses.npy'), test_perceptual_losses)
            np.save(os.path.join(self.opt.train['save_dir'], 'test_recon_losses.npy'), test_recon_losses)
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
                    print(key+": {:.5f}".format(eval_results[key]), end="  ")
                    eval_results["val/"+key] = eval_results.pop(key)
                print()

                self.logger.log(step=epoch, data=eval_results)
                t0 = time.time()

        self.save_checkpoint()
        self.logger.finish()
