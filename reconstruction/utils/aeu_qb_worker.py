import time
import torch
import os
from sklearn import metrics
from utils.util import compute_best_dice
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.ae_worker import AEWorker
from utils.aeu_worker import AEUWorker
from utils.util import AverageMeter

class AEU_QBWorker(AEUWorker):
    def __init__(self, opt):
        super(AEU_QBWorker, self).__init__(opt)
        self.pixel_metric = True if self.opt.dataset == "brats" else False
        self.firing_rate_cost_weight = self.opt.model['firing_rate_cost_weight']

    def train_epoch(self, force_firing=False, firing_cost_multiplier=1.0):
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
            img = img.cuda()

            net_out = self.net(img)

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

        test_imgs, test_imgs_hat, test_scores, test_score_maps, test_names, test_labels, test_masks = \
            [], [], [], [], [], [], []
        test_firing_rates = []
        test_real_firing_rates = []
        test_recon_losses = []
        test_perceptual_losses = []
        test_firing_rates = []
        test_real_firing_rates = []
        
        test_repts = []
        # with torch.no_grad():
        for idx_batch, data_batch in enumerate(self.test_loader):
            # test batch_size=1
            img, label, name = data_batch['img'], data_batch['label'], data_batch['name']
            img = img.cuda()
            img.requires_grad = self.grad_flag  # Will be True for gradient-based methods

            net_out = self.net(img)

            test_firing_rate = net_out['firing_rate']
            test_real_firing_rate = net_out['real_firing_rate']
            test_firing_rates += test_firing_rate.cpu().detach().numpy().tolist()
            test_real_firing_rates += test_real_firing_rate.cpu().detach().numpy().tolist()

            # anomaly_score_map = self.criterion(img, net_out, anomaly_score=True, keepdim=True).detach().cpu()
            lossset = self.criterion(img, net_out, all_scores=True, force_firing=False)
            anomaly_score_map = lossset['anomaly_score_maps'].cpu().detach()  # Nx1xHxW
            test_score_maps.append(anomaly_score_map)

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

        test_score_maps = torch.cat(test_score_maps, dim=0)  # Nx1xHxW
        test_scores = torch.mean(test_score_maps, dim=[1, 2, 3]).cpu().detach().numpy()  # N

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
        auc_firing = metrics.roc_auc_score(test_labels, test_scores_firing)
        auc_image_derived = metrics.roc_auc_score(test_labels, test_image_derived_losses)
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
                    'AUC_recon': auc_recon
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
        else:
            test_masks = None

        # others
        test_normal_score = np.mean(test_scores[np.where(test_labels == 0)])
        test_abnormal_score = np.mean(test_scores[np.where(test_labels == 1)])
        results.update({"normal_score": test_normal_score, "abnormal_score": test_abnormal_score})

        # latent represenations
        test_repts = np.concatenate(test_repts, axis=0)  # Nxd
        plt.imsave(os.path.join(self.opt.train['save_dir'], f'repts_Ep{epoch}.png'), test_repts[:,:])

        # reconstruction results
        test_imgs_first = torch.cat(test_imgs, dim=0)[0:4,:,:,:]
        test_imgs_first_hat = torch.cat(test_imgs_hat, dim=0)[0:4,:,:,:]
        test_imgs_last = torch.cat(test_imgs, dim=0)[-5:-1,:,:,:]
        test_imgs_last_hat = torch.cat(test_imgs_hat, dim=0)[-5:-1,:,:,:]
        test_imgs_ = torch.cat((test_imgs_first, test_imgs_last), dim=0)
        test_imgs_hat_ = torch.cat((test_imgs_first_hat, test_imgs_last_hat), dim=0)
        img = torch.stack((test_imgs_, test_imgs_hat_, test_imgs_-test_imgs_hat_), dim=4)
        img = torch.permute(img, (4,2,0,3,1))
        img = img.reshape((img.shape[0]*img.shape[1], img.shape[2]*img.shape[3], img.shape[4]))
        if(img.shape[2] == 1):
            img = img.reshape((img.shape[0], img.shape[1]))
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        plt.imsave(os.path.join(self.opt.train['save_dir'], f'imgs_Ep{epoch}.png'), img, cmap='gray')

        # rept vsne
        test_tsne = TSNE(n_components=2).fit_transform(test_repts)  # Nx2
        normal_tsne = test_tsne[np.where(test_labels == 0)]
        abnormal_tsne = test_tsne[np.where(test_labels == 1)]
        plt.rcParams['font.family'] = 'Times New Roman'
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

        self.enable_network_grad()
        return results

    def run_train(self):
        num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        t0 = time.time()
        for epoch in range(1, num_epochs + 1):

#            train_loss, loss_recon, loss_logvar, loss_firing, loss_perceptual, firing_rate, real_firing_rate = \
#                self.train_epoch(force_firing=False, firing_cost_multiplier=np.minimum(1.0, epoch/100.0))
            train_loss, loss_recon, loss_logvar, loss_firing, loss_perceptual, firing_rate, real_firing_rate = \
                self.train_epoch(force_firing=True)

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
