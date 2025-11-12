import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import nibabel as nib

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.models.losses.custom_losses import EdgeLoss

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

def combine_coils_rss(coil_tensor):
    # Assume coil_tensor is a numpy array of shape (32, H, W)
    combined = np.sqrt(np.sum(np.square(coil_tensor), axis=0))
    # Normalize to [0, 1] if necessary
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    return combined

def psd_loss(residual, noise_map):
    """
    Compute the power spectral density (PSD) loss between the residual and noise map.
    
    Args:
        residual (Tensor): Estimated noise (shape: [B, C, H, W]).
        noise_map (Tensor): Ground truth noise map (shape: [B, C, H, W]).
        
    Returns:
        loss (Tensor): The L1 loss between the PSDs.
    """
    # Compute the FFT
    res_fft = torch.fft.fft2(residual)
    noise_fft = torch.fft.fft2(noise_map)
    
    # Compute the power spectral densities (magnitude squared)
    res_psd = torch.abs(res_fft) ** 2
    noise_psd = torch.abs(noise_fft) ** 2
    
    # Compute the L1 loss between the PSDs
    loss = F.l1_loss(res_psd, noise_psd)
    return loss

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
            

            # ---------------- Edge loss -----------------
            edge_cfg = train_opt.get('edge_opt')
            if edge_cfg:
                self.cri_edge = EdgeLoss().to(self.device)
                self.edge_weight = edge_cfg.get('weight', 0.05)
            else:
                self.cri_edge = None

            # ---------------- Perceptual loss -----------
            perc_cfg = train_opt.get('perc_opt')
            if perc_cfg:
                # pull out the weight so it doesn’t get passed to __init__
                perc_weight = perc_cfg.pop('weight', 0.01)
                # pull out the class name
                perc_type   = perc_cfg.pop('type')
                perc_cls    = getattr(loss_module, perc_type)
                self.cri_perc   = perc_cls(**perc_cfg).to(self.device)
                self.perc_weight = perc_weight
                print("percept weight: ", self.perc_weight)
            else:
                self.cri_perc = None

            # -------- Total-variation weight ------------
            self.tv_weight = train_opt.get('tv_weight', 0.0)


        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]

        # ---------- Pixel loss ----------
        l_total = 0.0
        l_pix   = sum(self.cri_pix(p, self.gt) for p in preds)
        l_total += l_pix
        loss_dict = OrderedDict(l_pix=l_pix)

        # ---------- Edge loss ----------
        if self.cri_edge is not None:
            l_edge = self.edge_weight * self.cri_edge(self.output, self.gt)
            l_total += l_edge
            loss_dict['l_edge'] = l_edge

        # ---------- Perceptual loss -----
        if self.cri_perc is not None:
            l_perc = self.perc_weight * self.cri_perc(self.output, self.gt)
            l_total += l_perc
            loss_dict['l_perc'] = l_perc

        # ---------- TV on residual ------
        if self.tv_weight > 0:
            residual = self.output - self.lq
            tv = (residual[:, :, :-1] - residual[:, :, 1:]).abs().mean() + \
                (residual[:, :, :, :-1] - residual[:, :, :, 1:]).abs().mean()
            l_tv = self.tv_weight * tv
            l_total += l_tv
            loss_dict['l_tv'] = l_tv

        # ---------- Back-prop ----------
        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)



        # preds = self.net_g(self.lq)
        # if not isinstance(preds, list):
        #     preds = [preds]

        # self.output = preds[-1]

        # loss_dict = OrderedDict()
        # # Pixel loss
        # l_pix = 0.
        # for pred in preds:
        #     l_pix += self.cri_pix(pred, self.gt)
        # loss_dict['l_pix'] = l_pix

        # l_pix.backward()

        # if self.opt['train']['use_grad_clip']:
        #     torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        # self.optimizer_g.step()

        # self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # # -----------------------
        # # Save visualization images at set intervals.
        # # -----------------------
        # vis_interval = self.opt['train'].get('visualization_interval', 100000)
        # if current_iter % vis_interval == 0:
        #     # --- 1. RSS Visualizations ---
        #     # self.lq shape: [B, 64, H, W]
        #     # First 32 channels: LR MRI images, last 32 channels: noise maps.
        #     lr_input = self.lq[:, :32, :, :]   # [B, 32, H, W]
        #     noise_map = self.lq[:, 32:, :, :]    # [B, 32, H, W]
        #     # Combine coils using RSS (per sample) to form a single image:
        #     lr_combined = torch.sqrt(torch.sum(lr_input ** 2, dim=1, keepdim=True))       # [B, 1, H, W]
        #     noise_combined = torch.sqrt(torch.sum(noise_map ** 2, dim=1, keepdim=True))     # [B, 1, H, W]

        #     # --- Combine the ground truth and prediction ---
        #     # Both self.gt and self.output have shape: [B, 32, H, W]
        #     gt_combined = torch.sqrt(torch.sum(self.gt ** 2, dim=1, keepdim=True))          # [B, 1, H, W]
        #     pred_combined = torch.sqrt(torch.sum(self.output ** 2, dim=1, keepdim=True))      # [B, 1, H, W]

        #     # For visualization, select the first sample in the batch.
        #     lr_vis = lr_combined[0]      # shape: [1, H, W]
        #     noise_vis = noise_combined[0]
        #     gt_vis = gt_combined[0]
        #     pred_vis = pred_combined[0]

        #     # Replicate the single channel to 3 channels for tensor2img.
        #     lr_vis = lr_vis.repeat(3, 1, 1)        # shape: [3, H, W]
        #     noise_vis = noise_vis.repeat(3, 1, 1)
        #     gt_vis = gt_vis.repeat(3, 1, 1)
        #     pred_vis = pred_vis.repeat(3, 1, 1)

        #     # Convert to numpy images using tensor2img.
        #     lr_img_np = tensor2img(lr_vis, rgb2bgr=False)
        #     noise_img_np = tensor2img(noise_vis, rgb2bgr=False)
        #     gt_img_np = tensor2img(gt_vis, rgb2bgr=False)
        #     pred_img_np = tensor2img(pred_vis, rgb2bgr=False)

        #     # Construct save paths.
        #     vis_dir = self.opt['path'].get('visualization', './visualization')
        #     os.makedirs(vis_dir, exist_ok=True)
        #     lr_save_path = os.path.join(vis_dir, f"iter_{current_iter}_lr.png")
        #     noise_save_path = os.path.join(vis_dir, f"iter_{current_iter}_noise.png")
        #     gt_save_path = os.path.join(vis_dir, f"iter_{current_iter}_gt.png")
        #     pred_save_path = os.path.join(vis_dir, f"iter_{current_iter}_pred.png")

        #     # Save RSS images.
        #     imwrite(lr_img_np, lr_save_path)
        #     imwrite(noise_img_np, noise_save_path)
        #     imwrite(gt_img_np, gt_save_path)
        #     imwrite(pred_img_np, pred_save_path)

        #     # -----------------------
        #     # 2. Separate Coil Visualizations (without RSS)
        #     # -----------------------
        #     # For the first sample in the batch, save the individual coil images as a grid.
        #     lr_separate = self.lq[0, :32, :, :].unsqueeze(1)    # [32, 1, H, W]
        #     noise_separate = self.lq[0, 32:, :, :].unsqueeze(1)   # [32, 1, H, W]
        #     gt_separate = self.gt[0].unsqueeze(1)                 # [32, 1, H, W]
        #     pred_separate = self.output[0].unsqueeze(1)           # [32, 1, H, W]

        #     lr_sep_img = tensor2img(lr_separate, rgb2bgr=False)
        #     noise_sep_img = tensor2img(noise_separate, rgb2bgr=False)
        #     gt_sep_img = tensor2img(gt_separate, rgb2bgr=False)
        #     pred_sep_img = tensor2img(pred_separate, rgb2bgr=False)

        #     lr_sep_save_path = os.path.join(vis_dir, f"iter_{current_iter}_lr_sep.png")
        #     noise_sep_save_path = os.path.join(vis_dir, f"iter_{current_iter}_noise_sep.png")
        #     gt_sep_save_path = os.path.join(vis_dir, f"iter_{current_iter}_gt_sep.png")
        #     pred_sep_save_path = os.path.join(vis_dir, f"iter_{current_iter}_pred_sep.png")

        #     imwrite(lr_sep_img, lr_sep_save_path)
        #     imwrite(noise_sep_img, noise_sep_save_path)
        #     imwrite(gt_sep_img, gt_sep_save_path)
        #     imwrite(pred_sep_img, pred_sep_save_path)

        #     # -----------------------
        #     # 3. Residual Visualizations (Noisy - Prediction)
        #     # -----------------------
        #     # Calculate the residual between the noisy input (first 32 channels) and the prediction.
        #     residual = lr_input - self.output  # [B, 32, H, W]

        #     # RSS residual: combine coil residuals via RSS.
        #     residual_rss = torch.sqrt(torch.sum(residual ** 2, dim=1, keepdim=True))  # [B, 1, H, W]
        #     residual_rss_vis = residual_rss[0].repeat(3, 1, 1)  # replicate to 3 channels

        #     # Convert and save the RSS residual image.
        #     residual_rss_img_np = tensor2img(residual_rss_vis, rgb2bgr=False)
        #     residual_rss_save_path = os.path.join(vis_dir, f"iter_{current_iter}_residual_rss.png")
        #     imwrite(residual_rss_img_np, residual_rss_save_path)

        #     # Separate residual: show each coil's residual as a grid for the first sample.
        #     residual_sep = residual[0].unsqueeze(1)  # [32, 1, H, W]
        #     residual_sep_img = tensor2img(residual_sep, rgb2bgr=False)
        #     residual_sep_save_path = os.path.join(vis_dir, f"iter_{current_iter}_residual_sep.png")
        #     imwrite(residual_sep_img, residual_sep_save_path)

        #     # -----------------------
        #     # 4. Save 4D NIfTI Volumes (Noisy, Prediction, Residual)
        #     # -----------------------
        #     # For the first sample in the batch:
        #     # Noisy input: shape [32, H, W] from the first 32 channels.
        #     noisy_4d = self.lq[0, :32, :, :].detach().cpu().numpy()   # [32, H, W]
        #     pred_4d = self.output[0].detach().cpu().numpy()            # [32, H, W]
        #     residual_4d = noisy_4d - pred_4d                            # [32, H, W]

        #     # Rearrange to [H, W, 32] for NIfTI storage.
        #     noisy_4d = np.transpose(noisy_4d, (1, 2, 0))
        #     pred_4d = np.transpose(pred_4d, (1, 2, 0))
        #     residual_4d = np.transpose(residual_4d, (1, 2, 0))

        #     # Create NIfTI images (using identity affine; adjust if necessary)
        #     noisy_nii = nib.Nifti1Image(noisy_4d, np.eye(4))
        #     pred_nii = nib.Nifti1Image(pred_4d, np.eye(4))
        #     residual_nii = nib.Nifti1Image(residual_4d, np.eye(4))

        #     # Construct NIfTI save paths.
        #     noisy_nii_path = os.path.join(vis_dir, f"iter_{current_iter}_noisy.nii.gz")
        #     pred_nii_path = os.path.join(vis_dir, f"iter_{current_iter}_pred.nii.gz")
        #     residual_nii_path = os.path.join(vis_dir, f"iter_{current_iter}_residual.nii.gz")

        #     nib.save(noisy_nii, noisy_nii_path)
        #     nib.save(pred_nii, pred_nii_path)
        #     nib.save(residual_nii, residual_nii_path)


    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        # print(self.output)
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()
            visuals = self.get_current_visuals()

            # ---------------------------
            # 1. Process Prediction (Result)
            # ---------------------------
            result_tensor = visuals['result']  # Expected shape: (1, 32, H, W)
            print("Original result_tensor shape:", result_tensor.shape)
            if result_tensor.dim() == 4:
                result_tensor = result_tensor[0]  # Now shape: (32, H, W)
            print("Result tensor after removing batch dimension:", result_tensor.shape)

            if result_tensor.shape[0] > 3:
                # RSS combine across channels: [1, H, W]
                combined_result = torch.sqrt(torch.sum(result_tensor ** 2, dim=0, keepdim=True))
                # Replicate to 3 channels for visualization
                combined_result = combined_result.repeat(3, 1, 1)
                print("Combined result shape after RSS and repeat:", combined_result.shape)
            else:
                combined_result = result_tensor

            print("Result tensor min/max:", result_tensor.min().item(), result_tensor.max().item())
            print("Combined result min/max before normalization:", combined_result.min().item(), combined_result.max().item())
            combined_result = (combined_result - combined_result.min()) / (combined_result.max() - combined_result.min() + 1e-8)
            print("Combined result min/max after normalization:", combined_result.min().item(), combined_result.max().item())

            sr_img = tensor2img(combined_result, rgb2bgr=rgb2bgr)

            # ---------------------------
            # 2. Process Ground Truth (if available)
            # ---------------------------
            if 'gt' in visuals:
                gt_tensor = visuals['gt']
                print("Original gt_tensor shape:", gt_tensor.shape)
                if gt_tensor.dim() == 4:
                    gt_tensor = gt_tensor[0]  # shape: (32, H, W)
                if gt_tensor.shape[0] > 3:
                    combined_gt = torch.sqrt(torch.sum(gt_tensor ** 2, dim=0, keepdim=True))
                    combined_gt = combined_gt.repeat(3, 1, 1)
                    print("Combined gt shape after RSS and repeat:", combined_gt.shape)
                else:
                    combined_gt = gt_tensor
                combined_gt = (combined_gt - combined_gt.min()) / (combined_gt.max() - combined_gt.min() + 1e-8)
                gt_img = tensor2img(combined_gt, rgb2bgr=rgb2bgr)
                del self.gt
            else:
                gt_img = None

            # Memory cleanup for GPU tensors.
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # ---------------------------
            # 3. Save Combined (RSS) PNG Visualizations
            # ---------------------------
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_gt.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                if gt_img is not None:
                    imwrite(gt_img, save_gt_img_path)

            # # ---------------------------
            # # 4. Separate Coil Visualizations (Without RSS)
            # # ---------------------------
            # # Use the original input from val_data.
            # lr_separate = val_data['lq'][0, :32, :, :].unsqueeze(1)    # [32, 1, H, W]
            # noise_separate = val_data['lq'][0, 32:, :, :].unsqueeze(1)   # [32, 1, H, W]
            # pred_separate = result_tensor.unsqueeze(1)                  # [32, 1, H, W]
            # if 'gt' in visuals:
            #     gt_separate = gt_tensor.unsqueeze(1)                    # [32, 1, H, W]
            # else:
            #     gt_separate = None

            # lr_sep_img = tensor2img(lr_separate, rgb2bgr=rgb2bgr)
            # noise_sep_img = tensor2img(noise_separate, rgb2bgr=rgb2bgr)
            # pred_sep_img = tensor2img(pred_separate, rgb2bgr=rgb2bgr)
            # if gt_separate is not None:
            #     gt_sep_img = tensor2img(gt_separate, rgb2bgr=rgb2bgr)

            # lr_sep_save_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_lr_sep.png')
            # noise_sep_save_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_noise_sep.png')
            # pred_sep_save_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_pred_sep.png')
            # if gt_separate is not None:
            #     gt_sep_save_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt_sep.png')

            # imwrite(lr_sep_img, lr_sep_save_path)
            # imwrite(noise_sep_img, noise_sep_save_path)
            # imwrite(pred_sep_img, pred_sep_save_path)
            # if gt_separate is not None:
            #     imwrite(gt_sep_img, gt_sep_save_path)

            # # ---------------------------
            # # 5. Residual Visualizations (Noisy - Prediction)
            # # ---------------------------
            # # Use the noisy input (first 32 channels from val_data) and the prediction.
            # lr_input_eval = val_data['lq'][0, :32, :, :]  # [32, H, W]
            # residual = lr_input_eval - result_tensor       # [32, H, W]
            # # RSS residual:
            # residual_rss = torch.sqrt(torch.sum(residual ** 2, dim=0, keepdim=True))  # [1, H, W]
            # residual_rss_vis = residual_rss.repeat(3, 1, 1)  # [3, H, W]
            # residual_rss_img_np = tensor2img(residual_rss_vis, rgb2bgr=rgb2bgr)
            # residual_rss_save_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_residual_rss.png')
            # imwrite(residual_rss_img_np, residual_rss_save_path)

            # # Separate residual (grid of individual coil residuals):
            # residual_sep = residual.unsqueeze(1)  # [32, 1, H, W]
            # residual_sep_img = tensor2img(residual_sep, rgb2bgr=rgb2bgr)
            # residual_sep_save_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_residual_sep.png')
            # imwrite(residual_sep_img, residual_sep_save_path)

            # # ---------------------------
            # # 6. Save 4D NIfTI Volumes (Noisy, Prediction, Residual)
            # # ---------------------------
            # # For the first sample in the batch:
            # noisy_4d = val_data['lq'][0, :32, :, :].detach().cpu().numpy()   # [32, H, W]
            # pred_4d = result_tensor.detach().cpu().numpy()                    # [32, H, W]
            # residual_4d = noisy_4d - pred_4d                                    # [32, H, W]

            # # Rearrange to [H, W, 32] for NIfTI storage.
            # noisy_4d = np.transpose(noisy_4d, (1, 2, 0))
            # pred_4d = np.transpose(pred_4d, (1, 2, 0))
            # residual_4d = np.transpose(residual_4d, (1, 2, 0))
            # import nibabel as nib
            # noisy_nii = nib.Nifti1Image(noisy_4d, np.eye(4))
            # pred_nii = nib.Nifti1Image(pred_4d, np.eye(4))
            # residual_nii = nib.Nifti1Image(residual_4d, np.eye(4))
            # noisy_nii_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_noisy.nii.gz')
            # pred_nii_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_pred.nii.gz')
            # residual_nii_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_residual.nii.gz')
            # nib.save(noisy_nii, noisy_nii_path)
            # nib.save(pred_nii, pred_nii_path)
            # nib.save(residual_nii, residual_nii_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
                cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric



    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
