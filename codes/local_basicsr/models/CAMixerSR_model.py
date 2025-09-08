import torch
from torch.nn import functional as F

from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

import math
from tqdm import tqdm
from os import path as osp

#新加
import os

@MODEL_REGISTRY.register()
class CAModel(SRModel):

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output, loss_ratio = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        l_total += loss_ratio
        loss_dict['l_ratio'] = loss_ratio
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        #新加
    # 添加验证结果保存功能
    # 添加验证结果保存功能
    def validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation function with result saving to txt file."""
        
        # 调用父类的验证方法
        super().validation(dataloader, current_iter, tb_logger, save_img)
        
        # 保存验证结果到txt文件
        self._save_validation_results(dataloader.dataset.opt['name'], current_iter)

    def _save_validation_results(self, dataset_name, current_iter):
        """Save validation results to txt file."""
        if hasattr(self, 'metric_results') and self.metric_results:
            # 创建results目录
            results_dir = osp.join(self.opt['path']['log'], 'validation_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 创建txt文件路径
            txt_path = osp.join(results_dir, f'{dataset_name}_validation_results.txt')
            
            # 写入结果
            with open(txt_path, 'a') as f:
                f.write(f"Iteration {current_iter}, Dataset {dataset_name}:\n")
                for metric, value in self.metric_results.items():
                    f.write(f"  {metric}: {value:.6f}\n")
                f.write("\n")