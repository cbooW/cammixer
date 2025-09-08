import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

import math
from tqdm import tqdm
from os import path as osp

#新加
import os

@MODEL_REGISTRY.register()
class TileModel(SRModel):

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

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.tile_test(self.lq, self.net_g_ema)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.tile_test(self.lq, self.net_g)
            self.net_g.train()

    def tile_test(self, img_lq, model):
        tile = self.opt['tile']
        if tile == 0:
            # test the image as a whole
            output = model(img_lq)
        else:
            # test the image tile by tile
            b, c, h, w = img_lq.size()
            tile = min(tile, h, w)
            tile_overlap = tile//16
            sf =  self.opt['scale']

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            output = E.div_(W)

        return output


        #新加新加
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