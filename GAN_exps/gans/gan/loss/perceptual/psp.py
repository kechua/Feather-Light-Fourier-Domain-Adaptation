import torch
from torch import nn, Tensor

from gan.loss.loss_base import Loss
from gan.loss.perceptual import id_loss, w_norm
from gan.loss.perceptual.lpips.lpips import LPIPS
import torch.nn.functional as F

from gan.models.base import requires_grad
from parameters.path import Paths


class PSPLoss(nn.Module):

    def __init__(self,
                 lpips_lambda=0.8,
                 id_lambda=0.1,
                 w_norm_lambda=0,
                 l2_lambda=1.0,
                 latent_avg=None,
                 model_path_ir_se50=f"{Paths.default.models()}/model_ir_se50.pth"):
        super().__init__()

        self.lpips_lambda = lpips_lambda
        self.id_lambda = id_lambda
        self.w_norm_lambda = w_norm_lambda
        self.l2_lambda = l2_lambda
        self.latent_avg = latent_avg

        if lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
            requires_grad(self.lpips_loss, False)
        if id_lambda > 0:
            self.id_loss = id_loss.IDLoss(model_path_ir_se50).cuda().eval()
            requires_grad(self.id_loss, False)
        if w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)


    def forward(self, x, y, y_hat, latent):

        loss_dict = {}
        loss = 0.0
        id_logs = None
        eps = 1e-8

        if self.id_lambda > eps:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.id_lambda

        if self.l2_lambda > eps:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.l2_lambda

        if self.lpips_lambda > eps:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.lpips_lambda

        if self.w_norm_lambda > eps:
            loss_w_norm = self.w_norm_loss(latent, self.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.w_norm_lambda

        loss_dict['brule_loss'] = float(loss)

        return Loss(loss)
