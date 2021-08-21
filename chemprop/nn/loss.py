import torch
import torch.nn as nn
from chemprop.args import TrainArgs


def get_loss_func(args: TrainArgs) -> nn.Module:
    if args.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    if args.dataset_type == 'regression':
        if args.mve:
            assert args.quantileregression == 'None'
            return heteroscedastic_loss
        elif args.quantileregression == 'Pinball':
            return Pinball()
        elif args.quantileregression == 'Interval':
            return Interval()
        elif args.quantileregression == 'Calibration':
            return Calibration()
        elif args.quantileregression == 'ScaledCalibration':
            return ScaledCalibration()
        else:
            return nn.MSELoss(reduction='none')
    if args.dataset_type == 'multiclass':
        return nn.CrossEntropyLoss(reduction='none')
    raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')


def heteroscedastic_loss(true, mean, log_var):
    precision = torch.exp(-log_var)
    loss = precision * (true - mean) ** 2 + log_var
    return loss


class Pinball(torch.nn.Module):
    def __init__(self):
        super(Pinball, self).__init__()

    def forward(self,
                y,
                yhat,
                additional_outputs):
        tau, y_hat_reversed = additional_outputs
        y = y.to(yhat.device)
        tau = tau.to(yhat.device)
        y_hat_reversed = y_hat_reversed.to(yhat.device)

        diff1 = yhat - y
        mask1 = (diff1.ge(0).float() - tau).detach()
        loss1 = (mask1 * diff1).mean()

        diff2 = y_hat_reversed - y
        mask2 = (diff2.ge(0).float() - (1-tau)).detach()
        loss2 = (mask2 * diff2).mean()

        loss = loss1 + loss2
        return loss


class Interval(torch.nn.Module):
    def __init__(self):
        super(Interval, self).__init__()

    def forward(self,
                y,
                yhat,
                additional_outputs):
        lhat = yhat
        tau, uhat = additional_outputs
        y = y.to(yhat.device)
        tau = tau.to(yhat.device)
        uhat = uhat.to(yhat.device)

        interval = torch.abs(uhat-lhat)
        l = (2/tau) * ((lhat-y)*lhat.ge(y).float() + (y-uhat)*y.ge(uhat).float())

        return l + interval


class Calibration(torch.nn.Module):
    def __init__(self):
        super(Calibration, self).__init__()

    def forward(self,
                y,
                yhat,
                additional_outputs):
        tau, _ = additional_outputs
        tau = tau[0, 0].detach().cpu().item()
        p_obs_avg = (y <= yhat).float().mean().detach().cpu().item()
        if p_obs_avg < tau:
            loss = (y-yhat)*((y>yhat).float())
        elif p_obs_avg >= tau:
            loss = (yhat-y)*((yhat-y)).float()
        return loss


class ScaledCalibration(torch.nn.Module):
    def __init__(self):
        super(ScaledCalibration, self).__init__()

    def forward(self,
                y,
                yhat,
                additional_outputs):
        tau, _ = additional_outputs
        tau = tau[0, 0].detach().cpu().item()
        p_obs_avg = (y <= yhat).float().mean().detach().cpu().item()
        if p_obs_avg < tau:
            loss = (y-yhat)*((y>yhat).float())
        elif p_obs_avg >= tau:
            loss = (yhat-y)*((yhat-y).float())
        return loss*abs(p_obs_avg-tau)



 