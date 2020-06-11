import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
# from torch.utils.tensorboard import SummaryWriter
from datasets import make_train_loader
from efficientnet_pytorch import EfficientNet
# from model import efficientnet, EffNet, Resnest
import json


def efficientnet(level='b1'):
    model = EfficientNet.from_pretrained(f'efficientnet-{level}')
    model._fc.out_features = 3
    return model


class EffNet(nn.Module):
    def __init__(self, level='b7'):
        super(EffNet, self).__init__()
        self.model = EfficientNet.from_pretrained(f'efficientnet-{level}')

        self.fc_list = nn.Sequential(nn.Linear(1000, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 3)
                                     )

    def forward(self, x):
        return self.fc_list(self.model(x))



class EffNetTrainer(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(EffNetTrainer, self).__init__(*args, **kwargs)
        self.hparams = hparams
        # self.model = efficientnet('b7')
        # self.model = Resnest()
        self.model = EffNet('b7')
        self.loss_func = nn.CrossEntropyLoss()

    def prepare_data(self):
        self.train_loader, self.valid_loader = make_train_loader(self.hparams)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def configure_optimizers(self):
        opt_cfg = self.hparams['optimizer']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 5, 1e-7)
        return {'optimizer': self.optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        src, trg = batch
        logits = self(src)
        loss = self.loss_func(logits, trg)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'train_loss': loss_mean.item()}
        self.logger.log_metrics(logs, self.current_epoch)
        results = {'progress_bar': logs, 'train_loss': loss_mean}
        return results

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        logits = self(src)
        loss = self.loss_func(logits, trg)
        correct = torch.tensor(logits.max(1)[1] == trg, dtype=torch.float64)
        return {'val_loss': loss, 'correct': correct}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['correct'] for x in outputs]).mean()
        self.hparams['lr'] = self.optimizer.state_dict()[
            'param_groups'][0]['lr']
        logs = {'val_acc': acc.item(), 'val_loss': val_loss_mean.item(),
                'lr': self.hparams['lr']}
        self.logger.log_metrics(logs, self.current_epoch)
        self.logger.log_hyperparams(self.hparams, logs)
        # self.logger.finalize('success')
        return {'val_loss': val_loss_mean.cpu(), 'progress_bar': logs}