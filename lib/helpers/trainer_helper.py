import os
import tqdm

import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.losses.centernet_loss import compute_centernet3d_loss

writer = SummaryWriter('./log')
class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(model=self.model.to(self.device),
                                         optimizer=self.optimizer,
                                         filename=cfg['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.gpu_ids = list(map(int, cfg['gpu_ids'].split(',')))
        # print("gpuid:",self.gpu_ids)
        self.model = torch.nn.DataParallel(model, device_ids=self.gpu_ids).to(self.device)



    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            epoch_loss=self.train_one_epoch()
            writer.add_scalar('loss/total_loss',epoch_loss['total_loss'],epoch)
            writer.add_scalar('loss/seg_loss',epoch_loss['seg'],epoch)
            writer.add_scalar('loss/offset2d_loss',epoch_loss['offset2d'],epoch)
            writer.add_scalar('loss/size_loss',epoch_loss['size2d'],epoch)
            writer.add_scalar('loss/offset3d_loss',epoch_loss['offset3d'],epoch)
            writer.add_scalar('loss/depth_loss',epoch_loss['depth'],epoch)
            writer.add_scalar('loss/size3d_loss',epoch_loss['size3d'],epoch)
            writer.add_scalar('loss/heading_cls_loss',epoch_loss['heading_cls'],epoch)
            writer.add_scalar('loss/heading_reg_loss',epoch_loss['heading_reg'],epoch)
            writer.add_scalar('loss/group_loss',epoch_loss['grouploss'],epoch)
            
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()


            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs('checkpoint_first_group', exist_ok=True)
                ckpt_name = os.path.join('checkpoint_first_group', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)

            progress_bar.update()
        writer.close()

        return None


    def train_one_epoch(self):
        self.model.train()
        loss_dicts={"total_loss":0,'seg':0,'offset2d':0,'size2d':0,'offset3d':0,'depth':0,'size3d':0,
        'heading_cls':0,'heading_reg':0,'grouploss':0}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            total_loss, stats_batch = compute_centernet3d_loss(outputs, targets,info)
            loss_dicts["total_loss"]+=total_loss.item()
            loss_dicts['seg']+=stats_batch['seg']
            loss_dicts['offset2d']+=stats_batch['offset2d']
            loss_dicts['size2d']+=stats_batch['size2d']
            loss_dicts['offset3d']+=stats_batch['offset3d']
            loss_dicts['depth']+=stats_batch['depth']
            loss_dicts['size3d']+=stats_batch['size3d']
            loss_dicts['heading_cls']+=stats_batch['heading_cls']
            loss_dicts['heading_reg']+=stats_batch['heading_reg']
            loss_dicts['grouploss']+=stats_batch['grouploss']

            writer.add_scalar('perbatch_loss/total_loss',total_loss,batch_idx)
            writer.add_scalar('perbatch_loss/seg_loss',stats_batch['seg'],batch_idx)
            writer.add_scalar('perbatch_loss/offset2d_loss',stats_batch['offset2d'],batch_idx)
            writer.add_scalar('perbatch_loss/size_loss',stats_batch['size2d'],batch_idx)
            writer.add_scalar('perbatch_loss/offset3d_loss',stats_batch['offset3d'],batch_idx)
            writer.add_scalar('perbatch_loss/depth_loss',stats_batch['depth'],batch_idx)
            writer.add_scalar('perbatch_loss/size3d_loss',stats_batch['size3d'],batch_idx)
            writer.add_scalar('perbatch_loss/heading_clsloss',stats_batch['heading_cls'],batch_idx)
            writer.add_scalar('perbatch_loss/heading_regloss',stats_batch['heading_reg'],batch_idx)
            writer.add_scalar('perbatch_loss/grouploss',stats_batch['grouploss'],batch_idx)

            total_loss.backward()
            self.optimizer.step()

            progress_bar.update()
        
        progress_bar.close()
        return loss_dicts




