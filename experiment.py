import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from RNADataset import *

class RNAExperiment(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(RNAExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.dataobject = RNAData(self.params['data_path'], self.params['test_path'], self.params['fc_path'])
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        if batch_idx == 0:
            self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        print("train loss : ", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        val_loss['val_loss'] = val_loss.pop('loss')
        val_loss['val_Reconstruction_Loss'] = val_loss.pop('Reconstruction_Loss')
        val_loss['val_KLD'] = val_loss.pop('KLD')

        if batch_idx == 0:
            self.log("val_loss", val_loss['val_loss'])
            self.logger.experiment.log({key: val.item() for key, val in val_loss.items()})

        print("val loss : ", val_loss)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        #self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # def sample_images(self):
    #     # Get sample reconstruction image
    #     test_input, test_label = next(iter(self.sample_dataloader))
    #     test_input = test_input.to(self.curr_device)
    #     test_label = test_label.to(self.curr_device)
    #     recons = self.model.generate(test_input, labels = test_label)
    #     vutils.save_image(recons.data,
    #                       f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
    #                       f"recons_{self.logger.name}_{self.current_epoch}.png",
    #                       normalize=True,
    #                       nrow=12)

    #     # vutils.save_image(test_input.data,
    #     #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
    #     #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
    #     #                   normalize=True,
    #     #                   nrow=12)

    #     try:
    #         samples = self.model.sample(144,
    #                                     self.curr_device,
    #                                     labels = test_label)
    #         vutils.save_image(samples.cpu().data,
    #                           f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
    #                           f"{self.logger.name}_{self.current_epoch}.png",
    #                           normalize=True,
    #                           nrow=12)
    #     except:
    #         pass


    #     del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        #transform = self.data_transforms()

        if self.params['dataset'] == 'lindel':
            dataset = self.dataobject.get_dataset('train')
            print(dataset.shape)
            print(dataset)
            input()
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=False,
                          num_workers=1
                          )

    @data_loader
    def val_dataloader(self):
        #transform = self.data_transforms()

        if self.params['dataset'] == 'lindel':
            self.sample_dataloader =  DataLoader(self.dataobject.get_dataset('val'),
                                                 batch_size = self.params['batch_size'],
                                                 shuffle = False,
                                                 num_workers=1
                                                 )
            self.num_val_imgs = len(self.sample_dataloader)
            print("Number of val images = ", self.num_val_imgs)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'lindel':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform
