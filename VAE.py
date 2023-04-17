# General imports
import json
import os
import glob
from torch.utils.data import Dataset
import cv2
import json
import sys
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils, datasets
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import math

# Pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

# Raytune
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback


###################
# VAE Architecture
###################

class VAE(pl.LightningModule):
    def __init__(self, dataloader, config, inv_transform, latent_dim=256, num_channels=1):
        super().__init__()

        self.lr = config["lr"]
        self.vae_beta = config["vae_beta"]
        self.num_batches = len(dataloader)
        self.adam_beta1 = config["adam_beta1"]
        self.adam_beta2 = config["adam_beta2"]
        self.max_epochs = config["max_epochs"]
        self.hp_search_mode = config["hp_search_mode"]
        self.weight_decay = config["weight_decay"]
        self.inv_transform = inv_transform
        self.kl_anneal = config["kl_anneal"]
        self.latent_dim = latent_dim

        num_kl_anneal_cycle = max(4, 4*(int(self.max_epochs/100)))
        if self.kl_anneal:
            self.kl_anneal_schedule = self.frange_cycle_linear(0.0, self.vae_beta, self.max_epochs*self.num_batches, num_kl_anneal_cycle, 0.5)

        # starts with 32x32
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=3, padding=1, stride=2), # 16x16
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=3, padding=1, stride=2), # 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2), # 4x4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten() # 32x4x4=512
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32*4*4),
            nn.BatchNorm1d(32*4*4),
            nn.ReLU(),
            nn.Dropout1d(p=0.05),
            nn.Unflatten(1, (32,4,4)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 8x8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.05),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x16
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(p=0.05),
            nn.ConvTranspose2d(8, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x32
        )

        self.linear_mu = nn.Linear(32*4*4, latent_dim)
        self.linear_logvar = nn.Linear(32*4*4, latent_dim)

        self.train_rec_loss = []
        self.train_kl_loss = []
        self.train_spec_loss = []
        self.train_tot_loss = []
        self.train_org_img = []
        self.train_rec_img = []

        self.val_rec_loss = []
        self.val_kl_loss = []
        self.val_spec_loss = []
        self.val_tot_loss = []
        self.val_org_img = []
        self.val_rec_img = []

        self.save_hyperparameters()

    def encode(self, x):
        hidden = self.encoder(x.float())
        mu = self.linear_mu(hidden)
        log_var = self.linear_logvar(hidden)
        return mu, log_var

    def decode(self, x):
        x = self.decoder(x.float())
        return x

    def reparametrize(self, mu, log_var):
        sigma = torch.exp(0.5*log_var)
        z = torch.randn_like(sigma)
        return mu + sigma*z

    def compute_loss(self, inp_imgs, out_imgs, mu, log_var, kl_weight):
        
        batch_size = inp_imgs.shape[0]
        
        kl_loss = ((-0.5*(1+log_var - mu**2 - torch.exp(log_var))).sum())/batch_size
        recon_loss_criterion = nn.MSELoss(reduction="sum")
        recon_loss = recon_loss_criterion(out_imgs, inp_imgs.float())/batch_size

        total_loss = recon_loss + kl_weight*kl_loss

        return recon_loss, kl_loss, total_loss

    def forward(self, x):
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        output = self.decode(hidden)
        return mu, log_var, output

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        mu, log_var, output = self.forward(imgs.float())

        if self.kl_anneal:
            kl_weight = self.kl_anneal_schedule[self.current_epoch*self.num_batches + batch_idx]
        else:
            kl_weight=self.vae_beta
    
        recon_loss, kl_loss, total_loss = self.compute_loss(imgs, output, mu, log_var, kl_weight)
       
        ###########
        # Logging
        ###########
        
        self.train_rec_loss.append(recon_loss)
        self.train_kl_loss.append(kl_loss)
        self.train_tot_loss.append(total_loss)

        if batch_idx == 0:
            self.train_org_img.append(imgs)
            self.train_rec_img.append(output)

        return total_loss

    def on_train_epoch_end(self):

        avg_rec_loss = torch.stack(self.train_rec_loss).mean()
        avg_kl_loss = torch.stack(self.train_kl_loss).mean()
        avg_tot_loss = torch.stack(self.train_tot_loss).mean()

        avg_rec_loss = 1e10 if torch.isnan(avg_rec_loss) else avg_rec_loss
        avg_kl_loss = 1e10 if torch.isnan(avg_kl_loss) else avg_kl_loss
        avg_tot_loss = 1e10 if torch.isnan(avg_tot_loss) else avg_tot_loss
        
        ##############
        # Log scalars
        ##############
        self.log('train/rec_loss', avg_rec_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train/kl_loss', avg_kl_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train/tot_loss', avg_tot_loss, on_step=False,
                 on_epoch=True, prog_bar=True)


        ##############
        # Log Images
        ##############

        if (self.current_epoch+1)%20 == 0 and not self.hp_search_mode: 
            tensorboard = self.logger.experiment

            unnorm_org_img = self.inv_transform(self.train_org_img[-1])
            grid1 = utils.make_grid(unnorm_org_img, nrow=8)
            tensorboard.add_image('train/org_images', grid1, self.global_step)

            unnorm_rec_img = self.inv_transform(self.train_rec_img[-1])
            grid2 = utils.make_grid(unnorm_rec_img, nrow=8)
            tensorboard.add_image('train/rec_images', grid2, self.global_step)

            z_rand = torch.randn(64, self.latent_dim).to(self.train_org_img[-1].device)
            unnorm_rand_img = self.inv_transform(self.decode(z_rand))
            grid3 = utils.make_grid(unnorm_rand_img, nrow=8)
            tensorboard.add_image('train/rand_images', grid3, self.global_step)

        self.train_rec_loss.clear()
        self.train_kl_loss.clear()
        self.train_tot_loss.clear()

        self.train_org_img.clear()
        self.train_rec_img.clear()

    def validation_step(self, batch, batch_idx):
        
        self.eval()

        # with torch.no_grad():
        imgs, labels = batch
        mu, log_var, output = self.forward(imgs.float())

        recon_loss, kl_loss, total_loss = self.compute_loss(imgs, output, mu, log_var, kl_weight=self.vae_beta)

        self.val_rec_loss.append(recon_loss)
        self.val_kl_loss.append(kl_loss)
        self.val_tot_loss.append(total_loss)

        if batch_idx == 0:
            self.val_org_img.append(imgs)
            self.val_rec_img.append(output)

        self.train()
        
        return total_loss

    def on_validation_epoch_end(self):

        avg_rec_loss = torch.stack(self.val_rec_loss).mean()
        avg_kl_loss = torch.stack(self.val_kl_loss).mean()
        avg_tot_loss = torch.stack(self.val_tot_loss).mean()
        
        avg_rec_loss = 1e10 if torch.isnan(avg_rec_loss) else avg_rec_loss
        avg_kl_loss = 1e10 if torch.isnan(avg_kl_loss) else avg_kl_loss
        avg_tot_loss = 1e10 if torch.isnan(avg_tot_loss) else avg_tot_loss

        ##############
        # Log scalars
        ##############
        self.log('val/rec_loss', avg_rec_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('val/kl_loss', avg_kl_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('val/tot_loss', avg_tot_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        ##############
        # Log Images
        ##############

        if (self.current_epoch+1)%20 == 0 and not self.hp_search_mode:
            self.eval()
            tensorboard = self.logger.experiment

            unnorm_org_img = self.inv_transform(self.val_org_img[-1])
            grid1 = utils.make_grid(unnorm_org_img, nrow=8)
            tensorboard.add_image('val/org_images', grid1, self.global_step)

            unnorm_rec_img = self.inv_transform(self.val_rec_img[-1])
            grid2 = utils.make_grid(unnorm_rec_img, nrow=8)
            tensorboard.add_image('val/rec_images', grid2, self.global_step)

            z_rand = torch.randn(64, self.latent_dim).to(self.val_org_img[-1].device)
            unnorm_rand_img = self.inv_transform(self.decode(z_rand))
            grid3 = utils.make_grid(unnorm_rand_img, nrow=8)
            tensorboard.add_image('val/rand_images', grid3, self.global_step)
            self.train()

        self.val_rec_loss.clear()
        self.val_kl_loss.clear()
        self.val_tot_loss.clear()
        self.val_org_img.clear()
        self.val_rec_img.clear()

    def configure_optimizers(self):
        
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.linear_mu.parameters()) + list(self.linear_logvar.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.weight_decay)

        return [opt], []

    @staticmethod
    def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)*stop
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L

########################
# Hyperparameter Search
########################

def hp_search(args):
    
    print("Performing hyperparameter search")
    root_dir = f"ray_results_{args.dataset}"

    num_hpsearch_samples = args.num_hpsearch_samples
    num_hpsearch_epochs = args.num_hpsearch_epochs
    gpus_per_trial = 1

    def train_model_asha(config, num_hpsearch_epochs=50, num_gpus=0):
        train_dataloader = DataLoader(args.train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_dataloader = DataLoader(args.test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)

        model = VAE(dataloader=train_dataloader, config=config, inv_transform=args.inv_transform, latent_dim=args.latent_dim, num_channels=args.num_channels)
        logger = TensorBoardLogger(save_dir=f"tensorboard_logs")
        
        metrics = {"loss": args.hp_search_loss}
        trainer = Trainer(max_epochs=num_hpsearch_epochs, default_root_dir="./", devices=num_gpus, log_every_n_steps=len(train_dataloader), logger=logger, callbacks=[TuneReportCallback(metrics, on="validation_end")])
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    def tune_model_asha(num_hpsearch_samples=100, num_hpsearch_epochs=50, gpus_per_trial=1):

        config = {
                "lr": tune.loguniform(1e-5, 1e-2),
                "batch_size": 128,
                "vae_beta": 1.,# tune.loguniform(1e-2, 1),
                "rec_lambda": 1., # tune.loguniform(1e-4, 1), #tune.choice([0.1, 1, 64, 128, 256, 512, 1024]),
                "adam_beta1": tune.uniform(0.5, 0.99),
                "adam_beta2": tune.uniform(0.5, 0.99),
                "weight_decay": tune.loguniform(1e-5, 1e-1),
                "kl_anneal": True,
                "max_epochs": num_hpsearch_epochs,
                "hp_search_mode": True
                }

        scheduler = ASHAScheduler(
                        max_t=num_hpsearch_epochs,
                        grace_period=1,
                        reduction_factor=2)

        trainable = tune.with_parameters(train_model_asha, num_gpus=gpus_per_trial)
        resources_per_trial = {"cpu": 8, "gpu": gpus_per_trial}

        tuner = tune.Tuner(
                tune.with_resources(
                    trainable,
                    resources=resources_per_trial
                ),
                tune_config=tune.TuneConfig(
                    metric="loss",
                    mode="min",
                    scheduler=scheduler,
                    num_samples=num_hpsearch_samples,
                ),
                run_config=air.RunConfig(
                    name="tune_vae_asha",
                    local_dir=root_dir
                ),
                param_space=config,
            )

        results = tuner.fit()

        print("Best hyperparameters found were: ", results.get_best_result().config)

        return
    
    _ = tune_model_asha(num_hpsearch_samples, num_hpsearch_epochs, gpus_per_trial)

###################
# Train VAE
###################

def train_vae_model(args):

    print("Loading HP search results ... ")
    root_dir = f"ray_results_{args.dataset}"

    restored_tuner = tune.Tuner.restore(f"{root_dir}/tune_vae_asha/", resume_unfinished=False, resume_errored=False)
    result_grid = restored_tuner.get_results()
    config = result_grid.get_best_result(metric="loss").config
    config["kl_anneal"] = False
    config["max_epochs"] = 100
    config["hp_search_mode"] = False

    train_dataloader = DataLoader(args.train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(args.test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)

    model = VAE(dataloader=train_dataloader, config=config, inv_transform=args.inv_transform, latent_dim=args.latent_dim, num_channels=args.num_channels)
    logger = TensorBoardLogger(save_dir=f"tensorboard_logs_{root_dir}")

    print("Training VAE ... ")

    trainer = Trainer(max_epochs=config["max_epochs"], devices=1, default_root_dir="./", log_every_n_steps=len(train_dataloader), logger=logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

############################
# Display HP search results
############################

def load_and_display_hp_search_results(args):

    print("Loading and displaying HP search results ...")
    root_dir = f"ray_results_{args.dataset}"

    restored_tuner = tune.Tuner.restore(f"{root_dir}/tune_vae_asha/", resume_unfinished=False, resume_errored=False)
    result_grid = restored_tuner.get_results()

    print(result_grid.get_dataframe(filter_metric="loss"))
    print(result_grid.get_best_result(metric="loss").config)

############################
# Get Reconstruction errors
############################

def get_only_reconstruction_errors(args):

    root_dir = f"ray_results_{args.dataset}"
    ckpt_path = glob.glob(f"./tensorboard_logs_{root_dir}/lightning_logs/version_0/checkpoints/*.ckpt")[0]
    
    model = VAE.load_from_checkpoint(ckpt_path)
    model.eval()
    print("Loaded model successfully")

    crit1 = nn.L1Loss(reduction='none')
    crit2 = nn.MSELoss(reduction='none')

    rec_loss_l2_list = []
    rec_loss_l1_list = []

    test_dataloader = DataLoader(args.test_dataset, batch_size=256, shuffle=False, drop_last=False)

    for idx, (img,label) in enumerate(test_dataloader):
        with torch.no_grad():
            mu, log_var, output = model.forward(img)

        rec_loss_l1_list.append(crit1(output.flatten(), img.flatten()).cpu().numpy())
        rec_loss_l2_list.append(crit2(output.flatten(), img.flatten()).cpu().numpy())

    rec_loss_l1_list = np.concatenate(rec_loss_l1_list)
    rec_loss_l2_list = np.concatenate(rec_loss_l2_list)

    print(f"Reconstruction loss L1: {np.mean(rec_loss_l1_list)}")
    print(f"Reconstruction loss L2: {np.mean(rec_loss_l2_list)}")

#####################################
# Perform Uncertainty Quantification
#####################################

def perform_uncertainty_quantification(args):

    print("Performing uncertainty quantification ...")
    root_dir = f"ray_results_{args.dataset}"
    ckpt_path = glob.glob(f"./tensorboard_logs_{root_dir}/lightning_logs/version_0/checkpoints/*.ckpt")[0]
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = VAE.load_from_checkpoint(ckpt_path).to(device)

    ################################################
    # Uncertainty Quantification on train samples
    ################################################

    model.eval()
    z_train = []
    for idx, (img,label) in enumerate(args.train_dataset):
        with torch.no_grad():
            mu, logvar, _ = model.forward(img[None,...].to(device))

        z1 = model.reparametrize(mu, logvar)
        z_train.append(z1)

    z_train = torch.cat(z_train).to(device)
    print(z_train.shape)

    ################################################
    # Uncertainty Quantification on test samples
    ################################################

    model.eval()
    z_test = []
    for idx, (img,label) in enumerate(args.test_dataset):
        with torch.no_grad():
            mu, logvar, _ = model.forward(img[None,...].to(device))

        z1 = model.reparametrize(mu, logvar)
        z_test.append(z1)

    z_test = torch.cat(z_test).to(device)
    print(z_test.shape)

    ################################################
    # Uncertainty Quantification on random samples
    ################################################
    num_zs = 5000
    z_rand = torch.randn(num_zs,args.latent_dim).to(device)

    list_of_zs = [z_rand, z_train[0:num_zs], z_test[0:num_zs]]
    list_of_znames = ["z_rand", "z_train", "z_test"]

    for z, z_name in zip(list_of_zs, list_of_znames):

        save_df = pd.DataFrame()

        num_zs = z.shape[0]
        variances = []
        save_imgs = []

        for idx in range(num_zs):
            z_curr = z[idx,:][None,...].to(device)

            model.eval() # we want dropout to be activated
            for m in model.modules(): # But we want to disable batchnorm
                if isinstance(m, nn.Dropout1d) or isinstance(m, nn.Dropout2d):
                    m.train()

            decoder_outputs = []
            num_dec_samples = 100
            with torch.no_grad():
                decoder_outputs = torch.cat([model.decode(z_curr) for _ in range(num_dec_samples)])

            model.eval()
            mu, _, outputs = model.forward(decoder_outputs)
            
            variances.append(outputs.var(0).mean().item())
            save_imgs.append(model.inv_transform(model.decode(z_curr)))

            print(idx, variances[-1])

        save_df["variance"] = variances
        save_df.to_csv(f"{z_name}.csv")

        save_imgs = torch.cat(save_imgs).cpu().detach().numpy()
        np.savez(f"{z_name}_images.npz", save_imgs)


if __name__ == "__main__":
    ###################
    # General setup
    ###################

    parser = argparse.ArgumentParser(description='Choose dataset between mnist or cifar10')
    parser.add_argument("--hp_search", action='store_true')
    parser.add_argument("--hp_search_loss", type=str, default="val/tot_loss")
    parser.add_argument("--num_hpsearch_samples", type=int, default=100)
    parser.add_argument("--num_hpsearch_epochs", type=int, default=20)

    parser.add_argument("--display_hp_search_results", action='store_true')

    parser.add_argument("--train_vae", action='store_true')

    parser.add_argument("--get_only_rec_error", action='store_true')

    parser.add_argument("--perform_uncertainty_quantification", action='store_true')

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--latent_dim", type=int, default=128)

    args = parser.parse_args()

    if args.dataset == "mnist":
        args.num_channels = 1
    elif args.dataset == "cifar10":
        args.num_channels = 3
    else:
        sys.exit("Choose one of mnist or cifar10.")

    if args.dataset == "mnist":
        # download and transform train dataset
        train_dataset = datasets.MNIST('./mnist_data', 
                                        download=True, 
                                        train=True,
                                        transform=transforms.Compose([
                                            transforms.Resize((32,32)),
                                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                                            transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                        ]))

        # download and transform test dataset
        test_dataset = datasets.MNIST('./mnist_data', 
                                        download=True, 
                                        train=False,
                                        transform=transforms.Compose([
                                            transforms.Resize((32,32)),
                                            transforms.ToTensor(), # first, convert image to PyTorch tensor
                                            transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                        ]))

        args.inv_transform = transforms.Compose([ transforms.Normalize(mean=(0.,),
                                                     std=(1/0.3081,)),
                                transforms.Normalize(mean=(-0.1307,),
                                                     std=(1.,)),
                               ])

    args.train_dataset = train_dataset
    args.test_dataset = test_dataset

    if args.hp_search:
        hp_search(args=args)
    elif args.display_hp_search_results:
        load_and_display_hp_search_results(args=args)
    elif args.train_vae:
        train_vae_model(args=args)
    elif args.get_only_rec_error:
        get_only_reconstruction_errors(args=args)
    elif args.perform_uncertainty_quantification:
        perform_uncertainty_quantification(args=args)
    else:
        sys.exit("Select an option.")