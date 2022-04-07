import yaml
import argparse
import sys
import pandas as pd
# import umap
# import umap.plot
import torch
import numpy as np
from shutil import copyfile

from models import *
from experiment import RNAExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Argument Parser
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--save', '-s', action='store_true', dest='save', default=False)
parser.add_argument('--umap', '-u', action='store_true', dest='um', default=False)
parser.add_argument('--beta', type=float)
parser.add_argument('--h_dim')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Logs to the local filesystem given by 'save_dir' & 'name'
tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False
torch.cuda.empty_cache()

# Sets-up the VAE Model through 'VAEXperiment' into 'experiment'
if args.beta:
    config['model_params']['beta'] = args.beta
if args.h_dim:
    if args.h_dim == "None":
        config['model_params']['hidden_dims'] = None
    else:
        config['model_params']['hidden_dims'] = [int(x) for x in args.h_dim.split(' ')]

print("BETA BETA BETA BETA",config['model_params']['beta'], config['model_params']['latent_dim'], config['model_params']['hidden_dims'])
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = RNAExperiment(model, config['exp_params'])
print(experiment)

runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                 min_epochs=1,
                 logger=tt_logger,
                 num_sanity_val_steps=5,
                 callbacks=[EarlyStopping(monitor="val_loss", patience=3, verbose=False)],
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)

# When model is trained.
train_data = experiment.dataobject.get_dataset('trainval').data
test_data = experiment.dataobject.get_dataset('test').data
fc_data = experiment.dataobject.get_dataset('fc').data

y = experiment.dataobject.get_dataset('testy')

# Save log.
target = tt_logger.save_dir + tt_logger.name + "/version_" + str(tt_logger.experiment.version) + "/"


def final_encode(data):
    x = model.encode(torch.from_numpy(data))
    writing = x[0].detach().numpy()
    return writing


def save_latent(writing, split):
    np.save(config['exp_params']['out_path'] + config['model_params']['name'] + "/latent_features_{}_beta{}_h_dim{}.npy"
            .format(split, config['model_params']['beta'], config['model_params']['hidden_dims']), writing)


# Get latent space.
latent_train = final_encode(train_data)
latent_test = final_encode(test_data)
latent_fc = final_encode(fc_data)


if args.save:
    print("Writing latent features...")
    # Save latent space.
    save_latent(latent_train, "trainval")
    save_latent(latent_test, "test")
    save_latent(latent_fc, "fc")
    torch.save(model.state_dict(), config['exp_params']['out_path'] + config['model_params']['name'] + "/beta{}_h_dim{}.pth"
               .format(config['model_params']['beta'], config['model_params']['hidden_dims']))


if args.um:
    mapper = umap.UMAP(
                    n_neighbors=15,
                    min_dist=0.1,
                    n_components=2,
                    metric='euclidean'
                ).fit(latent_test)

    # y = np.load(config['exp_params']['data_path'] + config['exp_params']['data_path'].split("/")[-2] + "_train.npy")
    # y = np.load(config['exp_params']['data_path'] + "types_test.npy")
    p = umap.plot.points(mapper, color_key_cmap="Paired", background="white")
    umap.plot.plt.title("Lindel Test - Beta: {}, Latent_dim: {}, Hidden Dims : {}".format(config['model_params']['beta'], config['model_params']['latent_dim'], config['model_params']['hidden_dims']))
    # umap.plot.plt.show()
    umap.plot.plt.savefig(config['exp_params']['out_path'] + "/VanillaVAE/umap_test_beta{}_h_dim{}.png".format(config['model_params']['beta'], config['model_params']['hidden_dims']), dpi=800)

    mapper = umap.UMAP(
                    n_neighbors=15,
                    min_dist=0.1,
                    n_components=2,
                    metric='euclidean'
                ).fit(latent_fc)

    # y = np.load(config['exp_params']['data_path'] + config['exp_params']['data_path'].split("/")[-2] + "_train.npy")
    # y = np.load(config['exp_params']['data_path'] + "types_test.npy")
    p = umap.plot.points(mapper, color_key_cmap="Paired", background="white")
    umap.plot.plt.title("ForeCasT Test (10000+ samples) - Beta: {}, Latent_dim: {}, Hidden Dims : {}".format(config['model_params']['beta'], config['model_params']['latent_dim'], config['model_params']['hidden_dims']))
    # umap.plot.plt.show()
    umap.plot.plt.savefig(config['exp_params']['out_path'] + "/VanillaVAE/umap_fc_beta{}_h_dim{}.png".format(config['model_params']['beta'], config['model_params']['hidden_dims']), dpi=800)


# Save initialization yaml to log directory.
copyfile(args.filename, target + args.filename.split("/")[-1])
