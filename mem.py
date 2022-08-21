from model import *
from datasets import download_celeb_data, chunk_datasets, re_chunk_datasets, DatManipulator
from objective import generator_loss_0, discrim_loss_0
from utils import pre_build_G, pre_build_D, DefaultConfig
from run_experiment import Trainer

# download_celeb_data()
# chunk_datasets(dir_path="data/img_align_celeba", num=5, del_left=True)

train_config = DefaultConfig
g_config = pre_build_G
d_config = pre_build_D

train_config.epochs = 100
train_config.n_gpu = ["/GPU:0", "/GPU:1", "/GPU:2", "/GPU:3"]
train_config.clip_norm = None

img_size = (32, 32)
targ_img_size = (128, 128)
global_batch = 1000

generator = Generator_v0(img_size, 
                         targ_img_size, 
                         train_config.latent_z_dim,
                         up_type='deconv',
                         name='generator_test',
                        apply_resize=True)
generator.configbuild = g_config

discriminator = Discriminator_v0(img_size,
                                targ_img_size,
                                train_config.latent_z_dim,
                                name="discriminator_test",
                                extendable=False,
                                resizing=True)
discriminator.configbuild = d_config

trainer = Trainer(G=generator,
                 D=discriminator,
                 global_batch=global_batch,
                 log_dir="TrainGans01",
                 stopdata=True,
                 seed=1001,
                 TrainDict=train_config,
                 snap_short=True, 
                 dataset_dir="data/img_align_celeba")

trainer.initialize_trainer()
trainer.train()