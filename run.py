from model import *
from utils import DefaultConfig, pre_build_G, pre_build_D
from run_experiment import Trainer
from datasets import *

# download_celeb_data()
# chunk_datasets(dir_path="data/img_align_celeba", num=8, del_left=True)

train_config = DefaultConfig
train_config.epochs = 300
train_config.clip_norm = None
train_config.switch_ratio = {"G" : 1, "D" : 1}
g_config = pre_build_G
g_config["out_units"] = 200
g_config["num_layers"] = 4
g_config["dense_units"] = 200
g_config['kernel_size'] = (4, 4)
d_config = pre_build_D
d_config['out_units'] = 150
train_config.img_size=(64, 64)
train_config.targ_img_size=(64,64)
train_config.n_gpu = ["gpu:0", "gpu:1", "gpu:2", "gpu:3"]
train_config.G_lr = 0.000002
train_config.D_lr = 0.000002
train_config.max_G_lr = 0.0001
train_config.max_D_lr = 0.00005
train_config.lr_schedule = 'cosine'
train_config.optimizer = "RMSprop"

G_ = Generator_v0(img_size=train_config.img_size, 
                  targ_img_size=train_config.targ_img_size, 
                  z_dim=128, 
                  apply_resize=True)

D_ = Discriminator_v1(img_size=train_config.img_size,
                     targ_img_size=train_config.targ_img_size)

g_config["BaseFilters"] = [200, 300, 500, 500, 400, 200]

G_.configbuild = g_config
D_.configbuild = d_config
G_.z_dim = train_config.latent_z_dim
# G_.initialize_base()
# D_.initialize_base()

trainer = Trainer(G_, D_, 
                global_batch=290, 
                dataset_dir='data/img_align_celeba', 
                alpha_decay=True)

trainer.TrainDict = train_config
trainer.initialize_trainer()
trainer.TRAIN()