from model import *
from utils import DefaultConfig, pre_build_G, pre_build_D
from run_experiment import Trainer
from datasets import *
from run_schedule import TrainerConfigs, DiscriminatorConfigs, GeneratorConfigs

from multiprocessing import Process
# download_celeb_data()
# chunk_datasets(dir_path="data/img_align_celeba", num=8, del_left=True)

def run_exp(log_dir, dataset_dir, train_cf, d_cf, g_cf):

    G_ = Generator_v0(img_size=train_cf.img_size, 
            targ_img_size=train_cf.targ_img_size, 
            z_dim=train_cf.latent_z_dim,
            apply_resize=train_cf.applied_G_method,
                configbuild=g_cf)

    D_ = Discriminator_v1(img_size=train_cf.img_size,
                targ_img_size=train_cf.targ_img_size,
                apply_resize=train_cf.applied_D_method,
                    configbuild=d_cf)

    trainer = Trainer(G_, D_, 
                global_batch=50, 
                dataset_dir=dataset_dir,
                alpha_decay=True, 
                log_dir=log_dir,
                    TrainDict=train_cf)

    trainer.initialize_trainer()
    trainer.TRAIN()

if __name__ == "__main__":
    print("Initialize subprocess")
    dataset_dir = 'data/img_align_celeba'

    processes = []
    for i in range(len(TrainerConfigs)):
        log_dir = "TrainLogs" + str(i).zfill(3)
        processes.append(
            Process(target=run_exp, args=(log_dir, 
                                        dataset_dir, 
                                        TrainerConfigs[i],
                                        DiscriminatorConfigs[i],
                                        GeneratorConfigs[i]), name="subprocess".zfill(3)
                        )
                    )
    print("set up subprocess")

    for p in processes:
        print("start subprocess for {}".format(p.name))
        p.start()
        p.join()