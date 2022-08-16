import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt

import os
from collections import namedtuple
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from model import *
from datasets import DatManipulator, download_celeb_data, re_chunk_datasets, chunk_datasets
from objective import generator_loss_wg, discrim_loss_wg, discrim_loss_0, generator_loss_0
from utils import Timer

ResultsTrain = namedtuple("ResultsTrain", ['d_loss', 'g_loss', 'grad_d_norm', 'grad_g_norm', 'd_g_loss', 'time_epoch'])

@tf.function
def train_step(global_train_step, true_img_k, D, G, z_k, clip_norm=3, log_dir="TrainLog"):
    
    TM = Timer()
    TM.start()

    with tf.GradientTape() as D_tape, tf.name_scope("DiscriminatorLoss"):
        disc_L, disc_g_loss = discrim_loss_0(z=z_k, G=G, D=D, train_sets=true_img_k, g_penalty=grad_penalty)
    disc_G = D_tape.gradient(disc_L, D.model.trainable_variables)
    
    with tf.GradientTape() as G_tape, tf.name_scope("GeneratorLoss"):
        gen_L = generator_loss_0(z=z_k, G=G, D=D, train_sets=true_img_k)
    gen_G = G_tape.gradient(gen_L, G.model.trainable_variables)
    TM.stop()

    if clip_norm:
        disc_G = tf.clip_by_global_norm(disc_G, clip_norm, name='clip_norm_D')
        gen_G = tf.clip_by_global_norm(gen_G, clip_norm, name='clip_norm_G')
        
    D_opt.apply_gradients(zip(disc_G, D.model.trainable_variables))
    G_opt.apply_gradients(zip(gen_G, G.model.trainable_variables))
   	
   	with tf.name_scope("GradientNorm"):
	    grad_G_norm = tf.linalg.global_norm(gen_G)
	    grad_D_norm = tf.linalg.global_norm(disc_G)
    	
    return ResultsTrain(disc_L, gen_L, grad_D_norm, grad_G_norm, disc_g_loss,  TM.elapsedH)

def RunTraining(epochs=10, img_size=(32, 32), global_batch=3000, num_gpu=2, limit_mem=None, targ_img_size=(128, 128), 
		data="celebA", log_dir="train_log_gans", stopdata=False, seed=5050, G=None, D=None, G_lr=0.0001, D_lr=0.0001,
		G_dict="default", D_dict="default", schedule=None, snap_short=True, num_chunk=None):
	
	tf.random.set_seed(seed)
	np.random.seed(seed)
	# tf.debugging.set_log_device_placement(True)
	gpus = tf.config.list_physical_devices("GPU")
	assert num_gpu <= len(gpus), "provided num_gpu is less than available GPUs"
	if global_batch % num_gpu != 0:
		global_batch -= global_batch - (global_batch // num_gpu)
		print("global_batch is not divisible by num_gpu, resize global_batch to {}".format(global_batch))
	per_replica_batch = int(global_batch / num_gpu)
	strategy = tf.distribute.MirroredStrategy(["GPU:{}".format(i) for i in range(num_gpu)])

	if data == "celebA":
		download_celeb_data()
		path_cb = os.path.join(log_dir, "data_celeb")
		os.mkdir(path_cb)
	if targ_img_size is None:
		targ_img_size = img_size # static image size
	if num_chunk is None:
		
	chunk_datasets(path_cb, num=8, del_left=True)
	# re_chunk_datasets(path, num=5, del_left=True, suf_name='fold_chunk_')
	DataInstance = DatManipulator(global_batch, path_cb, strategy=strategy, target_image_size=targ_img_size, 
                             chunk_first=False, non_stop=stopdata)
	latent_z_dim = 200
	if G is None and D is None:
		G = Generator_v1(image_size=img_size, targ_img_size=targ_img_size, up_type="deconv", 
				latent_space=latent_z_dim, seed=seed, strategy_scope=strategy)
		D = Discriminator_v1(image_size=img_size, targ_img_size=targ_img_size, 
				non_extendable=True, seed=seed, strategy_scope=strategy)

	G.initialize_base(filters=G_dict["filters"])
	D.initialize_base(filters=D_dict["filters"], units_dense=D_dict["units_dense"], act_out=D_dict["act_out"])

	with strategy.scope():
		G_opt = keras.optimizers.RMSprop(G_lr)
		D_opt = keras.optimizers.RMSprop(D_lr)

	summary_log_dir = os.path.join(log_dir, "TrainLog")
	z_k_dist = tfp.distributions.Normal(0, 0.1)

	for eps in range(epochs):

		z_k = z_k_dist.sample(global_batch, latent_z_dim)
		Results = strategy.run(trainstep, arg=(eps, next(DataInstance), D, G, z_k, 3, summary_log_dir, True))

		disc_L = tf.reduce_mean(strategy.experimental_local_results(Results.d_loss))
		gen_L = tf.reduce_mean(strategy.experimental_local_results(Results.g_loss))
		grad_D_norm = tf.reduce_mean(strategy.experimental_local_results(Results.grad_d_norm))
		grad_G_norm = tf.reduce_mean(strategy.experimental_local_results(Results.grad_g_norm))
		disc_g_loss = tf.reduce_mean(strategy.experimental_local_results(Results.d_g_loss))

		tf.summary.scalar("time_epoch", Results.time_epoch, eps)
		tf.summary.scalar("generator_loss", gen_L, eps)
		tf.summary.scalar("discriminator_loss", disc_L, eps)
		tf.summary.scalar("grad_D_norm", grad_D_norm, eps)
		tf.summary.scalar("grad_G_norm", grad_G_norm, eps)
		tf.summary.scalar("disc_G_loss", disc_g_loss, eps)

		if snap_short:
			with tf.name_scope("SnapaShort"):
				snp = G.forward_model(z_k)
			tf.summary.image("Snapshot", snp, eps)


