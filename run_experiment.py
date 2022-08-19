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
from utils import Timer, DefaultConfig

ResultsTrain = namedtuple("ResultsTrain", ['d_loss', 'g_loss', 'grad_d_norm', 'grad_g_norm', 'd_g_loss', 'time_epoch'])

class Trainer:

	def __init__(self, z_dim=200, img_size=32, targ_img_size=128, global_batch=3000,
		log_dir="TrainGans", stopdata=False, seed=5005, TrainDict=DefaultConfig, snap_short=True,
		num_chunk=10, clip_norm=3, mem_limit=20000, dataset_dir="data"):
	
		self.G = dict()
		self.D = dict()
		self.num_model = len(self.G)
		self.z_dim = z_dim
		self.img_size = img_size
		self.targ_img_size = targ_img_size
		self.global_batch = global_batch
		self.log_dir = log_dir
		self.stopdata = stopdata
		self.seed = seed
		self.TrainDict = TrainDict
		self.snap_short = snap_short
		self.num_chunk = num_chunk
		self.clip_norm = clip_norm
		self.mem_limit = mem_limit
		self.strategy = None
		self.load_data_fn = None
		self.data_handler = None
		self.dataset_dir = dataset_dir
		self.D_update_const = 0.4
		self.global_train_steps = tf.Variable(0, dtype=tf.float64, name="global_training_steps")
		self.D_opt = None
		self.G_opt = None

	def register_model(self, G, D, name=None):
		assert issubclass(BaseGenerator, G), "G must be a subclass of BaseGenerator"
		assert issubclass(BaseDiscriminator, D), "D must be a subclass of BaseDiscriminator"
		if name is not None:
			assert name not in self.G.keys(), "name already exists"
		else :
			name = "registered_model_00" + str(self.num_model)
		self.num_model += 1
		self.G[name] = G
		self.D[name] = D
		
	def initialize_trainer(self):
		tf.random.set_seed(self.seed)
		np.random.seed(self.seed)
		self.available_gpus = tf.config.list_physical_devices("GPU")
		assert all(self.TrainDict["n_gpu"]) in self.available_gpus, "provided DefaultConfig.n_gpu do not currently available"
		if self.mem_limit:
			for each_gpu in self.TrainDict["n_gpu"]:
				tf.config.experimental.set_memory_growth(each_gpu, 
					[tf.config.LogicalDeviceConfiguration(memory_limit=self.mem_limit)])
		self.strategy = tf.distribute.MirroredStrategy(self.TrainDict["n_gpu"])
		if self.global_batch % len(self.TrainDict["n_gpu"]) != 0:
			self.global_bach -= int(self.global_batch - self.global_batch % len(self.TrainDict["n_gpu"]))
			print("global_batch is not divisible by number of gpus, reset globa_batch to {}".format(self.global_bach))
		self.per_replica_batch = int(self.global_batch / len(self.TrainDict["n_gpu"]))
		self.data_handler = DatManipulator(self.global_batch, self.dataset_dir, self.strategy, 
			(self.targ_img_size, self.tar_img_size), chunk_first=False, non_stop=self.stopdata)
		
		for each_model in self.G.values(): each_model.initialize_base()
		for each_model in self.D.values(): each_model.initialize_base()
		with self.strategy.scope():
			self.D_opt = keras.optimizers.RMSprop(self.TrainDict.D_lr)
			self.G_opt = keras.optimizers.RMSprop(self.TrainDict.G_lr)
	@tf.function
	def train_step(global_train_step, true_img_k, D, G, z_k, clip_norm=3):
    
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

	def train(self):
    		
		z_k = z_k_dist.sample(global_batch, latent_z_dim)
		Results = strategy.run(trainstep, arg=(eps, next(DataInstance), D, G, z_k, 3))

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