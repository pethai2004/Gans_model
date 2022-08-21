import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt

import os, shutil
from collections import namedtuple
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from model import *
from datasets import DatManipulator, download_celeb_data, re_chunk_datasets, chunk_datasets
from objective import generator_loss_wg, discrim_loss_wg, discrim_loss_0, generator_loss_0
from utils import Timer, DefaultConfig

ResultsTrain = namedtuple("ResultsTrain", ['d_loss', 'g_loss', 'grad_d_norm', 'grad_g_norm', 'd_g_loss', 'time_D', 'time_G'])

class Trainer:

	def __init__(self, G, D, global_batch=3000, log_dir="TrainGans01", stopdata=False, seed=5005, TrainDict=DefaultConfig, snap_short=True,
			mem_limit=None, dataset_dir="data", save_model=100):

		assert isinstance(G, BaseGenerator), "G must be a subclass of BaseGenerator"
		assert isinstance(D, BaseDiscriminator), "D must be a subclass of BaseDiscriminator"
		self.G = G
		self.D = D
		self.z_dim = TrainDict.latent_z_dim
		self.img_size = TrainDict.img_size
		self.targ_img_size = TrainDict.targ_img_size
		self.global_batch = global_batch
		self.log_dir = log_dir
		self.stopdata = stopdata
		self.seed = seed
		self.TrainDict = TrainDict
		self.snap_short = snap_short
		self.mem_limit = mem_limit
		assert mem_limit == None, "mem_limit is not compatible for now, allocate full memory"
		self.strategy = None
		self.data_handler = None
		self.dataset_dir = dataset_dir
		self.D_update_const = 0.0
		self.global_train_steps = None
		self.D_opt = None
		self.G_opt = None
		self.summarizer = None
		self.save_model = save_model
		self.cur_up_size = None

	def initialize_trainer(self):
		assert self.G.img_size == self.img_size, "G.img_size must be equal to Trainer.img_size"
		assert self.G.targ_img_size == self.targ_img_size, "G.targ_img_size must be equal to Trainer.targ_img_size"
		assert self.D.targ_img_size == self.targ_img_size, "D.targ_img_size must be equal to Trainer.targ_img_size"

		tf.random.set_seed(self.seed)
		np.random.seed(self.seed)
		self.available_gpus = tf.config.list_physical_devices("GPU")
		# assert all(self.TrainDict.n_gpu) in self.available_gpus, "provided DefaultConfig.n_gpu do not currently available"
		if self.mem_limit:
			for each_gpu in self.available_gpus:
				tf.config.experimental.set_memory_growth(each_gpu, 
					[tf.config.LogicalDeviceConfiguration(memory_limit=self.mem_limit)])
		self.strategy = tf.distribute.MirroredStrategy(self.TrainDict.n_gpu)
		if self.global_batch % len(self.TrainDict.n_gpu) != 0:
			self.global_bach -= int(self.global_batch - self.global_batch % len(self.TrainDict.n_gpu))
			print("global_batch is not divisible by number of gpus, reset globa_batch to {}".format(self.global_bach))
		self.per_replica_batch = int(self.global_batch / len(self.TrainDict.n_gpu))
		self.data_handler = DatManipulator(self.global_batch, self.dataset_dir, self.strategy, 
				self.targ_img_size, chunk_first=False, non_stop=self.stopdata)
		if os.path.isdir(self.log_dir):
			shutil.rmtree(self.log_dir) # remove past existing log directory
		self.summarizer = tf.summary.create_file_writer(self.log_dir)
		tf.summary.trace_on(graph=True, profiler=True)
		with self.strategy.scope():
			if self.TrainDict.optimizer == "RMSprop":
				self.D_opt = keras.optimizers.RMSprop(self.TrainDict.D_lr)
				self.G_opt = keras.optimizers.RMSprop(self.TrainDict.G_lr)
			elif self.TrainDict.optimizer == "Adam":
				self.D_opt = keras.optimizers.Adam(self.TrainDict.D_lr)
				self.G_opt = keras.optimizers.Adam(self.TrainDict.G_lr)
			elif self.TrainDict.optimizer == "SGD":
				self.D_opt = keras.optimizers.SGD(self.TrainDict.D_lr)
				self.G_opt = keras.optimizers.SGD(self.TrainDict.G_lr)
		self.G.strategy_scope = self.strategy
		self.D.strategy_scope = self.strategy
		self.G.initialize_base()
		self.D.initialize_base()

		if self.TrainDict.train_ratio == None and self.G.extendable:
			self.TrainDict.train_ratio = self.TrainDict.epochs // (self.targ_img_size[0] / self.img_size[0])
			print("set train_ratio to {} and neglect remaining epochs of {}".format(self.TrainDict.train_ratio, 
																		self.TrainDict.epochs % (self.targ_img_size[0] / self.img_size[0])))
			self.cur_up_size = 0	
		elif self.TrainDict.train_ratio != None:
			assert np.sum(self.TrainDict.train_ratio) == self.TrainDict.epochs, "train_ratio must sum to epochs"
		self.global_train_steps = tf.Variable(0, dtype=tf.float64, name="global_training_steps")
	def set_lr_schedule(self):
		pass
		
	@tf.function
	def get_latent(self):
		normal = tfp.distributions.Normal(0, 1)
		return normal.sample(
            (int(self.global_batch / self.strategy.num_replicas_in_sync), self.z_dim)
        )
    
	@tf.function
	def train_step(self, true_img_k, z_k):
		TM_D = Timer()
		TM_G = Timer()
		def forw_D(): # use function for now since there is a problem with allocating memory not released
			with TM_D:
				with tf.GradientTape() as D_tape, tf.name_scope("DiscriminatorLoss"):
					disc_L, disc_g_loss = discrim_loss_0(z=z_k, G=self.G, D=self.D, train_sets=true_img_k, g_penalty=self.TrainDict.grad_penalty)
				disc_G = D_tape.gradient(disc_L, self.D.model.trainable_variables)
			if self.TrainDict.clip_norm:
				disc_G = tf.clip_by_global_norm(disc_G, self.TrainDict.clip_norm, name='clip_norm_D')
			if self.D_update_const > 0:
				with tf.name_scope("Discrim_grad_smooth_update"):
					raise NotImplementedError
			else:		
				self.D_opt.apply_gradients(zip(disc_G, self.D.model.trainable_variables))
			with tf.name_scope("GradientNorm"):
				grad_G_norm = tf.linalg.global_norm(disc_G)
			return disc_L, disc_g_loss, grad_G_norm

		def forw_G():
			with TM_G:
				with tf.GradientTape() as G_tape, tf.name_scope("GeneratorLoss"):
					gen_L = generator_loss_0(z=z_k, G=self.G, D=self.D, train_sets=true_img_k)
				gen_G = G_tape.gradient(gen_L, self.G.model.trainable_variables)
			if self.TrainDict.clip_norm:
				gen_G = tf.clip_by_global_norm(gen_G, self.TrainDict.clip_norm, name='clip_norm_G')

			self.G_opt.apply_gradients(zip(gen_G, self.G.model.trainable_variables))
		
			with tf.name_scope("GradientNorm"):
				grad_D_norm = tf.linalg.global_norm(gen_G)
			return gen_L, grad_D_norm

		disc_L, disc_g_loss, grad_G_norm = forw_D()
		gen_L, grad_D_norm = forw_G()
		
		return ResultsTrain(disc_L, gen_L, grad_D_norm, grad_G_norm, disc_g_loss, TM_D.elapsedH, TM_G.elapsedH)

	def train(self):
		for _ in range(self.TrainDict.epochs):
			z_k = self.strategy.run(self.get_latent)
			Results = self.strategy.run(self.train_step, args=(next(self.data_handler), z_k))
			
			disc_L = tf.reduce_mean(self.strategy.experimental_local_results(Results.d_loss))
			gen_L = tf.reduce_mean(self.strategy.experimental_local_results(Results.g_loss))
			grad_D_norm = tf.reduce_mean(self.strategy.experimental_local_results(Results.grad_d_norm))
			grad_G_norm = tf.reduce_mean(self.strategy.experimental_local_results(Results.grad_g_norm))
			disc_g_loss = tf.reduce_mean(self.strategy.experimental_local_results(Results.d_g_loss))

			with self.summarizer.as_default():
				if self.global_train_steps == 0:
					tf.summary.trace_export(name="graph_trace", step=0, profiler_outdir=self.log_dir)
				tf.summary.scalar("time_epoch_D", Results.time_D, self.global_train_steps)
				tf.summary.scalar("time_epoch_G", Results.time_G, self.global_train_steps)
				tf.summary.scalar("generator_loss", gen_L, self.global_train_steps)
				tf.summary.scalar("discriminator_loss", disc_L, self.global_train_steps)
				tf.summary.scalar("grad_D_norm", grad_D_norm, self.global_train_steps)
				tf.summary.scalar("grad_G_norm", grad_G_norm, self.global_train_steps)
				tf.summary.scalar("disc_G_loss", disc_g_loss, self.global_train_steps)
				
				if self.snap_short:
					with tf.name_scope("SnapShort"):
						snp = self.G.forward_model(z_k[:10])
					tf.summary.image("Snapshot", snp, self.global_train_steps)
			
			self.global_train_steps.assign_add(1)
			if self.global_train_steps % self.save_model == 0 :
				pass
			if self.TrainDict.train_ratio[self.cur_up_size] % self.global_train_steps == 0 and self.G.extendable:
				self.cur_up_size += 1
				self.G.auto_extend()
				if self.D.extendable:
					self.D.auto_extend()
				self.data_handler.targ_img_size = self.cur_up_size
			print("EPOCH : {} TIME : {} D_LOSS : {} D_LOSS : {}".format(self.global_train_steps.numpy(), 
                    Results.time_D + Results.Time_G, disc_L, gen_L))
			tf.keras.backend.clear_session()