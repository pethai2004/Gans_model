# TODO : Fix tf variable creation in update generator (optimizer), try fix in model instead
# TODO : implement proper datasets batch in changing image size
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
from datasets import DatManipulator
from objective import generator_loss_wg, discrim_loss_wg, discrim_loss_0, generator_loss_0
from utils import Timer, DefaultConfig, AlphaDecay
ResultsTrainG = namedtuple("ResultsTraiG", ['g_loss', 'grad_g_norm', 'time_G'])
ResultsTrainD = namedtuple("ResultsTraiD", ['d_loss', 'grad_d_norm', 'd_g_loss', 'time_D'])

class Trainer:
	'''Trainer for generative adversarial model'''
	def __init__(self, G, D, global_batch=3000, log_dir="TrainGans01", stopdata=False, seed=5005, TrainDict=DefaultConfig, snap_short=True,
			mem_limit=None, dataset_dir="data", save_model=100, alpha_decay=True, save_freq=20):

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
		self.cur_img_size = None
		self.alpha_decay = alpha_decay if alpha_decay is None else {"D":None, "G":None}
		self.save_freq = save_freq

	def initialize_trainer(self):
		assert self.G.img_size == self.img_size, "G.img_size must be equal to Trainer.img_size"
		assert self.G.targ_img_size == self.targ_img_size, "G.targ_img_size must be equal to Trainer.targ_img_size"
		assert self.D.targ_img_size == self.targ_img_size, "D.targ_img_size must be equal to Trainer.targ_img_size"
		assert self.G.z_dim == self.z_dim, "G.z_dim must be equal to Trainer.z_dim"

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
			self.global_batch -= int(self.global_batch - self.global_batch % len(self.TrainDict.n_gpu))
			print("global_batch is not divisible by number of gpus, reset globa_batch to {}".format(self.global_batch))
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
		self.G.optimizer = self.G_opt
		self.D.optimizer = self.D_opt
		self.G.initialize_base()
		self.D.initialize_base()

		if self.TrainDict.train_ratio == None:
			bt = int(np.log2(self.targ_img_size[0])) - int(np.log2(self.img_size[0])) + 1
			start = int(np.log2(self.img_size[0]))
			n = [(int(self.TrainDict.epochs // bt))] * bt
			self.TrainDict.train_ratio = {2**i  : j for (i, j) in zip(range(start, start + bt), n)}
			print("set train_ratio to {} and neglect remaining epochs of {}".format(self.TrainDict.train_ratio, 
																		self.TrainDict.epochs % bt))
		# elif self.TrainDict.train_ratio != None:
		# 	assert np.sum([i for i in self.TrainDict.train_ratio.values()]) == self.TrainDict.epochs, "train_ratio must sum to epochs"
		with tf.device("cpu:0"): 
			self.generator_steps = tf.Variable(0, dtype=tf.int64, name="generator_steps")
			self.discriminator_steps = tf.Variable(0, dtype=tf.int64, name="discriminator_steps")
			self.global_train_steps = tf.Variable(0 , dtype=tf.int64, name="global_train_steps")
		self.cur_img_size = list(self.TrainDict.train_ratio)[0]
		if self.alpha_decay is not None:
			self.alpha_decay["G"] = AlphaDecay(self.TrainDict.epochs, lr_max=self.TrainDict.max_G_lr, lr_min=self.TrainDict.G_lr, decay_type=self.TrainDict.lr_schedual)
			self.alpha_decay["D"] = AlphaDecay(self.TrainDict.epochs, lr_max=self.TrainDict.max_D_lr, lr_min=self.TrainDict.D_lr, decay_type=self.TrainDict.lr_schedual)

	def set_lr_schedule(self, summary=True):
		if self.alpha_decay is not None:
			cur_g_lr = self.alpha_decay["G"](self.generator_steps.numpy())
			cur_d_lr = self.alpha_decay["D"](self.discriminator_steps.numpy())
			with self.strategy.scope():
				self.G_opt.learning_rate.assign(cur_g_lr)
				self.D_opt.learning_rate.assign(cur_d_lr)
			if summary:
				with self.summarizer.as_default():
					tf.summary.scalar("G_lr", self.G_opt.learning_rate, step=self.generator_steps)
					tf.summary.scalar("D_lr", self.D_opt.learning_rate, step=self.discriminator_steps)

	@tf.function
	def get_latent(self):
		normal = tfp.distributions.Normal(0, 1)
		return normal.sample(
            (int(self.global_batch / self.strategy.num_replicas_in_sync), self.z_dim)
        )
    
	@tf.function
	def g_train_step(self, true_img_k, z_k, trainable_variables):
		TM_G = Timer()
		with TM_G, tf.GradientTape() as G_tape, tf.name_scope("GeneratorLoss"):
			gen_L = generator_loss_0(z=z_k, G=self.G, D=self.D, train_sets=true_img_k)
		gen_G = G_tape.gradient(gen_L, trainable_variables)
		assert gen_G, "generator gradient is empty"
		if self.TrainDict.clip_norm:
			gen_G = tf.clip_by_global_norm(gen_G, self.TrainDict.clip_norm, name='clip_norm_G')
		self.G.update_params(gen_G)
		with tf.name_scope("GradientNorm"):
			grad_D_norm = tf.linalg.global_norm(gen_G)
		return ResultsTrainG(gen_L, grad_D_norm, TM_G.elapsed)
	
	@tf.function
	def d_train_step(self, true_img_k, z_k, trainable_variables):
		TM_D = Timer()
		with TM_D, tf.GradientTape() as tape_D, tf.name_scope("DiscriminatorLoss"):
			disc_L, disc_g_loss = discrim_loss_0(z=z_k, G=self.G, D=self.D, train_sets=true_img_k, g_penalty=self.TrainDict.grad_penalty)
			disc_G = tape_D.gradient(disc_L, trainable_variables)
		assert disc_G, "discriminator gradient is empty"
		if self.TrainDict.clip_norm:
			disc_G = tf.clip_by_global_norm(disc_G, self.TrainDict.clip_norm, name='clip_norm_D')
		if self.D_update_const > 0:
			with tf.name_scope("Discrim_grad_smooth_update"):
				raise NotImplementedError
		else:		
			self.D.update_params(disc_G)
		with tf.name_scope("GradientNorm"):
			grad_G_norm = tf.linalg.global_norm(disc_G)
		return ResultsTrainD(disc_L, disc_g_loss, grad_G_norm, TM_D.elapsed)

	def TRAIN(self, num_epochs=None):
		num_epochs = self.TrainDict.epochs if num_epochs == None else num_epochs
		for _ in range(num_epochs):
			self.train_step()

	def train_step(self):
		if self.TrainDict.switch_ratio["D"] == self.TrainDict.switch_ratio["G"]:
			data_k = next(self.data_handler)
		else : data_k = None
		for _ in range(self.TrainDict.switch_ratio["D"]):
			z_k = self.strategy.run(self.get_latent)
			if data_k is None: data_k = next(self.data_handler) 
			else: pass
			ResultsD = self.strategy.run(self.d_train_step, args=(data_k, z_k, self.D.get_trainable()))
			
			disc_L = tf.reduce_mean(self.strategy.experimental_local_results(ResultsD.d_loss))
			grad_D_norm = tf.reduce_mean(self.strategy.experimental_local_results(ResultsD.grad_d_norm))
			disc_g_loss = tf.reduce_mean(self.strategy.experimental_local_results(ResultsD.d_g_loss))
			time_D = tf.reduce_mean(self.strategy.experimental_local_results(ResultsD.time_D))

			with self.summarizer.as_default():
				if self.global_train_steps == 0:
					tf.summary.trace_export(name="graph_trace", step=0, profiler_outdir=self.log_dir)
				tf.summary.scalar("time_epoch_D", time_D, self.discriminator_steps)
				tf.summary.scalar("discriminator_loss", disc_L, self.discriminator_steps)
				tf.summary.scalar("grad_D_norm", grad_D_norm, self.discriminator_steps)
				tf.summary.scalar("disc_G_loss", disc_g_loss, self.discriminator_steps)
			self.discriminator_steps.assign_add(1)

		for _ in range(self.TrainDict.switch_ratio["G"]):
			z_k = self.strategy.run(self.get_latent)
			if data_k is None: data_k = next(self.data_handler) 
			else: pass
			ResultsG = self.strategy.run(self.g_train_step, args=(data_k, z_k, self.G.get_trainable()))

			gen_L = tf.reduce_mean(self.strategy.experimental_local_results(ResultsG.g_loss))
			grad_G_norm = tf.reduce_mean(self.strategy.experimental_local_results(ResultsG.grad_g_norm))
			time_G = tf.reduce_mean(self.strategy.experimental_local_results(ResultsG.time_G))

			with self.summarizer.as_default():
				if self.d_train_step == 0:
					tf.summary.trace_export(name="graph_trace", step=0, profiler_outdir=self.log_dir)
				tf.summary.scalar("time_epoch_D", time_G, self.generator_steps)
				tf.summary.scalar("generator_loss", gen_L, self.generator_steps)
				tf.summary.scalar("grad_G_norm", grad_G_norm, self.generator_steps)
				if self.snap_short:
					with tf.name_scope("SnapShort"):
						z_local = self.strategy.experimental_local_results(z_k)[0][:10]
						snp = self.G.forward_model(z_local)
					tf.summary.image("Snapshot", snp, self.global_train_steps)
			self.generator_steps.assign_add(1)

		self.global_train_steps.assign_add(1)
		if (self.generator_steps + self.discriminator_steps) % self.TrainDict.train_ratio[self.cur_img_size] == 0:
			if self.TrainDict.img_size != self.TrainDict.targ_img_size:
				self.G.auto_extend()
				# solve update step temperary after calling Generator.auto_extend()
				with self.strategy.scope():
					if isinstance(self.G_opt, tf.keras.optimizers.RMSprop):
						self.G_opt = tf.keras.optimizers.RMSprop(self.TrainDict.G_lr)
					elif isinstance(self.G_opt, tf.keras.optimizers.Adam):
						self.G_opt = tf.keras.optimizers.Adam(self.TrainDict.G_lr)
					else : self.G_opt = tf.keras.optimizers.SGD(self.TrainDict.G_lr)
				self.G.optimizer = self.G_opt
				self.cur_img_size *= 2
				self.data_handler.targ_img_size = self.cur_img_size

		self.set_lr_schedule()
		if self.global_train_steps % self.save_freq == 0:
			self.G.save_model(path=os.path.join(self.log_dir, "generator_checkpoint"), name="G_" + str(self.global_train_steps))
			self.D.save_model(path=os.path.join(self.log_dir, "discriminator_checkpoint"), name="D_" + str(self.global_train_steps))
		tf.keras.backend.clear_session()

		print("EPOCH : {} TIME : {} D_LOSS : {} G_LOSS : {}".format(self.global_train_steps.numpy(), 
				time_D + time_G, disc_L, gen_L))