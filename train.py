
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

import numpy as np
import os, shutil
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from objective import generator_loss_wg, discrim_loss_wg, discrim_loss_0, generator_loss_0
from utils import TrainLog, Timer, Logger
from datasets import DataHandler
from model import Generator, Discriminator

class Trainer:
	'''Trainer for generative adversarial model'''
	def __init__(self, train_config, build_config, log_dir="TrainGans", dataset_dir='data', trace_graph=True, save_shots=1000, save_interval=1000):
		
		self.G = None
		self.D = None
		self.log_dir = log_dir
		self.dataset_dir = dataset_dir
		self.train_config = train_config
		self.build_config = build_config
		self.build_config.update({"img_size" : train_config.img_size})
		self.build_config.update({"targ_img_size" : train_config.targ_img_size})
		print("set img_size to {} and targ_img_size to {}".format(
      				train_config.img_size, train_config.targ_img_size))
		self.strategy = None
		self.data_handler = None
		
		self.global_train_steps = None
		self.generator_steps = None
		self.discriminator_steps = None
		self.D_opt = None
		self.G_opt = None
		self.summarizer = None
		self.graph_and_trace = trace_graph
		self.save_shots = save_shots
		self.save_interval = save_interval
		self.distribution = tfp.distributions.Normal(0, 1)
		self.g_loss_fn = "gen"
		self.d_loss_fn = "gen"
		self.fix_latent = None
  
	def initialize_trainer(self):
     
		tf.random.set_seed(self.train_config.seed)
		np.random.seed(self.train_config.seed)
		# update build_config to match train_config 
		_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=self.train_config.seed)
		self.build_config.update({"initializer" : _initializer})

		#check for global batch
		if self.train_config.global_batch % len(self.train_config.n_gpu) != 0:
			self.train_config.global_batch -= int(self.train_config.global_batch - self.train_config.global_batch % len(self.train_config.n_gpu))
			print("train_config.global_batch is not divisible by number of gpus, reset globa_batch to {}".format(self.train_config.global_batch))
		self._available_gpu = tf.config.list_physical_devices('GPU')
		if len(self._available_gpu) < len(self.train_config.n_gpu):
			print("Number of available gpus is less than number of gpus specified in train_config.n_gpu")
			print("Available Physical GPUs: {}".format(self._available_gpu))
			print("Provided GPUs: {}".format(self.train_config.n_gpu))
   
		self.strategy = tf.distribute.MirroredStrategy(self.train_config.n_gpu)
		self._per_replica_batch = int(self.train_config.global_batch / len(self.train_config.n_gpu))
		self.data_handler = DataHandler(self.dataset_dir, 
									self.strategy, 
									self.train_config.global_batch, 
									init_res=self.train_config.img_size[0],
									fin_res=self.train_config.targ_img_size[0])

		print("DataHandler is initialized successfully for dataset : {}".format(self.dataset_dir))
		print("Found subdirectories: {}".format(self.data_handler._num_chunk))
		self.summarizer = Logger(self.log_dir, self.strategy)
		print("Initialized Logger for logging to {}".format(self.summarizer.log_dir))
		self.log_dir = self.summarizer.log_dir # set log_dir to actual log_dir if it is already existing
		if self.graph_and_trace : tf.summary.trace_on(graph=True, profiler=True)

		with tf.device("cpu:0"): # create variable step seperately
			self.generator_steps = tf.Variable(0, dtype=tf.int64, name="generator_steps")
			self.discriminator_steps = tf.Variable(0, dtype=tf.int64, name="discriminator_steps")
			self.global_train_steps = tf.Variable(0 , dtype=tf.int64, name="global_train_steps")

		# setting up loss function for generative and discriminative model
		if self.g_loss_fn == "wgan": self.g_loss_fn = generator_loss_wg
		else: self.g_loss_fn = generator_loss_0
		if self.d_loss_fn == "wgan": self.d_loss_fn = discrim_loss_wg
		else: self.d_loss_fn = discrim_loss_0
		# setting up optimizer for generative and discriminative model
		with self.strategy.scope():
			self.D_opt = self.train_config.D_optimizer(**self.train_config.D_opt_config)
			self.G_opt = self.train_config.G_optimizer(**self.train_config.G_opt_config)
		# construct generator and discriminator
		self.G = Generator(self.build_config, "g_model", self.strategy)
		self.D = Discriminator(self.build_config, "d_model", self.strategy)
		with tf.name_scope("GeneratorNet") : self.G.initialize_base() # already initialize in strategy scope of strategy
		with tf.name_scope("DiscriminatorNet") : self.D.initialize_base() 
		# Need to have optimizer create variable first since which will solve the tf.Variable creation
		_trainable_g = self.G.get_trainable(targ_res=self.train_config.targ_img_size[0]) # receive full trainable
		_trainable_d = self.D.get_trainable(targ_res=self.train_config.targ_img_size[0])
		@tf.function
		def _fOpt_g(): self.G_opt._create_all_weights(_trainable_g)
		@tf.function
		def _fOpt_d(): self.D_opt._create_all_weights(_trainable_d)
		self.strategy.run(_fOpt_g)
		self.strategy.run(_fOpt_d)
		self.fix_latent = self.distribution.sample(self.G.input_shape[-1])
		self.G._get_easy_model(self._curr_img_size) # set back to initilial state of input
		self.D._get_easy_model(self._curr_img_size)
  
	def _get_fix_lt(self, batch_size):
		return tf.convert_to_tensor([self.fix_latent for _ in range(batch_size)]) 

	def set_lr_schedule(self, summary=True):
		self.train_config.D_lr = self.train_config.G_lr_schedule(self.discriminator_steps)
		self.train_config.G_lr = self.train_config.G_lr_schedule(self.generator_steps)
		self.D_opt.learning_rate = self.train_config.D_lr
		self.G_opt.learning_rate = self.train_config.G_lr
		if summary:
			self.summarizer.log_scalar("D_lr", self.D_opt.learning_rate, self.discriminator_steps, reduce=None)
			self.summarizer.log_scalar("G_lr", self.G_opt.learning_rate, self.generator_steps, reduce=None)
    
	def _per_replica_latent(self, batch_size, output_shape):
		'''Call inside the scope of strategy'''
		_single_sampled = []
		for shape_out in output_shape:
			_single_sampled.append(self.distribution.sample((batch_size, *shape_out)))
		_single_sampled.append(self._get_fix_lt(batch_size))
		return _single_sampled

	@tf.function
	def d_train_step(self, x, var_train):
		d_timer = Timer()
		z_k = self._per_replica_latent(x.shape[0], self.D.input_shape[:-1])
		with tf.GradientTape() as d_tape, d_timer:
			lossD, lossgradD, args_v = self.d_loss_fn(z_k, self.G, self.D, x)
		gradD = d_tape.gradient(lossD, var_train)
		assert gradD, "discriminator gradient is None"
		self.G_opt.apply_gradients(zip(gradD, var_train))
		normgradD= tf.linalg.global_norm(gradD)
		return TrainLog(D_loss=lossD, Dg_loss=lossgradD, Dg_norm=normgradD, Dg_time=d_timer.elapsed)
	
	@tf.function
	def g_train_step(self, x, var_train):
		g_timer = Timer()
		z_k = self._per_replica_latent(x.shape[0], self.G.input_shape[:-1])
		with tf.GradientTape() as g_tape, g_timer:
			lossG = self.g_loss_fn(z_k, self.G, self.D, x)
		gradG = g_tape.gradient(lossG, var_train)
		assert gradG, "generator gradient is None"
		self.G_opt.apply_gradients(zip(gradG, var_train))
		normgradG = tf.linalg.global_norm(gradG)
		return TrainLog(G_loss=lossG, Gg_norm=normgradG, Gg_time=g_timer.elapsed)

	def train_step(self):
		D_time_io, G_time_io = Timer(), Timer()
		data_k = self.data_handler.get_batch(self._curr_img_size)

		with D_time_io:
			_d_trainable = self.D.get_trainable(targ_res=self._curr_img_size)
			d_trainlog = self.strategy.run(self.d_train_step, args=(data_k, _d_trainable))
		end_d_timeIO = D_time_io.elapsed
		with G_time_io:
			g_trainlog = _g_trainable = self.G.get_trainable(targ_res=self._curr_img_size)
			self.strategy.run(self.g_train_step, args=(data_k, _g_trainable))
		end_g_timeIO = G_time_io.elapsed
  
		if (self.global_train_steps == 0 and self.trace_graph):
			with self.summarizer._summary_writer():
				tf.summary.trace_export(name="graph_trace", step=0, profiler_outdir=self.log_dir)

		with tf.name_scope("Generator"):
			self.summarizer.log_scalar('io_time_g', end_d_timeIO, self.generator_steps, reduce=None)
			self.summarizer.log_scalar('loss', g_trainlog.G_loss, self.generator_steps, reduce='MEAN')
			self.summarizer.log_scalar('grad_norm', g_trainlog.Gg_norm, self.generator_steps, reduce=None)
			self.summarizer.log_scalar('time_compute_grad', g_trainlog.Gg_time, self.generator_steps, reduce=None)
			g_params_norm = tf.linalg.global_norm(self.strategy.experimental_local_results(_g_trainable))
			self.summarizer.log_scalar('params_norm', g_params_norm, self.generator_steps, reduce='MEAN')
   
		with tf.name_scope("Discriminator"):
			self.summarizer.log_scalar('io_time_d', end_g_timeIO, self.discriminator_steps, reduce=None)
			self.summarizer.log_scalar('loss', d_trainlog.D_loss, self.discriminator_steps, reduce=None)
			self.summarizer.log_scalar('grad_norm', d_trainlog.Dg_norm, self.discriminator_steps, reduce=None)
			self.summarizer.log_scalar('time_compute_grad', d_trainlog.Dg_time, self.discriminator_steps, reduce=None)
			self.summarizer.log_scalar('grad_loss_penalty', d_trainlog.Dg_loss, self.discriminator_steps, reduce=None)
			d_params_norm = tf.linalg.global_norm(self.strategy.experimental_local_results(_d_trainable))
			self.summarizer.log_scalar('params_norm', d_params_norm, self.discriminator_steps, reduce='MEAN')

		with tf.name_scope("Evaluation_Matrix"):
			pass

		if self.global_train_steps * self.train_config.global_batch % self.train_config.save_shots == 0:
			input_x = None
			targ_size = None
			img_x = self.G.get_snap_rgb(input_x, targ_size)
			self.summarizer.log_image_grid("GeneratedImage", self.G.get_image_grid(), self.global_train_steps, reduce=None)
		
		if self.global_train_steps * self.train_config.global_batch % self.train_config.save_interval == 0:
			self.G.save_model()

		self.global_train_steps.assign_add(1)
		self.generator_steps.assign_add(1)
		self.discriminator_steps.assign_add(1)

		print("EPOCH : {} TIME : {} D_LOSS : {} G_LOSS : {}".format(None, None, None, None))