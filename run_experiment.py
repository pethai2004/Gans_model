import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import activations
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from model import Discriminator_v0, Generator_v0
from datasets import DatManipulator, download_celeb_data, re_chunk_datasets, chunk_datasets
from objective import generator_loss_wg, discrim_loss_wg, discrim_loss_0, generator_loss_0

# download_celeb_data()
path = path = './celeba_gan/img_align_celeba'

# chunk_datasets(dir_path=path, num=10)
# re_chunk_datasets(path, num=15)

per_replica_batch = 200
use_gpu = 4
global_batch = use_gpu * per_replica_batch
latent_dim = 128
target_image_size = (100, 100)
summary = tf.summary.create_file_writer(logdir="train_gan")
pm_prior0 = tf.zeros((per_replica_batch, latent_dim))
pm_prior1 = tf.ones((per_replica_batch, latent_dim)) * 0.01
grad_penalty = 1

strategy = tf.distribute.MirroredStrategy(['/GPU:0', '/GPU:1', '/GPU:2', '/GPU:3'])
dat_instance = DatManipulator(global_batch=global_batch, parents_dir=path, 
                        strategy=strategy, target_image_size=(100, 100), chunk_first=False)

with strategy.scope():
    
    D = Discriminator_v0(image_shape=(*target_image_size, 3))
    D.initialize_model()

    G = Generator_v0(image_shape=(*target_image_size, 3))
    G.initialize_model()
    
    G_opt = tf.keras.optimizers.RMSprop(0.0001)
    D_opt = tf.keras.optimizers.RMSprop(0.0001)

@tf.function
def train_step(true_img_k, D, G, z_k):

    with tf.GradientTape() as D_tape:
        disc_L, disc_g_loss = discrim_loss_0(z=z_k, G=G, D=D, train_sets=true_img_k, g_penalty=grad_penalty)
    disc_G = D_tape.gradient(disc_L, D.model.trainable_variables)
    D_opt.apply_gradients(zip(disc_G, D.model.trainable_variables))

    with tf.GradientTape() as G_tape:
        gen_L = generator_loss_0(z=z_k, G=G, D=D, train_sets=true_img_k)
    gen_G = G_tape.gradient(gen_L, G.model.trainable_variables)
    G_opt.apply_gradients(zip(gen_G, G.model.trainable_variables))

    return (tf.reduce_mean(gen_L + disc_L), 
            tf.reduce_mean(gen_L), 
            tf.reduce_mean(disc_L), 
            G.forward_model(z_k[:3]))

def summarising(X, Y, Z, printing=False):
    with summary.as_default():
        tf.summary.scalar('generator_loss', X, k_k)
        tf.summary.scalar('discriminator_loss', Y, k_k)
        tf.summary.image('generated_image', Z, k_k)
    if printing:
        tf.print('generator_loss', X, 'discriminator_loss', Y)

k_k = 0

while k_k < 300:
    
    prior = tfp.distributions.Normal(pm_prior0, pm_prior1)
    dist_data = next(dat_instance)
    epoch_loss, g_k_loss, d_k_loss, im_geg = strategy.run(train_step, args=(dist_data, D, G, prior.sample()))
    strategy.run(summarising, args=(g_k_loss, d_k_loss, im_geg))
    tf.print('epoch:', k_k, )
    k_k += 1
    if k_k % 50 == 0:
        plt.imshow(strategy.experimental_local_results(im_geg)[0][1])



