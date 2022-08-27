from collections import defaultdict
import time
import tensorflow as tf
import numpy as np
import pprint
import pickle
import os

class DefaultConfig(object):
    """Default configuration for training"""
    # General parameters
    epochs = 10
    G_lr = 0.00001
    D_lr = 0.00001
    max_G_lr = 0.0001
    max_D_lr = 0.0001
    optimizer = "RMSprop"
    lr_schedual = "linear"
    latent_z_dim = 200
    img_size = (32, 32)
    targ_img_size = (128, 128)
    n_gpu = ["GPU:0", "GPU:1"]
    clip_norm = 3
    train_ratio = None
    grad_penalty = 2
    applied_D_method = "resize"
    switch_ratio = {"G" : 2, "D" : 1}
    
    def __init__(self, **kwargs):
        """Update default configuration"""
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def __str__(self):
        """Print configuration"""
        return pprint.pformat(self.__dict__)
        
    def save(self, filename):
        """Save configuration to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filename):
        """Load configuration from file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
            
    def update(self, kwargs):
        """Update configuration"""
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def set_seed(self):
        """Set random seed"""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
    def set_device(self):
        """Set device"""
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Timer:
    def __init__(self):
        self.end = 0
        self.elapsed = 0
        self.elapsedH = 0

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
      self.stop()

    def start(self):
        self.begin = time.time()
        return self

    def stop(self):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)

pre_build_G = {"BaseFilters" : [300, 400 ,500, 400, 300],
                "filters" : ([300, 300, 200], [300, 300, 200]),
                "out_units" : 300, 
                "add_noise" : True,
                "dense_act" : "selu",
                "conv_act" : "selu",
                "out_act" : "selu",
                "num_layers" : 6,
                "dense_units" : 200,
                "kernel_size" : (5, 5)}

pre_build_D = {"filters" : [(200, 2), (300, 2), (300, 2), 
                             (200, 2), (300, 2), (300, 2), (300, 2)] ,
                "out_units" : 300, 
                "units_dense" : [300 , 300],
                "add_noise" : True,
                "conv_act" : "selu",
                "dense_act" : "selu",
                "out_act" : "selu",
                "num_layers" : 6,
                "kernel_size" : (5, 5),
                "dense_units" : 200}

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def exponential_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.exp(-step / total_steps))

def linear_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * step / total_steps

class AlphaDecay:

    def __init__(self, total_steps, lr_max=0.0001, lr_min=0.000001, decay_type="linear"):
        
        self.decay_type = decay_type
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min

    def __call__(self, step):
        if self.decay_type == "linear":
            return linear_annealing(step, self.total_steps, self.lr_max, self.lr_min)
        elif self.decay_type == "cosine":
            return cosine_annealing(step, self.total_steps, self.lr_max, self.lr_min)
        elif self.decay_type == "exponential":
            return exponential_annealing(step, self.total_steps, self.lr_max, self.lr_min)
        else:
            raise ValueError("Unknown decay type: {}".format(self.decay_type))