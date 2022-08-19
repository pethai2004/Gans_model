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
    epochs = 100
    G_lr = 0.0001
    D_lr = 0.0001
    lr_schedual = "linear"
    latent_z_dim = 200
    img_size = (32, 32)
    targ_img_size = (128, 128)
    n_gpu = ["GPU:0", "GPU:1"]
    log_dir = "TrainGans"
    stopdata = False
    seed = 5005
    TrainDict = None
    clip_norm = 3
    train_ratio = None

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
        print('====> [{}] Time: {:7.3f}s or {}'.format(self.elapsed, time.strftime("%H:%M:%S", self.elapsedH)))

pre_build_G = {"BaseFilters" : [300, 400, 500],
                "filters" : ([300, 300, 200], [300, 300, 200], [300, 300, 200]),
                "out_units" : 300, 
                "add_noise" : True,
                "dense_act" : "selu",
                "conv_act" : "selu",
                "out_act" : "selu",
                "num_layers" : 6,
                "dense_units" : 200,
                "kernel_size" : (5, 5)}

pre_build_D = {"BaseFilters" : [300, 400, 500],
                "filters" : ([300, 300, 200], [300, 300, 200], [300, 300, 200]),
                "out_units" : 300, 
                "add_noise" : True,
                "activation" : "selu",
                "out_act" : "selu",
                "num_layers" : 6,
                "dense_units" : 200}