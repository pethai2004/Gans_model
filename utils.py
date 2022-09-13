
import time
import tensorflow as tf
import numpy as np
import pprint
import pickle
import os 
import matplotlib.pyplot as plt
from tensorflow import keras
from default_params import G_filters, D_filters, _train_ratio
from PIL import Image

def get_var_by_name(var, name):
    Vlist = []
    for _v in var:
        if _v.name == name:
            Vlist.append(_v)
    return Vlist

def var_intersection(var0, var1):
    assert len(var0) <= len(var1), 'len of first argument must be less than the last one'
    intersected = []
    var_ref0 = [v0.ref() for v0 in var0]
    for v1 in var1:
        if v1.ref() not in var_ref0:
            intersected.append(v1)
    return intersected

def get_name_var(var): return [_i.name for _i in var]

def function_call(functional, inputx):
    x_ = functional.layers[0](inputx)
    for _lays in functional.layers[1:]:
        x_ = _lays(x_)
    return x_

def matrix_FID(img0, img1, func):
    """Frechet Inception Distance (FID) https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""
    with tf.device("/cpu:0"):
        mu_fake = np.mean(func(img0), axis=0)
        sigma_fake = np.cov(func(img0), rowvar=False)
        mu_real = np.mean(img1, axis=0)
        sigma_real = np.cov(img1, rowvar=False)
        
        m = np.square(mu_fake - mu_real).sum()
        s, error = tf.linalg.sqrtm(np.dot(sigma_fake, sigma_real)) 
        _dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        _dist = tf.math.real(_dist) 
    return _dist

def matrix_IS():
    """Inception Score (IS)  https://github.com/openai/improved-gan/blob/master/inception_score/model.py"""
    pass

def matrix_KID():
    """Kernel Inception Distance (KID) https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py"""
    pass

def get_sample(targ_m):
    return [np.expand_dims(np.random.rand(*i[1:]),0) for i in targ_m.input_shape]
    
class TrainingConfig(object):
    """Default configuration for training"""
    # General parameters
    global_batch = 300
    max_image = 50_000
    epochs = 10
    G_lr = 0.00001
    D_lr = 0.00001
    max_G_lr = 0.0001
    max_D_lr = 0.0001
    G_optimizer = keras.optimizers.RMSprop
    D_optimizer = keras.optimizers.RMSprop
    G_lr_schedual = tf.keras.optimizers.schedules.CosineDecayRestarts(
            G_lr, first_decay_steps=30, t_mul=2.0, m_mul=1.0, alpha=0.1,name='generator_lr_schedule')
    D_lr_schedual = tf.keras.optimizers.schedules.CosineDecayRestarts(
            G_lr, first_decay_steps=30, t_mul=2.0, m_mul=1.0, alpha=0.1,name='generator_lr_schedule')
    img_size = (32, 32)
    targ_img_size = (128, 128)
    n_gpu = ["GPU:0", "GPU:1"]
    train_ratio = _train_ratio
    grad_penalty = 1
    switch_ratio = {"G" : 2, "D" : 1}
    seed = 2002

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
    def __repr__(self):
        return self.name

class ModelConfig(object):
    """Default configuration for generator"""
    name = None
    latent_dim = 256
    img_size = (8, 8)
    targ_img_size = (64, 64)
    initializer = None
    seed = 1010
    lerp = 0.5 # interpolation
    decay_lerp = 0.

    G_kernel_size = (4, 4)
    G_dense_units = [200] * 5
    G_conv_act = "LeakyReLU"
    G_dense_act = "LeakyReLU"
    G_conv_out_act = None
    G_dense_out_act = None
    G_filters = G_filters
    G_up_method = "upsampling"

    D_kernel_size = (4, 4)
    D_dense_units = [200] * 2
    D_conv_act = "LeakyReLU"
    D_dense_act = "LeakyReLU"
    D_conv_out_act = None
    D_dense_out_act = None
    D_filters = D_filters
    D_up_method = "upsampling"

    def __init__(self, **kwargs):
        """Update default configuration"""
        for k, v in kwargs.items(): setattr(self, k, v)
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
        for k, v in kwargs.items(): setattr(self, k, v)
    def set_seed(self):
        """Set random seed"""
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
    def __repr__(self):
        return self.name
    
class NameCaller(object):
    def __init__(self):
        self.num = 0
    @property
    def n(self):
        self.num += 1
        return str(self.num).zfill(3)

class Timer:
    def __init__(self):
        self.end = 0
        self.elapsed = 0
        self.elapsedH = 0
    def __enter__(self): self.start()
    def __exit__(self, *args): self.stop()
    def start(self):
        self.begin = time.time()
        return self
    def stop(self):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)

class TrainLog(object):
    
    def __init__(self, **kwargs):
        """Update default configuration"""
        for k, v in kwargs.items(): setattr(self, k, v)
    def update(self, kwargs):
        """Update configuration"""
        for k, v in kwargs.items(): setattr(self, k, v)
            
def show_batch(image_batch):
    plt.figure(figsize=(10,10))
    for n in range(9):
        ax = plt.subplot(3,3,n+1)
        plt.imshow(image_batch[n])
        plt.axis('off')
        
def merge_image_grid(images, rows, cols):
    """Merge a grid of images into one image.
    Args:
        images: List of images to merge. Images must all be the same size.
        rows: Number of rows in grid.
        cols: Number of columns in grid.
    Returns:
        Merged image with shape (rows * height, cols * width, channels).
    """
    height, width = images[0].shape[:2]
    channels = images[0].shape[2]
    merged = np.zeros((rows * height, cols * width, channels))
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        merged[row * height:(row + 1) * height, col * width:(col + 1) * width, :] = image
    return merged

class Logger:
    
    def __init__(self, log_dir="TRAINGANS", strategy=None, image_record_dir="images"):
        
        if os.path.exists(log_dir): 
            f = 0
            while os.path.exists(log_dir + "_" + str(f)): 
                f += 1
            log_dir = log_dir + "_" + str(f)
        print("Start Logging at {}".format(log_dir))
        self.log_dir = log_dir
        self._save_model = "model"
        self._summary_writer = tf.summary.create_file_writer(os.path.join("summary", self.log_dir))
        self._strategy = strategy
        self.time_dict = {}
        self.record_img_dir = os.path.join(self.log_dir, image_record_dir)
        
    def add_time_dict(self, name):
        self.time_dict[name] = Timer()
        
    def start_time_dict(self, name):
        self.time_dict[name].start()
        
    def stop_time_dict(self, name):
        self.time_dict.stop()
        
    def log_scalar(self, name, value, step, reduce="MEAN"):
        if isinstance(value, tf.distribute.DistributedValues):
            assert self._strategy, "strategy is None"
            value = self._strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None) 
            ## Is this equivilent to tf.reduce_mean(self.strategy.experimental_local_results(value))?
        with self._summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)
    
    def log_image(self, name, image, step, allow_multiple=False):
        if isinstance(image, tf.distribute.DistributedValues):
            assert self._strategy, "strategy is None"
            image = self._strategy.experimental_local_results(image)
        if len(image) > 1 and not allow_multiple:
            raise ValueError("image must be a single image")
        with self._summary_writer.as_default():
            tf.summary.image(name, image, step=step)
    
    def log_image_grid(self, name, image, step, rows=3, cols=3):
        if isinstance(image, tf.distribute.DistributedValues):
            assert self._strategy, "strategy is None"
            image = self._strategy.experimental_local_results(image)
        assert image.shape[0] == rows * cols, "image.shape[0] != rows * cols"
        image_grid = merge_image_grid(image, rows, cols)
        self.log_image(name, image_grid, step)
        
    def log_TrainLog(self, train_log, step):
        raise NotImplementedError
        assert isinstance(train_log, TrainLog), "train_log must be a TrainLog"
        for k, v in train_log.__dict__.items():
            pass
        
    def print_log(self, train_log):
        raise NotImplementedError
    
    def record_image(self, image):
        if not os.path.exists(self.record_img_dir): os.makedirs(self.record_img_dir)
        if isinstance(image, tf.distribute.DistributedValues):  
            image = tf.concat(self._strategy.experimental_local_results(image), axis=0)
        for img in image:
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
            image.save(os.path.join(self.record_img_dir, "{}.png".format(time.time())))
    
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
