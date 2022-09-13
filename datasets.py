# TODO : Compare these solution
# 1. finish datahandler read from chunked TFRecord dataset ---> checked
# 2. finish datahandler read from pure chunked dataset 
# 3. (Optional) finish datahandler read from pure TFRecord dataset

import os, shutil, glob
import gdown
from zipfile import ZipFile
from PIL import Image

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior

from utils import _bytes_feature, _float_feature, _int64_feature
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def _Log2(_x) : return int(np.log2(_x))
############################################################################################################
class SerializeTF:
    
    def __init__(self, data_dir, resolution=32, record_dir="TFRecord", valid_suffix="jpg"):
        self.data_dir = data_dir
        self.cur_idx = 0
        self.valid_suffix = valid_suffix
        self.record_dir = record_dir
        self.resolution = resolution 
        self.record_prefix = "TFrecord_{}x{}".format(resolution, resolution)
        if not os.path.exists(record_dir) : os.mkdir(record_dir)
        self.writer = tf.io.TFRecordWriter(os.path.join(self.record_dir, self.record_prefix + ".tfrecords"))
    
    def serialize(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, "*."+self.valid_suffix)))
        for f in files:
            img = Image.open(f)
            if self.resolution is not None:
                img = img.resize((self.resolution, self.resolution))
            img = np.array(img)
            img_shape = img.shape
            img = img.tobytes()
            feature = {'shape': _int64_feature(img_shape), 'image': _bytes_feature(img)}
            example = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
            self.writer.write(example)
            self.cur_idx += 1
            if self.cur_idx % 10000 == 0: print("Finish serialize {} images".format(self.cur_idx))
        self.writer.close()

class DeserializeTF:
    
    def __init__(self, record_file, preprocess_fn=None, **pipeline_kwargs):
        self.record_file = record_file
        self.preprocess_fn = preprocess_fn if preprocess_fn is not None else preprocess
        self.pipeline_args = pipeline_kwargs
        self.RecordData = None
        self.shape_data = None
        self.num_read = tf.data.experimental.AUTOTUNE
        self.cast_type = tf.int64
        self.feat_img_name = "image"
        
    def deserialize(self, apply_iterator=False):

        self.RecordData = tf.data.TFRecordDataset(self.record_file, num_parallel_reads=self.num_read)
        feature = {'shape': tf.io.FixedLenFeature([3], tf.int64),
                   self.feat_img_name: tf.io.FixedLenFeature([], tf.string)}
        dataset = self.RecordData.map(lambda _x : tf.io.parse_single_example(_x, feature)[self.feat_img_name], num_parallel_calls=self.num_read)
        self.shape_data = self.RecordData.map(lambda _x : tf.io.parse_single_example(_x, feature)["shape"], num_parallel_calls=self.num_read)
        dataset = dataset.map(lambda _x : tf.io.decode_raw(_x, tf.uint8), num_parallel_calls=self.num_read)
        _shape = next(iter(self.shape_data)).tolist() #TODO : using tf.data.Dataset.get_single_element() instead
        dataset = dataset.map(lambda _x : tf.reshape(_x, _shape))
        dataset = dataset.map(lambda _x : tf.cast(_x, self.cast_type))
        dataset = self.preprocess_fn(dataset, **self.pipeline_args)
        if apply_iterator : dataset = iter(dataset)
        return dataset
    
    def inspect_shape(self):
        _rec = tf.data.TFRecordDataset(self.record_file, num_parallel_reads=self.num_read)
        _feature = {'shape': tf.io.FixedLenFeature([3], tf.int64)}
        return tf.io.parse_single_example(next(iter(_rec)).numpy(), _feature)["shape"].tolist()
    
    def __repr__(self):
        return "DeserializeTF({})".format(self.record_file)
############################################################################################################

def extract_to_record(data_dir, init_res=4, targ_res=64, record_dir="TFRecord", valid_suffix="jpg"): #TODO : add parallel process in for loop
    '''Extract image from data_dir to TFRecord with resolution init_res to targ_res'''
    if not os.path.exists(record_dir): os.mkdir(record_dir)
    for i in range(_Log2(init_res), _Log2(targ_res)+1):
        print("Start serialize {}x{}".format(2 ** i, 2 ** i))
        serialize = SerializeTF(data_dir, resolution=2 ** i, record_dir=record_dir, valid_suffix=valid_suffix)
        serialize.serialize()

def ext_ch(data_dir, init_res=4, targ_res=64, based_rec_dir="ChunkTFRecord", valid_suffix="jpg", chunk_size=10): 
    '''Extract image from data_dir to TFRecord with resolution init_res to targ_res'''
    prefix, targ_path = "ImgSum", "Datasets"
    chunk_datasets(data_dir, num=chunk_size, prefix=prefix, targ_path=targ_path)
    if not os.path.exists(based_rec_dir) : os.mkdir(based_rec_dir)
    dir_ch_list = [os.path.join(targ_path, f) for f in os.listdir(targ_path)]
    if not os.path.exists(based_rec_dir) : os.mkdir(based_rec_dir)
    for dir_chunks in dir_ch_list:
        rec_dir = os.path.join(based_rec_dir, os.path.basename(dir_chunks))
        print(rec_dir)
        extract_to_record(dir_chunks, init_res, targ_res, rec_dir, valid_suffix)
        print("Done serializing chunked datasets")

def ext_without_replace_ch(data_dir, init_res=4, targ_res=64, based_rec_dir="ChunkTFRecord", valid_suffix="jpg", chunk_size=10): 
    '''Behaving like ext_ch but will not use the same set in different resolution sets'''
    pass
def chunk_datasets(dir_path, num=3, prefix="ImgSum", targ_path="Datasets"):
    files_all = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    length = len(files_all) // num
    start = 0
    _dir = os.path.join(os.getcwd(), targ_path)
    if not os.path.exists(_dir): os.mkdir(_dir)
    for P in range(num):
        name_fold = str(prefix + str(P).zfill(2))
        _dir_tar = os.path.join(targ_path, name_fold)
        if not os.path.exists(_dir_tar): os.mkdir(_dir_tar)
        file_move = files_all[start : start+length]
        for each_f in file_move:
            curr_p = os.path.join(dir_path, each_f)
            new_p = os.path.join(_dir_tar, each_f)
            shutil.move(curr_p, new_p)        
        start = start+length
    print("Done chunk")
        
def chunk_with_ratio(dir_path, ratio, prefix="ImgSum", targ_path="Datasets"):
    if isinstance(ratio, int):
        pass
    files_all = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    eps = np.sum([i for i in ratio.values()])
    num = {i : int((j / eps) * len(files_all))  for (i, j) in ratio.items()}
    _dir = os.path.join(os.getcwd(), targ_path)
    if not os.path.exists(_dir): os.mkdir(_dir)
    start, _idx = 0, 0
    for i, length in num.items():
        name_fold = str(prefix + "{}x{}_{}".format(i, i, _idx))
        _dir_tar = os.path.join(targ_path, name_fold)
        if not os.path.exists(_dir_tar): os.mkdir(_dir_tar)
        file_move = files_all[start : start+length]
        for each_f in file_move:
            curr_p = os.path.join(dir_path, each_f)
            new_p = os.path.join(_dir_tar, each_f) 
            shutil.move(curr_p, new_p)      
        start = start+length
        _idx+=1
        print("Finish make Folder to subset {}/{} length : {}".format(targ_path, name_fold, length))

def re_chunk_with_ratio(dir_path, ratio):
    files_all = []
    _dir0 = os.listdir(dir_path)
    valid_suffix = ['.jpg', '.png']
    _temp_name = "nn_ch"
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if os.path.splitext(name)[-1] in valid_suffix: files_all.append(os.path.join(path, name))
    files_all = sorted(files_all)
    eps = np.sum([i for i in ratio.values()])
    num = {i : int((j / eps) * len(files_all))  for (i, j) in ratio.items()}
    start, _idx = 0, 0
    for i, length in num.items():
        name_fold = str(_temp_name + "{}x{}_{}".format(i, i, _idx))
        try: os.mkdir(os.path.join(dir_path, name_fold))
        except: pass
        file_move = files_all[start : start+length]
        for each_f in file_move:
            new_p = os.path.join(dir_path, name_fold, os.path.basename(each_f))
            shutil.move(each_f, new_p)      
        start = start+length
        print("Finish remake Folder to subset {} length : {}".format(new_p, length) )
        _idx+=1
    for i in _dir0: 
        if os.path.isdir(i) :shutil.rmtree(i) 

def img_from_file(dir_path, suffix="jpg"):
        files = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if suffix in i]
        _data = tf.data.Dataset.from_tensor_slices(files)
        
        def load_image(x):
            img = tf.io.read_file(x)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, (128, 128))
            return img

        _data = _data.map(lambda x: load_image(x), num_parallel_calls=tf.data.AUTOTUNE)
        return _data
    
def preprocess(dataset, batch_size=300, channel=3, img_size=None, shuffle=True, seed=2002, resize_method="nearest", normalize=True, shuff_size=20000, face_detect=False, dtype=tf.float32):
    assert not face_detect, "Not support face_detect yet"
    with tf.name_scope("PreprocessingImage"):
        if normalize:
            dataset = dataset.map(lambda _x : _x / 255, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=shuff_size, seed=seed) if shuffle else dataset
        if img_size:
            dataset = dataset.map(lambda _x: tf.image.resize(_x, img_size, method=resize_method), num_parallel_calls=tf.data.AUTOTUNE)
        if channel < 3:
            dataset = dataset.map(lambda _x: tf.image.rgb_to_grayscale(_x), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda _x: tf.cast(_x, dtype), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        if batch_size : dataset = dataset.batch(batch_size)
    return dataset

def list_record_features(tfrecords_path):
    #ref : https://stackoverflow.com/questions/63562691/reading-a-tfrecord-file-where-features-that-were-used-to-encode-is-not-known
    features = {}
    for rec in tf.data.TFRecordDataset([str(tfrecords_path)]):
        example_bytes = rec.numpy()
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        for key, value in example.features.feature.items():
            kind = value.WhichOneof('kind')
            size = len(getattr(value, kind).value)
            if key in features:
                kind2, size2 = features[key]
                if kind != kind2:
                    kind = None
                if size != size2:
                    size = None
            features[key] = (kind, size)
    return features

class DataHandler:
    '''Specific class for handling datasets use in training generative model'''
    def __init__(self, record_dir, strategy, global_batch, preprocess_fn = None, init_res=4, fin_res=64):
        ''' Form of record_dir must be :
        root_dir ---> chunk_tf_records001 ---> 001*.tfrecords #of resolution 4
                                          ---> 002*.tfrecords #of resolution 8
                      chunk_tf_records002 ---> 001*.tfrecords #of resolution 4
                                          ---> 002*.tfrecords #of resolution 8
        '''
        self.record_dir = record_dir
        self.global_batch = global_batch
        self.init_res = init_res # initial resolution
        self.fin_res = fin_res # final resolution
        self._preprocess_fn = preprocess_fn if preprocess_fn != None else preprocess 
        self._per_replica_batch = int(global_batch / strategy.num_replicas_in_sync) # batch size per replica
        self._strategy = strategy # strategy for distributed training
        self._distributed_data = None # distributed dataset
        self._record_dir = [os.path.join(record_dir, i) for i in os.listdir(record_dir) if not i.startswith('.')] # chunk that TFrecord are stored
        self._num_chunk = len(self._record_dir) # number of chunk
        self._record_dict = {i : j for i, j in enumerate(sorted(self._record_dir))} #{idx : path to chunk}
        
        length_f = [os.listdir(f) for f in self._record_dir]
        assert all(element == length_f[0] for element in length_f)
        self._name_rec = length_f[0] # all TFrecord in chunk, same for all chunk    
        self._num_tf_rec = len(length_f[0]) # number of TFrecord in chunk
        self._record_dir_iter = iter(self._record_dir) # iterator for chunk
        assert self._num_tf_rec == _Log2(fin_res) - _Log2(init_res) + 1, "tf record do not match number of res" 
        self._idx_chunk = { 2 **i : 0 for i in range(_Log2(init_res), _Log2(fin_res))} # idx of chunk for each resolution
        self._deserialize_objects = {} # dict of all associated DeserializeTF object for each TFrecord
        self._get_dsr_dict()  # get all DeserializeTF object for each resolution and assign to self._deserialize_objects
        self._fixed_targ_img = None
        
    def _get_dsr_dict(self):
        dict_rec = {}
        for rec, name_dir in self._record_dict.items():
            for name_rec in self._name_rec:
                path_f = os.path.join(name_dir, name_rec)
                dsr = DeserializeTF(path_f, batch_size=None, img_size=None)
                sh = dsr.inspect_shape()[1]
                dict_rec["dir : {} shape : {}x{}".format(rec, sh, sh)] = dsr
        self._deserialize_objects = dict_rec
        assert len(self._deserialize_objects) == self._num_chunk * self._num_tf_rec, "record dict not match"
    
    def _get_dsr(self, idx_chunk, res):
        key = "dir : {} shape : {}x{}".format(idx_chunk, res, res)
        return self._deserialize_objects[key]
    
    def _get_distributed(self, targ_size):
        self._fixed_targ_img = targ_size
        _dsr = self._get_dsr(self._idx_chunk[targ_size], targ_size)
        def _data_func_dsr(input_context):
            batch_size_repc = input_context.get_per_replica_batch_size(self.global_batch)
            _dataset_dsr = _dsr.deserialize(apply_iterator=None)
            _dataset_dsr = _dataset_dsr.batch(batch_size_repc, 
                                              drop_remainder=True, 
                                              num_parallel_calls=tf.data.AUTOTUNE)
            return _dataset_dsr
        self._distributed_data = iter(self._strategy.distribute_datasets_from_function(_data_func_dsr))
        print("Sampled from chunk {} for resolution {}x{}".format(self._idx_chunk[targ_size], targ_size, targ_size))
    
    def require_new_size(self, targ_size):
        "Must call this before calling get_batch if require new resolution"
        self._fixed_targ_img = targ_size
        if self._get_distributed is not None : self._get_distributed(targ_size) 
        
    def get_batch(self, targ_size):
        if self._distributed_data is None:
            self._get_distributed(targ_size)
        assert targ_size == self._fixed_targ_img, "call require_new_size before calling get_batch"
        _next_batch = self._distributed_data.get_next_as_optional()
        if _next_batch.has_value(): return _next_batch.get_value()
        else:
            if self._idx_chunk[targ_size] == self._num_chunk - 1:
                raise StopIteration("No more data")
            self._idx_chunk[targ_size] += 1
            self._get_distributed(targ_size)
            self.get_batch(targ_size)
    
    def __repr__(self):
        return "DataHandler"
    
def get_celeb_data(Name="DataImgCelebA", quiet=False, rmt=True):
    url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    try: os.makedirs(Name)
    except: pass
    output = os.path.join(Name, "data.zip")
    gdown.download(url, output, quiet=quiet)
    with ZipFile(output, "r") as zipobj:
        zipobj.extractall()
    if rmt: shutil.rmtree(Name)
    os.rename("img_align_celeba", Name)
    print("----------------------------Finish extract dataset----------------------------")

def get_tfr_celeb(name_record="CelebTFRecord", init_res=4, targ_res=128, chunk_size=10):
    get_celeb_data("DataImgCelebA")
    extract_to_record_with_chunk("DataImgCelebA", init_res, targ_res, 
        based_rec_dir=name_record, valid_suffix="jpg", chunk_size=chunk_size)

def get_celebMaskHQ_data(Name="DataImgCelebMask", quiet=False):
    url = "https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view"
    try: os.makedirs(Name)
    except: pass
    output = os.path.join(Name, "data.zip")
    gdown.download(url, output, quiet=quiet)
    with ZipFile(output, "r") as zipobj:
        zipobj.extractall()
        
def get_FFHQ128(Name="DataImgFFHQ128", quiet=False):
    url = "https://drive.google.com/drive/folders/1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv" # 128
    try: os.makedirs(Name)
    except: pass
    output = os.path.join(Name, "data.zip")
    gdown.download(url, output, quiet=quiet)
    with ZipFile(output, "r") as zipobj:
        zipobj.extractall(os.path.join(Name,"data"))