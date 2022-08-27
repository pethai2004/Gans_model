import os, shutil
import gdown
from zipfile import ZipFile

from tensorflow import keras
import tensorflow as tf

def download_celeb_data():
    try:
        os.makedirs("data")
    except:
        pass
    url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    output = "data/data.zip"
    gdown.download(url, output, quiet=True)

    with ZipFile("data/data.zip", "r") as zipobj:
        zipobj.extractall("data")

def chunk_datasets(dir_path, num=3, del_left=True):
    files_all = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    length = len(files_all) // num
    start = 0
    for P in range(num):
        name_fold = str("fold_chunk_" + str(P))
        try:
            os.mkdir(os.path.join(dir_path, name_fold))
        except:
            pass
        file_move = files_all[start : start+length]
        for each_f in file_move:
            curr_p = os.path.join(dir_path, each_f)
            new_p = os.path.join(dir_path, name_fold, each_f)
            shutil.move(curr_p, new_p)        
        start = start+length
        print("finish_chunk")

    if del_left:
        for ek in os.listdir(dir_path):
            if 'fold_chunk' not in ek:
                try:
                    os.remove(ek)
                except:
                    pass
    else:
        raise NotImplementedError('not support data remainder chunk')

def re_chunk_datasets(dir_path, num=3, del_left=True, suf_name='fold_chunk_'):
    kD = []
    valid_suffix = ['.jpg', '.png']
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if os.path.splitext(name)[-1] in valid_suffix:
                kD.append(os.path.join(path, name))
    kD = sorted(kD)
    length = len(kD) // num
    start = 0
    
    for P in range(num):
        name_fold = str("nn_ch_" + str(P))
        try:
            os.mkdir(os.path.join(dir_path, name_fold))
        except:
            pass
        file_move = kD[start : start+length]
        for each_f in file_move:
            
            new_p = os.path.join(dir_path, name_fold, os.path.basename(each_f))
            shutil.move(each_f, new_p)        
        start = start+length
        print("finish_chunk")
    
    for i in os.listdir(dir_path):
        if suf_name in i:
            shutil.rmtree(os.path.join(os.getcwd(), dir_path, i))
            
    for (k, i) in enumerate(os.listdir(dir_path)):
        if 'nn_ch' in i:
            os.rename(os.path.join(dir_path, i), os.path.join(dir_path, str(suf_name + str(k))))
            
    if del_left:
        for ek in os.listdir(dir_path):
            if 'fold_chunk' not in ek:
                try:
                    os.remove(ek)
                except:
                    pass

def rename_all(dir_path):
    idx = 0
    for P in dir_path:
        os.rename(P, str(idx) + os.path.splitext(P)[-1])
        idx+=1

class DatManipulator:
    
    def __init__(self, global_batch, parents_dir, strategy, target_image_size=(100, 100),chunk_first=False, non_stop=True):
        self.parents_dir = parents_dir
        self.strategy = strategy
        self.global_batch = global_batch
        self.target_image_size = target_image_size
        if chunk_first:
            chunk_datasets(self.parents_dir, num=3)
        self.dir_chunk = iter(sorted([i for i in os.listdir(parents_dir) if 'fold_chunk_' in i]))
        self.idx_data = 0
        self.current_dataset_ch = None
        self.curr_dir_num_file = 0 
        self.hist_idx_data = 0 
        self.non_stop = non_stop

    def get_data_distributed(self):
        '''Get distributed data from subdirectory'''

        def data_func(input_context):

            batch_size_repc = input_context.get_per_replica_batch_size(self.global_batch)
            self.curr_dir_num_file = len(os.listdir(current_dir))
            assert self.global_batch < self.curr_dir_num_file , 'global_batch mush be smaller than each dataset subdirectory'
            
            with tf.device("CPU:0"):
                datasets = keras.utils.image_dataset_from_directory(current_dir,
                    labels=None, label_mode=None, class_names=None, color_mode='rgb', batch_size=None,
                    image_size=self.target_image_size, shuffle=False)

                datasets = datasets.map(lambda img : img / 255)
                datasets = datasets.shuffle(30000)
                datasets = datasets.batch(batch_size_repc)
                datasets = datasets.prefetch(3)

            return datasets
        try:
            current_dir = os.path.join(self.parents_dir, next(self.dir_chunk))
        except:
            if self.non_stop:
                print('since DatManipulator specified non_stop, data loop continue sampling from past history automatically')
                self.reset_instance()
                current_dir = os.path.join(self.parents_dir, next(self.dir_chunk))
            else:
                raise StopIteration('Dataset is been sampled, to resample pass datasets, use reset_instance')
                
        self.current_dataset_ch = iter(self.strategy.distribute_datasets_from_function(data_func))
        print('continue sample from directory :', os.path.split(current_dir)[-1])

    def get_batch(self):
        '''Call this everytime step, get a batch from distributed data'''
        if (self.idx_data == 0 and self.hist_idx_data == 0): #instantiate data
            self.get_data_distributed()

        self.idx_data += self.global_batch
        self.hist_idx_data += self.global_batch # here to keep track only

        if self.idx_data >= self.curr_dir_num_file:
            self.get_data_distributed()
            self.idx_data = 0 
            
        instant_data = next(self.current_dataset_ch)
        print('current_global_dix : ', self.idx_data, 'sampled_so_far : ', self.hist_idx_data)
        print(self.curr_dir_num_file // self.global_batch, self.curr_dir_num_file - self.idx_data)
        if ((self.curr_dir_num_file - self.idx_data) % self.global_batch != 0 and 
                        self.idx_data // self.global_batch == self.curr_dir_num_file // self.global_batch):
            self.get_data_distributed()
            self.idx_data = 0 
            print('stop sampled from current dir since the remaining data does not match global_batch')
        return instant_data

    def reset_instance(self):
        '''Resample pass datasets'''
        self.dir_chunk = iter(sorted([i for i in os.listdir(self.parents_dir) if 'fold_chunk_' in i]))

    def update_size(self, size=(32, 32)):
        print("chage image target size from {} to {}, this will apply for the next chunk only".format(self.target_image_size, size))
        self.target_image_size = size
        
    def __next__(self):
        return self.get_batch()
