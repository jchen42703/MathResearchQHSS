import keras
import numpy as np
import os
from glob import glob

# Utility Functions
def get_list_IDs(any_path, val_split = 0.8):
    '''
    param any_path: directory to either the align files or the .mpg files;
    i.e. s1_path or s1_align
    '''
    id_list =[os.path.splitext(file)[0] for file in os.listdir(any_path)]
    total = len(id_list)
    train = round(total * val_split)
    return {'train': id_list[:train], 'val': id_list[train:]
           }

def text_to_labels(text):
    '''
    Converts the align files to their encoded format.
    '''
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret

def enumerate_align_hash(align_path, absolute_max_string_len):
    '''
    Makes a dictionary of all of the align files
    * Make sure that `dir` ends with \\

    param align_path: path to the directory with all of the align files
    '''
    align_hash = {}
    video_list = glob(align_path+'*.align', recursive = True)
    for (i,video_path) in enumerate(video_list):
        video_id = os.path.splitext(video_path)[0].split('\\')[-1]
        align_hash[video_id] = Align(absolute_max_string_len, text_to_labels).from_file(video_path)
    return align_hash

# generator to inherit from
class BaseGenerator(keras.utils.Sequence):
    '''
    For generating 2D thread-safe data in keras. (no preprocessing and channels_last)
    Attributes:
      list_IDs: filenames (.nii files); must be same for training and labels
      data_dirs: list of [training_dir, labels_dir]
      batch_size: int of desired number images per epoch
      n_channels: <-
    '''
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(self, list_IDs, data_dirs, batch_size, shuffle = True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''
        Defines the fetching and on-the-fly preprocessing of data.
        '''
        # file names
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.data_gen(list_IDs_temp)
        return (X, y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.img_idx = np.arange(len(self.x))
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_gen(self, list_IDs_temp):
        '''
        Preprocesses the data
        Args:
            batch_x, batch_y
        Returns
            x, y
        '''
        raise NotImplementedError

# Align class from the LipNet repository
class Align(object):
    def __init__(self, absolute_max_string_len=32, label_func=None):
        self.label_func = label_func
        self.absolute_max_string_len = absolute_max_string_len

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
        self.build(align)
        return self

    def from_array(self, align):
        self.build(align)
        return self

    def build(self, align):
        self.align = self.strip(align, ['sp','sil'])
        self.sentence = self.get_sentence(align)
        self.label = self.get_label(self.sentence)
        self.padded_label = self.get_padded_label(self.label)

    def strip(self, align, items):
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):
        return " ".join([y[-1] for y in align if y[-1] not in ['sp', 'sil']])

    def get_label(self, sentence):
        return self.label_func(sentence)

    def get_padded_label(self, label):
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1
        return np.concatenate((np.array(label), padding), axis=0)

    @property
    def word_length(self):
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self):
        return len(self.sentence)

    @property
    def label_length(self):
        return len(self.label)
