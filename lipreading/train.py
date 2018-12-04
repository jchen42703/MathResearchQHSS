from MathResearchQHSS.lipreading.io.generator import FrameGenerator
from MathResearchQHSS.lipreading.models import LipNet
import os
from glob import glob
import skvideo

n_epochs = 200

data_path = 'C:\\Users\\Joseph\\Desktop\\Grid Corpus\\Video\\'
s1_path = 'C:\\Users\\Joseph\\Desktop\\Grid Corpus\\Video\\s1.mpg_6000.part1\\s1\\video\\mpg_6000\\'
s1_align_path = 'C:\\Users\\Joseph\\Desktop\\Grid Corpus\\Align\\s1\\align\\'
#
# s1_files = glob(s1_path + '*.mpg', recursive = True)
# s1_align_files = glob(s1_align_path + '*.align', recursive = True)

list_IDs = get_list_IDs(s1_path)
data_dirs = [s1_path, s1_align_path]
train_gen = FrameGenerator(list_IDs['train'], data_dirs, batch_size = 1)
val_gen = FrameGenerator(list_IDs['val'], data_dirs, batch_size = 1)
# gen.__getitem__(1)

model = LipNet()
model.fit_generator(generator = gen, n_epochs, workers = 2, mulitprocessing = True)
model.save_weights('train_weights_'+str(n_epochs)+'.h5')
