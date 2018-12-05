from MathResearchQHSS.lipreading.io.generator import FrameGenerator
from MathResearchQHSS.lipreading.models import LipNet
from MathResearchQHSS.lipreading.io.io_utils import get_list_IDs
import os
from glob import glob
import skvideo

n_epochs = 200
# paths to specific speaker
s1_path = '/content/s1/'
s1_align_path = '/content/s1_align/'

# initializes the generators
list_IDs = get_list_IDs(s1_path, val_split = 0.8)
data_dirs = [s1_path, s1_align_path]
train_gen = FrameGenerator(list_IDs['train'], data_dirs, batch_size = 1)
val_gen = FrameGenerator(list_IDs['val'], data_dirs, batch_size = 1)

# training
lipnet = LipNet()
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

lipnet.model.fit_generator(generator = train_gen, epochs = n_epochs, max_queue_size = 1, workers = 2, use_multiprocessing = True)
lipnet.model.save_weights('train_weights_'+str(n_epochs)+'.h5')
