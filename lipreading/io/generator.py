from MathResearchQHSS.lipreading.io.io_utils import *
import numpy as np
import keras
import skvideo
# path = 'C:\\Users\\jchen\\Downloads\\Programming\\ffmpeg\\ffmpeg\\bin\\'
# skvideo.setFFmpegPath(path)


# generator to load files iteratively (75 frames at a time)
class FrameGenerator(BaseGenerator):
    '''
    list_IDs: file IDs (without the suffix); files must be .mpg and .align files
    data_dirs: list of [training_dir, labels_dir]
    batch_size: int of desired number images per epoch
    n_channels: <-
    '''
    def __init__(self, list_IDs, data_dirs, batch_size, absolute_max_string_len = 32, output_size = 28, shuffle = True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs # [s1_path, s1_align_paths]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.output_size = output_size
        self.align = enumerate_align_hash(data_dirs[1], absolute_max_string_len)

    def data_gen(self, list_IDs_temp):
        '''
        Preprocesses the data
        Args:
            batch_x, batch_y
        Returns
            x, y
        '''
        import skvideo.io
        x = []
        y = []
        label_length = []
        input_length = []
        for file_id in list_IDs_temp:
            # for file_x, file_y in zip(batch_x, batch_y):
            file_x = os.path.join(self.data_dirs[0] + file_id + '.mpg')
            file_y = self.align[file_id]#os.path.join(self.data_dirs[1] + file_id + '.align')
            
            # assume 4d
            video = np.asarray(skvideo.io.vread(file_x))
            
            label_length.append(file_y.label_length) # CHANGED [A] -> A, CHECK!
            # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
            input_length.append(video.shape[0]) # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
            x.append(video), y.append(file_y.padded_label)
            
        inputs = {'the_input': x,
          'the_labels': y,
          'input_length': input_length,
          'label_length': label_length,
         }
        outputs = {'ctc': np.zeros([output_size])}
        return (inputs, outputs)
#
# list_IDs = get_list_IDs(s1_path)
# data_dirs = [s1_path, s1_align_path]
# gen = FrameGenerator(list_IDs['train'], data_dirs, batch_size = 1)
# gen.__getitem__(1)
