import os
from glob import glob
import skvideo
skvideo.setFFmpegPath('C:\\Users\\Joseph\\Desktop\\Misc. Packages\\ffmpeg-20181203-8ef0fda-win64-static\\ffmpeg-20181203-8ef0fda-win64-static\\bin\\')

data_path = 'C:\\Users\\Joseph\\Desktop\\Grid Corpus\\Video\\'
s1_path = 'C:\\Users\\Joseph\\Desktop\\Grid Corpus\\Video\\s1.mpg_6000.part1\\s1\\video\\mpg_6000\\'
s1_align_path = 'C:\\Users\\Joseph\\Desktop\\Grid Corpus\\Align\\s1\\align\\'

s1_files = glob(s1_path + '*.mpg', recursive = True)
s1_align_files = glob(s1_align_path + '*.align', recursive = True)
# print([i for i in zip(s1_files, s1_align_files)])
