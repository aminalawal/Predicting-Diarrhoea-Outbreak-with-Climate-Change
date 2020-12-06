
#!/usr/bin/env ipython
# Evaluation of models
#

import json
import pdb
import numpy as np
import pandas as pd
from eugenium_mmd import MMD_3_Sample_Test
from scipy.stats import ks_2samp
import mmd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# for keras
import keras
import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.backend import clear_session

import model
import data_utils
import plotting

import pickle



import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())

# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)


a = np.zeros((20000,0))

synth_X = model.sample_trained_model('test', epoch=150, num_samples=20000, C_samples=a)
np.save('C:/Users/Amina Lawal/Pictures/RGAN-master/RGAN-master/experiments/synthetic_dataset/gen_data150.npy', synth_X)