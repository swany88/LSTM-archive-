import os
import sys
sys.path.insert(0, '/home/erik/.local/lib/python3.12/site-packages')
import tensorflow as tf
import time
import pickle
import hashlib
from datetime import datetime

# Set optimal TensorFlow environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_SETTINGS'] = '1'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
os.environ['TF_ENABLE_XLA'] = '1'

# Add custom site-packages path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# TensorFlow and Keras imports
from keras_tuner import HyperModel
from keras_tuner.tuners import BayesianOptimization

# Data manipulation and visualization imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as sklearn_train_test_split, TimeSeriesSplit   
from statsmodels.graphics.tsaplots import plot_acf 

# Financial data imports
import yfinance as yf
from fredapi import Fred

# Signal processing and statistics imports
from scipy.fft import fft
from scipy.signal import detrend
from scipy.stats import kurtosis
from collections import defaultdict

