import os
import pickle
import hashlib
from datetime import datetime
from _1_config import *

def check_tensorflow_settings():
    print(tf.__version__)
    print(tf.__file__)
    # List available devices
    print(tf.config.list_physical_devices())

    # Print TensorFlow environment variables
    print("\nTensorFlow Environment Variables:")
    print(f"TF_NUM_INTRAOP_THREADS: {os.environ.get('TF_NUM_INTRAOP_THREADS')}")
    print(f"TF_NUM_INTEROP_THREADS: {os.environ.get('TF_NUM_INTEROP_THREADS')}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"TF_ENABLE_ONEDNN_OPTS: {os.environ.get('TF_ENABLE_ONEDNN_OPTS')}")
    print(f"TF_CPU_ENABLE_AVX512: {os.environ.get('TF_CPU_ENABLE_AVX512')}")
    print(f"TF_CPU_ENABLE_AVX2: {os.environ.get('TF_CPU_ENABLE_AVX2')}")
    print(f"TF_CPU_ENABLE_FMA: {os.environ.get('TF_CPU_ENABLE_FMA')}")
    print(f"TF_ENABLE_XLA: {os.environ.get('TF_ENABLE_XLA')}")
    print(f"TF_ENABLE_AUTO_MIXED_PRECISION: {os.environ.get('TF_ENABLE_AUTO_MIXED_PRECISION')}")

    # Print current thread settings
    print("\nTensorFlow Threading Info:")
    print(f"Intra-op parallelism threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
    print(f"Inter-op parallelism threads: {tf.config.threading.get_inter_op_parallelism_threads()}")

    # Print current optimizer settings
    print("\nOptimizer Settings:")
    # print(f"oneDNN enabled: {tf.test.is_built_with_one_dnn()}")

def calculate_hash(data):
    return hashlib.md5(pickle.dumps(data)).hexdigest()

def save_data(data, filename):
    # Create the stock_data directory if it doesn't exist
    os.makedirs('stock_data', exist_ok=True)
    
    # Update the filename to include the stock_data directory
    filepath = os.path.join('stock_data', filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump((data, datetime.now().date()), f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def data_needs_update(filename):
    if not os.path.exists(filename):
        return True
    saved_data, save_date = load_data(filename)
    return save_date != datetime.now().date()
