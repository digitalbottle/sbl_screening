import h5py
import pdb
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import sparse_lasso.sparse_linear_regression as slr
import sparse_lasso.lasso as sla
import utils.normalize



dict_h5 = h5py.File("./data/dictionary.h5", 'r')
target_h5 = h5py.File("./data/target.h5", 'r')

dict_list = np.asarray(dict_h5["dictionary"])
point_list = np.asarray(dict_h5["point"])
target_size = np.asarray(target_h5["target_num"])
target_list = []
target_point_list = []

pdb.set_trace()
for i in range(target_size):
  target_list.append(np.asarray(target_h5["targets_%02d"%i]))
  target_point_list.append(np.asarray(target_h5["point_%02d"%i]))

# ---------------------------------
# Target 0
# ---------------------------------
dictionary = tf.reshape(dict_list, [dict_list.shape[0], -1])
taregt = tf.reshape(target_list, [1, -1])
dictionary_norm = tf.dtypes.cast(tf.math.l2_normalize(dictionary, axis=1), tf.float32)
taregt_norm = tf.dtypes.cast(tf.math.l2_normalize(taregt, axis=1), tf.float32)
lambda_max = tf.reduce_max(tf.reduce_sum(dictionary_norm * taregt_norm, axis=1))



# ---------------------------------
# Lasso
# ---------------------------------
# lambda_lasso = lambda_max * 1e-5
# w_lasso, bg_lasso = sla.lasso_regression(lambda_lasso, taregt_norm, dictionary_norm)
# draw_reconstruction(w_lasso, bg_lasso, "./result/lasso.png")

# ---------------------------------
# Pan Wei
# ---------------------------------
lambda_pan = lambda_max * 0.5
w_pan, bg_pan = slr.pan_regression(lambda_pan, taregt_norm, dictionary_norm, dict_list, point_list, target_point_list[0])

# lambda_pan = lambda_max * 4e-6
# w_pan, bg_pan = slr.pan_regression(lambda_pan, taregt_norm, dictionary_norm, dict_list, point_list, target_point_list[0])

