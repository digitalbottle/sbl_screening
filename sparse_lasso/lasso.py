import numpy as np
import tensorflow as tf
import pdb
from time import time

lasso_delta = tf.constant(1e-6)
optimizer = tf.keras.optimizers.Adam(lr=0.1)
max_iter = tf.constant(10000)
grad_bound = tf.constant(0.001)

def lasso_regression(lambda_lasso, taregt_norm, dictionary_norm):
  dictionary_norm = tf.transpose(dictionary_norm)
  taregt_norm = tf.transpose(taregt_norm)
  dict_size = tf.shape(dictionary_norm)
  n = dict_size[1]
  w_iter = tf.Variable(tf.random.uniform([n, 1]), constraint=tf.keras.constraints.NonNeg())
  bg = tf.Variable(tf.random.uniform([1]) - 0.5)

  target = tf.reduce_mean(tf.square(taregt_norm))
  
  @tf.function
  def train_step():
    with tf.GradientTape() as tape:
      loss = 0.5 * tf.reduce_mean(tf.square(tf.linalg.matmul(dictionary_norm, w_iter) + bg - taregt_norm)) + \
        lambda_lasso * tf.norm(w_iter, ord=1)
      mse = tf.reduce_mean(tf.square(tf.linalg.matmul(dictionary_norm, w_iter) + bg - taregt_norm))
    gradient = tape.gradient(loss, [w_iter, bg])
    optimizer.apply_gradients(zip(gradient, [w_iter, bg]))
    return mse, gradient, loss
  
  start = time()
  mse_pre = tf.constant(np.inf)
  
  for _ in tf.range(max_iter):
    mse, grad, loss = train_step()
    print("grad: %.8f"%(tf.reduce_mean(tf.math.abs(grad[0])).numpy()), "mse/target: %.2f"%(mse.numpy() / target.numpy()), end="\n")
    if (mse_pre - mse) / mse < lasso_delta and \
       tf.reduce_mean(tf.math.abs(grad[0])) < grad_bound and \
       mse / target < tf.constant(0.02):
      print("success")
      break
    else:
      mse_pre = mse
  else:
    print("fail")
  
  stop = time()
  print("N", tf.reduce_sum(tf.cast(w_iter > 0, dtype=tf.int32)).numpy(), \
        "Max_w", tf.reduce_max(w_iter).numpy(), "Loss", loss.numpy(), \
        "grad", tf.norm(grad[0], ord=1).numpy(), "mse/target", mse.numpy() / target.numpy(), \
        "time", stop - start, end='\n')
  print("target", target.numpy())
  return w_iter, bg
