import numpy as np
import tensorflow as tf
import pdb
from time import time
import sampling.normalize
import os
import cv2
from skimage.io import imread
from skimage.io import imsave



# @tf.function
def pan_regression(lambda_pan, taregt_norm, dictionary_norm,dict_list, point_list, gt_list, save_path):
  delta = tf.constant(1e-6)
  MAXITER = tf.constant(100)
  result_path = os.path.join(save_path, "%.8f"%lambda_pan)
  if not os.path.isdir(result_path):
    os.makedirs(os.path.join(result_path, "pan"))
    os.makedirs(os.path.join(result_path, "pan_gt"))
  # ---------------------------------
  # Dataset
  # ---------------------------------
  dictionary_norm = tf.transpose(dictionary_norm)
  taregt_norm = tf.transpose(taregt_norm)
  dict_size = tf.shape(dictionary_norm)
  N, n = dict_size[0], dict_size[1]
  
  # ---------------------------------
  # Draw result
  # ---------------------------------
  def draw_reconstruction(weights, bg, file_path_rec, file_path_pt):
    gt = np.reshape(taregt_norm, dict_list.shape[1:])
    gt_norm = np.uint8(sampling.normalize.std_Normalize(gt) * 255)
    reconstruction = tf.reshape(tf.linalg.matmul(dictionary_norm, weights) + bg, dict_list.shape[1:])
    reconstruction_norm = np.uint8(sampling.normalize.std_Normalize(reconstruction.numpy()) * 255)
    gt_new = cv2.cvtColor(cv2.resize(gt_norm, (1024, 1024), interpolation = cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
    img_new = cv2.cvtColor(cv2.resize(reconstruction_norm, (1024, 1024), interpolation = cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
    # draw pred points
    for theta in point_list[np.where(weights > 0)[0], :]:
      pointx = theta[0] / dict_list.shape[2] * 1024
      pointy = theta[1] / dict_list.shape[1] * 1024
      dx = 0.5 / dict_list.shape[2] * 1024
      dy = 0.5 / dict_list.shape[1] * 1024
      ratio = np.sqrt((1024 / dict_list.shape[2]) * (1024 / dict_list.shape[1]))
      gt_new = cv2.circle(gt_new, (int(round(pointx + dx)), int(round(pointy + dy))), int(round(0.3 * ratio * theta[2])), (255, 0, 255), 2)
    # drew gt points
    for theta in gt_list[:, 1:]:
      pointx = theta[0] / dict_list.shape[2] * 1024
      pointy = theta[1] / dict_list.shape[1] * 1024
      dx = 0.5 / dict_list.shape[2] * 1024
      dy = 0.5 / dict_list.shape[1] * 1024
      ratio = np.sqrt((1024 / dict_list.shape[2]) * (1024 / dict_list.shape[1]))
      r = int(round(0.05 * ratio * theta[2]))
      gt_new = cv2.rectangle(gt_new, (int(round(pointx + dx)) - r, int(round(pointy + dy)) - r), (int(round(pointx + dx)) + r, int(round(pointy + dy)) + r), (0, 255, 0), 2)
    
    imsave(file_path_rec, img_new)
    imsave(file_path_pt, gt_new)
    print("save tiff :%s, %s"%(file_path_rec, file_path_pt))
    return
  
  # ---------------------------------
  # Observed variable
  # ---------------------------------
  w_estimate = tf.zeros([n, 1])
  bg_estimate = tf.zeros([1, 1])
  
  # ---------------------------------
  # Temporary variables
  # ---------------------------------
  Gamma = tf.zeros([n, 1])
  UU = tf.zeros([n, 1])
  dictionary_iter = tf.zeros([N, N])

  w_iter = tf.Variable(tf.random.uniform([dictionary_norm.shape[1], 1]), constraint=tf.keras.constraints.NonNeg())
  bg = tf.Variable(tf.random.uniform([1]) - 0.5)

  U_iter = tf.ones([n, 1])
  U_next = tf.ones([n, 1])
  w_before = tf.zeros([n, 1])
  # ---------------------------------
  # Main Loop
  # ---------------------------------
  target = tf.reduce_mean(tf.square(taregt_norm))
  mse, loss = tf.constant(0.1), tf.constant(0.1)
  grad = [tf.zeros(tf.shape(w_iter)), tf.zeros(tf.shape(bg))]

  print("Finding a sparse feasible point using l1-norm heuristic ...\n")
  for i in tf.range(MAXITER):
    print("--------------%03d---------------"%i.numpy())
    # TODO sparse Tensor
    U_iter = U_next
    # ---------------------------------
    # Lasso PanWei learning
    # ---------------------------------
    w_iter.assign(tf.random.uniform([dictionary_norm.shape[1], 1]))
    bg.assign(tf.random.uniform([1]) - 0.5)
    PanWei_learning = PanWei(taregt_norm)
    w_estimate, bg_estimate, loss, grad, mse = PanWei_learning(lambda_pan, U_iter, dictionary_norm, taregt_norm, w_iter, bg)
    print("N", tf.reduce_sum(tf.cast(w_iter > 0, dtype=tf.int32)).numpy(), \
          "Min_w", tf.reduce_min(w_iter).numpy(), "Max_w", tf.reduce_max(w_iter).numpy(), \
          "Loss", loss.numpy(), "grad", tf.norm(grad[0], ord=1).numpy(), \
          "mse/target", mse.numpy() / target.numpy(), end='\n')
    print("target", target.numpy())
    # ---------------------------------
    # Sparse Identification
    # ---------------------------------
    Gamma = U_iter ** (-1) * w_estimate
    dictionary_iter = lambda_pan * tf.eye(N) + \
        tf.linalg.matmul(tf.linalg.matmul(dictionary_norm, tf.linalg.tensor_diag(tf.reshape(Gamma, [n]))), dictionary_norm, transpose_b=True)
    UU = tf.linalg.tensor_diag_part(tf.linalg.matmul(dictionary_norm, tf.linalg.solve(dictionary_iter, dictionary_norm), transpose_a=True))
    U_next = tf.reshape(tf.abs(tf.sqrt(UU)), [n, 1])
    w_estimate = w_estimate * tf.where((w_estimate ** 2 /  tf.norm(w_estimate, ord=2) ** 2) < delta, tf.zeros(tf.shape(w_estimate)), tf.ones(tf.shape(w_estimate)))
    print("N_w:", tf.reduce_sum(tf.cast(w_estimate > 0, dtype=tf.int32)).numpy())
    draw_reconstruction(w_estimate.numpy(), bg_estimate.numpy(), os.path.join(result_path, "pan/%03d.tif"%i), os.path.join(result_path, "pan_gt/%03d.tif"%i))
    # ---------------------------------
    # stopping criterion
    # ---------------------------------
    if i > 1 and tf.reduce_max(tf.abs(w_estimate - w_before)) < tf.constant(1e-3):
      break
    else:
      w_before = w_estimate
  return w_estimate, bg_estimate

# ---------------------------------
# Lasso PanWei learning
# ---------------------------------
def PanWei(taregt_norm):
  # Lasso
  lasso_delta = tf.constant(0.00)
  optimizer = tf.keras.optimizers.Adam(lr=0.1)
  max_iter = tf.constant(10000)
  grad_bound = tf.constant(1e-4)
  target = tf.reduce_mean(tf.square(taregt_norm))
  # @tf.function
  def PanWei_learning(lambda_pan, U_iter, dictionary_norm, taregt_norm, w_iter, bg):
    def train_step():
      with tf.GradientTape() as tape:
        loss = 0.5 * tf.reduce_mean(tf.square(tf.linalg.matmul(dictionary_norm, w_iter) + bg - taregt_norm)) + \
          lambda_pan * tf.norm(U_iter * w_iter, ord=1)
        mse = tf.reduce_mean(tf.square(tf.linalg.matmul(dictionary_norm, w_iter) + bg - taregt_norm))
      gradient = tape.gradient(loss, [w_iter, bg])
      optimizer.apply_gradients(zip(gradient, [w_iter, bg]))
      return mse, gradient, loss

    mse_pre = tf.constant(np.inf)
    mse, loss = tf.constant(0.1), tf.constant(0.1)
    grad = [tf.zeros(tf.shape(w_iter)), tf.zeros(tf.shape(bg))]

    for i in tf.range(max_iter):
      mse, grad, loss = train_step()
      # print("grad: %.8f"%(tf.reduce_mean(tf.math.abs(grad[0])).numpy()), "mse/target: %.2f"%(mse.numpy() / target.numpy()), end="\n")
      if (mse_pre - mse) / target < lasso_delta and \
        tf.reduce_mean(tf.math.abs(grad[0])) < grad_bound and \
        mse < target and \
        i > tf.constant(3000):
        # print("success", i.numpy())
        break
      else:
        mse_pre = mse
    else:
      print("fail")
    return w_iter, bg, loss, grad, mse
  # w_estimate, bg_estimate, loss, grad, mse = \
  #   PanWei_learning(lambda_pan, U_iter, dictionary_norm, taregt_norm, w_iter, bg)
  return PanWei_learning