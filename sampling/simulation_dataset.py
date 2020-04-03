import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import pdb
from scipy.special import erf
import h5py 
import priors
import os
from normalize import std_Normalize, l2_Normalize
from skimage.io import imread
from skimage.io import imsave


pic_size = [28, 28]
point_number = 4
target_size = 200
dict_size = 3000
def point_spread_function_2d(p, meshgrid):
  X, Y = meshgrid
  x0 = p[0]
  y0 = p[1]
  sigmaxy = p[2]
  xk = X
  yk = Y
  I = ((-erf((-0.5 - x0 + xk)/(np.sqrt(2)*sigmaxy)) + erf((0.5 - x0 + xk)/(np.sqrt(2)*sigmaxy)))*
     (-erf((-0.5 - y0 + yk)/(np.sqrt(2)*sigmaxy)) + erf((0.5 - y0 + yk)/(np.sqrt(2)*sigmaxy))))/4.
  return I

def prior_transform_2d_dict(uniform_sample):
  ''' 
    map uniform distribution u ~ [0, 1] to prior distribution v 
    a = theta[0]
    x0 = theta[1]
    y0 = theta[2]
    sigmaxy = theta[3]
  '''
  priors_list = [priors.TopHat(mini=0.8, maxi=1.2),
                 priors.TopHat(mini=pic_size[0]*0.1, maxi=pic_size[0]*0.9), 
                 priors.TopHat(mini=pic_size[1]*0.1, maxi=pic_size[1]*0.9),
                 priors.TopHat(mini=2, maxi=4)]
  if len(uniform_sample) != (len(priors_list)):
    print(len(uniform_sample), len(priors_list))
    raise Exception("theta length not equal to priors number")
  theta_t = []
  for i, (u, p) in enumerate(zip(uniform_sample, priors_list)):
    func = p.unit_transform
    theta_t.append(func(u))
  return theta_t

def prior_transform_2d_target(uniform_sample):
  ''' 
    map uniform distribution u ~ [0, 1] to prior distribution v 
    a = theta[0]
    x0 = theta[1]
    y0 = theta[2]
    sigmaxy = theta[3]
  '''
  priors_list = [priors.TopHat(mini=0.8, maxi=1.2),
                 priors.TopHat(mini=pic_size[0]*0.2, maxi=pic_size[0]*0.8), 
                 priors.TopHat(mini=pic_size[1]*0.2, maxi=pic_size[1]*0.8),
                 priors.TopHat(mini=2, maxi=4)]
  if len(uniform_sample) != (len(priors_list)):
    print(len(uniform_sample), len(priors_list))
    raise Exception("theta length not equal to priors number")
  theta_t = []
  for i, (u, p) in enumerate(zip(uniform_sample, priors_list)):
    func = p.unit_transform
    theta_t.append(func(u))
  return theta_t

def simulation_dictionary_generator(meshgrid, psf, theta_length, prior_transform):
  I_list = []
  uniform_sample = np.random.rand(theta_length)
  theta = prior_transform(uniform_sample)
  point = theta[1:]
  I_list.append(psf(theta[1:], meshgrid))
  I_sim = np.sum(I_list, axis=0)
  return I_sim, point
  
def simulation_data_generator(meshgrid, psf, theta_length, prior_transform, noise_sigma, background, point_num):
  I_list = []
  point_list = []
  for _ in range(point_num):
    uniform_sample = np.random.rand(theta_length)
    theta = prior_transform(uniform_sample)
    point_list.append(theta)
    I_list.append(theta[0] * psf(theta[1:], meshgrid))
  I_sim = np.sum(I_list, axis=0)
  I_sim = std_Normalize(l2_Normalize(I_sim))
  I_sim_noise = np.random.normal(I_sim, np.max(I_sim) * noise_sigma)
  return I_sim, I_sim_noise, np.asarray(point_list)

def safe_mkdir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
  return

if __name__ == '__main__':
  ''' 2D microscopy simulation'''
  xs = np.linspace(0, pic_size[0]-1, pic_size[0])
  ys = np.linspace(0, pic_size[1]-1, pic_size[1])
  xx, yy= np.meshgrid(xs, ys)
  
  ''' Dictionary '''
  dict_file = "../data/simulate_dict/"
  safe_mkdir(dict_file)

  dict_list = []
  point_list = []
  dict_h5 = h5py.File(os.path.join(dict_file, "../dictionary.h5"), 'w')
  
  for i in range(dict_size):
    I_3d, point = simulation_dictionary_generator(meshgrid=[xx, yy], 
                                   psf=point_spread_function_2d, 
                                   theta_length=4, 
                                   prior_transform=prior_transform_2d_dict)
    dict_list.append(I_3d.copy())
    point_list.append(point)
    img = std_Normalize(I_3d) * 255
    img = img.astype(np.uint8)
    imsave(os.path.join(dict_file, "%05d.tif"%i), img.transpose(1, 0))
    print(os.path.join(dict_file, "%05d.tif"%i))
  dict_h5["dictionary"] = np.asarray(dict_list)
  dict_h5["point"] = np.asarray(point_list)
  dict_h5.close()
  ''' Target '''
  target_file = "../data/simulate_target/"
  safe_mkdir(target_file)
  target_list = []
  target_h5 = h5py.File(os.path.join(target_file, "../target.h5"), 'w')
  target_h5["target_num"] = target_size
  for i in range(target_size):
    I_3d, I_3d_noise, points = simulation_data_generator(meshgrid=[xx, yy],
                                  psf=point_spread_function_2d, 
                                  theta_length=4, 
                                  prior_transform=prior_transform_2d_target,
                                  noise_sigma=0.5,
                                  background=0.01,
                                  point_num=point_number)
    # No noise
    target_list = I_3d.copy()
    target_h5["targets_clean_%02d"%i] = np.asarray(target_list)
    # Noise
    target_list = I_3d_noise.copy()
    target_h5["targets_noise_%02d"%i] = np.asarray(target_list)
    target_h5["point_%02d"%i] = np.asarray(points)
    img = std_Normalize(I_3d_noise) * 255
    img = img.astype(np.uint8)
    imsave(os.path.join(target_file, "%05d.tif"%i), img.transpose(1, 0))
    print(os.path.join(target_file, "%05d.tif"%i))
  target_h5.close()

