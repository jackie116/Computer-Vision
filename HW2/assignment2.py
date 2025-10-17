# -*- coding: utf-8 -*-
"""EEP 596 HW2
"""

import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

class ComputerVisionAssignment():
  def __init__(self) -> None:
    self.ant_img = cv2.imread('ant_outline.png')
    self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def floodfill(self, seed = (0, 0)):

    # Define the fill color (e.g., bright green)
    fill_color = (0, 0, 255)  # (B, G, R)
    # Create a copy of the input image to keep the original image unchanged
    output_image = self.ant_img.copy()
    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray_image.shape
    visited_map = np.zeros_like(gray_image)
    # Define a stack for floodfill
    stack = [seed]
    target_color = gray_image[seed[1], seed[0]]
    
    while stack:
      x, y = stack.pop()
      
      if x < 0 or x >= image_width or y < 0 or y >= image_height:
        continue
      
      if visited_map[y, x] == 1:
        continue
      
      visited_map[y, x] = 1  # Mark as visited

      if gray_image[y, x] == target_color:
        output_image[y, x] = fill_color
        
        stack.append((x + 1, y))  # Right
        stack.append((x - 1, y))  # Left
        stack.append((x, y + 1))  # Down
        stack.append((x, y - 1))  # Up

    #cv2.imwrite('floodfille.jpg', output_image)
    return output_image

  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """
    kernel = [0.25, 0.5, 0.25] # 1D Gaussian kernel
    image = self.cat_eye.copy().astype(np.float64)
    H, W = image.shape
    self.blurred_images = []

    for i in range(5):
      # Apply convolution
      horizontal_image = np.zeros((H, W), dtype=np.float64)
      for y in range(H):
        for x in range(W):
          left = image[y, x-1] if x-1 >= 0 else 0
          mid = image[y, x]
          right = image[y, x+1] if x+1 < W else 0
          
          horizontal_image[y, x] = (kernel[0] * left +
                                    kernel[1] * mid + 
                                    kernel[2] * right)
      
      vertical_image = np.zeros((H, W), dtype=np.float64)
      for x in range(W):
        for y in range(H):
          top = horizontal_image[y-1, x] if y-1 >= 0 else 0
          mid = horizontal_image[y, x]
          bottom = horizontal_image[y+1, x] if y+1 < H else 0

          vertical_image[y, x] = (kernel[0] * top +
                         kernel[1] * mid + 
                         kernel[2] * bottom)
          
      image = np.clip(np.floor(vertical_image + 0.5), 0, 255).astype(np.uint8)
      
      # Store the blurred image        
      self.blurred_images.append(image.copy())
        
      #cv2.imwrite(f'gaussain blur {i}.jpg', image)
    return self.blurred_images

  def gaussian_derivative_vertical(self):
    # Define kernels
    smoothing_kernel = [0.25, 0.5, 0.25] # 1D Gaussian derivative kernel
    sobel_kernel = [0.5, 0, -0.5] # 1D Sobel kernel
    
    # Store images
    self.vDerive_images = []
    for i in range(5):
      # Apply horizontal and vertical convolution
      image = self.blurred_images[i].copy().astype(np.float64)
      H, W = image.shape
      # First apply smoothing in horizontal direction
      horizontal_image = np.zeros((H, W), dtype=np.float64)
      for y in range(H):
        for x in range(W):
          left = image[y, x-1] if x-1 >= 0 else 0
          mid = image[y, x]
          right = image[y, x+1] if x+1 < W else 0
          
          horizontal_image[y, x] = (smoothing_kernel[0] * left +
                                    smoothing_kernel[1] * mid + 
                                    smoothing_kernel[2] * right)
          
      # Then apply sobel in vertical direction
      vertical_image = np.zeros((H, W), dtype=np.float64)
      for x in range(W):
        for y in range(H):
          top = horizontal_image[y-1, x] if y-1 >= 0 else 0
          mid = horizontal_image[y, x]
          bottom = horizontal_image[y+1, x] if y+1 < H else 0

          vertical_image[y, x] = (sobel_kernel[0] * top +
                         sobel_kernel[1] * mid + 
                         sobel_kernel[2] * bottom)

      image = np.clip(np.round(2 * vertical_image + 127), 0, 255).astype(np.uint8)
      self.vDerive_images.append(image.copy())
      #cv2.imwrite(f'vertical {i}.jpg', image)
    return self.vDerive_images

  def gaussian_derivative_horizontal(self):
    #Define kernels
    smoothing_kernel = [0.25, 0.5, 0.25] # 1D Gaussian derivative kernel
    differentiating_kernel = [0.5, 0, -0.5] # 1D differentiating kernel
    # Store images after computing horizontal derivative
    self.hDerive_images = []

    for i in range(5):

      # Apply horizontal and vertical convolution
      image = self.blurred_images[i].copy().astype(np.float64)
      H, W = image.shape
      # First apply smoothing in vertical direction
      vertical_image = np.zeros((H, W), dtype=np.float64)
      for x in range(W):
        for y in range(H):
          top = image[y-1, x] if y-1 >= 0 else 0
          mid = image[y, x]
          bottom = image[y+1, x] if y+1 < H else 0

          vertical_image[y, x] = (smoothing_kernel[0] * top +
                                  smoothing_kernel[1] * mid + 
                                  smoothing_kernel[2] * bottom)
          
      # Then apply derivative in horizontal direction
      horizontal_image = np.zeros((H, W), dtype=np.float64)
      for y in range(H):
        for x in range(W):
          left = vertical_image[y, x-1] if x-1 >= 0 else 0
          mid = vertical_image[y, x]
          right = vertical_image[y, x+1] if x+1 < W else 0
          
          horizontal_image[y, x] = (differentiating_kernel[0] * left +
                         differentiating_kernel[1] * mid + 
                         differentiating_kernel[2] * right)

      image = np.clip(np.round(2 * horizontal_image + 127), 0, 255).astype(np.uint8)
      self.hDerive_images.append(image.copy())
      #cv2.imwrite(f'horizontal {i}.jpg', image)
    return self.hDerive_images

  def gradient_magnitute(self):
    #Define kernels
    smoothing_kernel = [0.25, 0.5, 0.25] # 1D Gaussian derivative kernel
    differentiating_kernel = [-0.5, 0, 0.5] # 1D differentiating kernel

    # Store the computed gradient magnitute
    self.gdMagnitute_images =[]

    for i in range(5):
      image = self.blurred_images[i].copy().astype(np.float64)
      H, W = image.shape
      # |gx|
      gx_image = np.zeros((H, W), dtype=np.float64)
      # First apply smoothing in vertical direction
      vertical_image = np.zeros((H, W), dtype=np.float64)
      for x in range(W):
        for y in range(H):
          top = image[y-1, x] if y-1 >= 0 else 0
          mid = image[y, x]
          bottom = image[y+1, x] if y+1 < H else 0

          vertical_image[y, x] = (smoothing_kernel[0] * top +
                                  smoothing_kernel[1] * mid + 
                                  smoothing_kernel[2] * bottom)
      
      # Then apply derivative in horizontal direction
      for y in range(H):
        for x in range(W):
          left = vertical_image[y, x-1] if x-1 >= 0 else 0
          mid = vertical_image[y, x]
          right = vertical_image[y, x+1] if x+1 < W else 0
          
          gx_image[y, x] = abs(differentiating_kernel[0] * left +
                            differentiating_kernel[1] * mid + 
                            differentiating_kernel[2] * right)
      # |gy|
      gy_image = np.zeros((H, W), dtype=np.float64)
      # First apply smoothing in horizontal direction
      horizontal_image = np.zeros((H, W), dtype=np.float64)
      for y in range(H):
        for x in range(W):
          left = image[y, x-1] if x-1 >= 0 else 0
          mid = image[y, x]
          right = image[y, x+1] if x+1 < W else 0
          
          horizontal_image[y, x] = (smoothing_kernel[0] * left +
                                    smoothing_kernel[1] * mid + 
                                    smoothing_kernel[2] * right)
      # Then apply derivative in vertical direction
      for x in range(W):
        for y in range(H):
          top = horizontal_image[y-1, x] if y-1 >= 0 else 0
          mid = horizontal_image[y, x]
          bottom = horizontal_image[y+1, x] if y+1 < H else 0

          gy_image[y, x] = abs(differentiating_kernel[0] * top +
                            differentiating_kernel[1] * mid + 
                            differentiating_kernel[2] * bottom)

      image = np.clip(4 * np.round(gx_image + gy_image), 0, 255).astype(np.uint8)
      self.gdMagnitute_images.append(image.copy())
      #cv2.imwrite(f'gradient {i}.jpg', image)
    return self.gdMagnitute_images
    
  def scipy_convolve(self):
    # Define the 2D smoothing kernel
    smoothing_kernel = [0.25, 0.5, 0.25] # 1D Gaussian derivative kernel
    differentiating_kernel = [0.5, 0, -0.5] # 1D differentiating kernel
    kernel_2d = np.outer(differentiating_kernel, smoothing_kernel)

    # Store outputs
    self.scipy_smooth = []

    for i in range(5):
      # Perform convolution
      image = self.blurred_images[i].copy().astype(np.float64)
      image = scipy.signal.convolve2d(image, kernel_2d, mode='same', boundary='fill', fillvalue=0)
      image = np.clip(2 * np.round(image) + 127, 0, 255).astype(np.uint8)
      
      self.scipy_smooth.append(image.copy())
      #cv2.imwrite(f'scipy smooth {i}.jpg', image)
    return self.scipy_smooth

  def box_filter(self, num_repetitions):
    # Define box filter
    box_filter = [1, 1, 1]
    out = [1, 1, 1]

    plt.figure(figsize=(7, 4))

    for round in range(num_repetitions):
      # Perform 1D conlve
      out_length = len(out) + len(box_filter) - 1
      new_out = np.zeros(out_length)
      for i in range(len(out)):
        for j in range(len(box_filter)):
          new_out[i + j] += out[i] * box_filter[j]
      
      out = new_out
      
      filt_norm = np.array(out, dtype=float) / np.sum(out)
      plt.plot(filt_norm, marker='o', label=f'{round + 1} convolutions')
    
    plt.title('Repeated Box Filtering')
    plt.xlabel('X (pixels)')
    plt.ylabel('G(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # Task 1 floodfill
    floodfill_img = ass.floodfill((100, 100))

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # Task 3 Convolution for differentiation along the vertical direction
    vertical_derivative = ass.gaussian_derivative_vertical()

    # Task 4 Differentiation along another direction along the horizontal direction
    horizontal_derivative = ass.gaussian_derivative_horizontal()

    # Task 5 Gradient magnitude.
    Gradient_magnitude = ass.gradient_magnitute()

    # Task 6 Built-in convolution
    scipy_convolve = ass.scipy_convolve()

    # Task 7 Repeated box filtering
    box_filter = ass.box_filter(5)
