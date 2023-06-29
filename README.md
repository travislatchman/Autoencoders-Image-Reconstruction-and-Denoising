# Autoencoders-Image-Reconstruction-and-Denoising

**Brief Description:** The project involved designing a convolutional auto-encoder network for image reconstruction using the CIFAR10 dataset. It also explored using this autoencoder for denoising.

## Implementation:
* Designed a convolutional auto-encoder network with a two block encoder and decoder.
* Trained the model on the CIFAR10 dataset for image reconstruction.
* Applied Gaussian noise to the image tensor before passing it to the auto-encoder for denoising tasks.

## Auto-encoder Design

The architecture consists of two main components: an encoder and a decoder. The encoder and decoder
are both implemented as sequential neural networks using the nn.Sequential class.

Encoder: The encoder is designed to reduce the dimensionality of the input data while preserving the
important features. It consists of two convolutional layers (nn.Conv2d), both followed by a ReLU activation
function (nn.ReLU). The first convolutional layer has 3 input channels, 8 output channels, a 3x3 kernel,
stride of 1, and no padding. The second convolutional layer has 8 input channels, 8 output channels, a 3x3
kernel, stride of 1, and no padding.

Decoder: The decoder is responsible for reconstructing the original input from the encoded representation. It consists of two transposed convolutional layers (nn.ConvTranspose2d), both followed by an
activation function. The first transposed convolutional layer has 8 input channels, 8 output channels, a
3x3 kernel, stride of 1, and no padding, followed by a ReLU activation function. The second transposed
convolutional layer has 8 input channels, 3 output channels, a 3x3 kernel, stride of 1, and no padding, followed by a Sigmoid activation function. The forward method defines the forward pass of the AutoEncoder. The input, x, is first passed through the encoder, and the output is then passed through the decoder. The
output of the decoder is returned as the final output of the AutoEncoder.

The purpose of the AutoEncoder is to learn a compact representation of the input data, then reconstruct
the input from that compact representation. This can be useful for tasks such as image compression,
denoising, or learning low-dimensional embeddings for data visualization or feature extraction.


## Results

* Part 1: Visualize 10 inputs with their corresponding 10 reconstructions. I have visualizations after training
the auto-encoder for both 20 and 50 epochs to see if it makes a difference.  

![image](https://github.com/travislatchman/Autoencoders-Image-Reconstruction-and-Denoising/assets/32372013/93054ad5-613b-4daa-b170-ba23a15f5b86) 

* Part 2: Visualize 10 noisy inputs with their corresponding 10 denoised images (after training for 20 and 50
epochs). Report the mean peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM)
obtained by your model on the test set

![image](https://github.com/travislatchman/Autoencoders-Image-Reconstruction-and-Denoising/assets/32372013/0c133160-fe35-4517-966e-16709e7301d0)

  20 Epochs  
  Mean PSNR: 24.79  
  Mean SSIM: 0.8988  
    
  50 epochs  
  Mean PSNR: 24.86  
  Mean SSIM: 0.9045  

  The PSNR levels are expected to be this low because the original images in the CIFAR10 dataset
(without noise) are blurry and have low resolution. The Peak Signal-to-Noise Ratio measures the
ratio between the maximum possible power of the original signal (image) and the power of the noise
(distortion). It is expressed in decibels (dB).
The SSIM is another metric used to assess the similarity between two images, with values ranging
from 0 to 1. A higher SSIM value indicates greater similarity between the reconstructed image and
the original image. A mean SSIM of 0.8988 for 20 epochs and 0.9045 for 50 epochs indicates a
reasonably good level of similarity between the original and reconstructed images. The auto-encoder
has a relatively good ability to preserve the structural and textural information in the original images,
despite the presence of noise.
These metrics suggest that the auto-encoder has good ability to reconstruct original images from
noisy inputs in the CIFAR10 dataset. The improvement in both metrics when training for more
epochs (from 20 to 50) is marginal, which suggests that the auto-encoder’s performance might have
plateaued.

