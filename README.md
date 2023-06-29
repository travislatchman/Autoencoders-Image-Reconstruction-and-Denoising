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
