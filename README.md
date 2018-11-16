# PyTorch misc
Collection of code snippets I've written for the [PyTorch discussion board](https://discuss.pytorch.org/).

All scripts were testes using the PyTorch 1.0 preview and torchvision `0.2.1`.

Additional libraries, e.g. `numpy` or `pandas`, are used in a few scripts.

Some scripts might be a good starter to create a tutorial.

## Overview

  * [accumulate_gradients](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/accumulate_gradients.py) - Comparison of accumulated gradients/losses to vanilla batch update.
  * [adaptive_batchnorm](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/adaptive_batchnorm.py)- Adaptive BN implementation using two additional parameters: `out = a * x + b * bn(x)`.
  * [adaptive_pooling_torchvision](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/adaptive_pooling_torchvision.py) - Example of using adaptive pooling layers in pretrained models to use different spatial input shapes.
  * [batch_norm_manual](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/batch_norm_manual.py) - Comparison of PyTorch BatchNorm layers and a manual calculation.
  * [change_crop_in_dataset](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/change_crop_in_dataset.py) - Change the image crop size on the fly using a Dataset.
  * [conv_rnn](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/conv_rnn.py) - Combines a 3DCNN with an RNN; uses windowed frames as inputs.
  * [csv_chunk_read](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/csv_chunk_read.py) - Provide data chunks from continuous .csv file.
  * [densenet_forwardhook](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/densenet_forwardhook.py) - Use forward hooks to get intermediate activations from `densenet121`. Uses separate modules to process these activations further.
  * [edge_weighting_segmentation](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/edge_weighting_segmentation.py) - Apply weighting to edges for a segmentation task.
  * [image_rotation_with_matrix](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/image_rotation_with_matrix.py) - Rotate an image given an angle using 1.) a nested loop and 2.) a rotation matrix and mesh grid.
  * [LocallyConnected2d](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/LocallyConnected2d.py) - Implementation of a locally connected 2d layer.
  * [mnist_autoencoder](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/mnist_autoencoder.py) - Simple autoencoder for MNIST data. Includes visualizations of output images, intermediate activations and conv kernels.
  * [mnist_permuted](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/mnist_permuted.py) - MNIST training using permuted pixel locations.
  * [model_sharding_data_parallel](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/model_sharding_data_parallel.py) - Model sharding with `DataParallel` using 2 pairs of 2 GPUs.
  * [momentum_update_nograd](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/momentum_update_nograd.py) - Script to see how parameters are updated when an optimizer is used with momentum/running estimates, even if gradients are zero.
  * [shared_array](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/shared_array.py) - Script to demonstrate the usage of shared arrays using multiple workers.
  * [unet_demo](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/unet_demo.py) - Simple UNet demo.
  * [weighted_sampling](https://raw.githubusercontent.com/ptrblck/pytorch_misc/master/weighted_sampling.py) - Usage of WeightedRandomSampler using an imbalanced dataset with class imbalance 99 to 1.


Feedback is very welcome!