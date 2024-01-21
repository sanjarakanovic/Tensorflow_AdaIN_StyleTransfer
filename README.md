# AdaIN style transfer

This is an implementation of the paper: https://arxiv.org/pdf/1703.06868.pdf.


![image of style transfer network](https://github.com/sanjarakanovic/Tensorflow_AdaIN_StyleTransfer/blob/main/imgs/style-transfer-net.png)


The encoder utilizes a pre-trained VGG19 network up to relu4_1, extracting features from both content and style images.  
The AdaIN layer performs style transfer in the feature space, aligning the channel-wise mean and variance of the content input to match those of the style input.  
The decoder largely mirrors the encoder, with all pooling layers replaced by the nearest up-sampling. Up-sampling is preferred over transposed convolution to avoid checkerboard artifacts. Reflection padding is applied before convolution to mitigate border artifacts.


## Running on Google Colab

This project is designed to run seamlessly on Google Colab, and no additional installations are required. The basic implementation, or foundational code, is provided herein, serving as a robust foundation for conducting independent experiments.
To execute the code:

1. Open the provided notebook in Google Colab.
2. Execute the code cells sequentially to observe the intended workflow.
3. Experiment with different hyperparameters to get the best results.

## Results
Here are some of the test examples, with my own paintings as the style images. 

![image of results](https://github.com/sanjarakanovic/Tensorflow_AdaIN_StyleTransfer/blob/main/imgs/test.png)

## The impact of the weight of the style loss on results

The degree of stylization can be controlled during training of the decoder by adjusting the style weight hyperparameter in the total loss equation:

$$\ L = L_c + \lambda L_s $$


![image of different style weights](https://github.com/sanjarakanovic/Tensorflow_AdaIN_StyleTransfer/blob/main/imgs/style-weights.png)


## Runtime controls

Runtime controls are only applied at runtime using the same network,
without any modification to the training procedure.

## Content-style trade-off

The degree of style transfer can be controlled by interpolating between the affine parameters of AdaIN:  


$$\ T(c, s, α) = g((1 − α)f(c) + αAdaIN(f(c), f(s)))  $$

![image of content_style_trade_off](https://github.com/sanjarakanovic/Tensorflow_AdaIN_StyleTransfer/blob/main/imgs/trade-off.png)

## Style interpolation

Interpolating between a set of К style images is achieved by interpolating between feature maps:


$$\ T(c, s_1, s_2, \ldots, s_K, w_1, w_2, \ldots, w_K) = g(\sum_{k=1}^K w_k AdaIN(f(c), f(s_k)))  $$

![image of style_interpolation](https://github.com/sanjarakanovic/Tensorflow_AdaIN_StyleTransfer/blob/main/imgs/style-interpolation.png)

## Color control

Color control is achieved by matching the color histogram of the style image to that of the content image, then performing normal style transfer using the color-aligned style image as the style input.

![image of color_control](https://github.com/sanjarakanovic/Tensorflow_AdaIN_StyleTransfer/blob/main/imgs/color-control.png)

## Spatial control

Spatial control is achieved by performing AdaIN separately to different regions in the content feature maps using statistics from
different style inputs, so a binary mask is used.


![image of spatial_control](https://github.com/sanjarakanovic/Tensorflow_AdaIN_StyleTransfer/blob/main/imgs/spatial-control.png)


