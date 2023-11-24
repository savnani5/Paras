# SobelOperator

1) We train a SobelOperator model class having 2 5x5 kernels (Grad_x and Grad_y) with learnable parameters. The kernels are initialized with random numbers from normal distribution. 
2) For data preparation we save the input and output images beforehand in custom hierarchy so that we don't apply *cv2.Sobel* for ground truth at each image retrieval and slow down training.
3) One caveat I noticed was that ouput images as they only have edges have majorly black pixels so this is a form of data imbalance and network can learn to bias the parameters to output zeros in pixels and still have a low loss. Solution: We can use Focal loss to focus more on pixels with higher values for the network.


## How to prepare data?

1) Download the **coco_minitrain_25k** dataset to the current directory [Download link](https://ln5.sync.com/dl/0324da1d0/rmi7abjx-2dj4ktii-d9jcwgc5-s7fwwrb7) 
[Github link](https://github.com/giddyyupp/coco-minitrain).

2) Run the following command:
```
python prep_dataset.py --path <Path to dataset directory> 
```
**NOTE:** You only need to provide path argument if you have downloaded the dataset somewhere else, otherwise it will take current dir as default.

## How to run training?

Run following command:
```
python train.py 
```

**NOTE:** Edit config.py to change data preparation or training configurations. 

## Answers to questions:

1) For this solution, I am resizing the input and output images to a standard size 250x250, so the network is agnostic to input size (another option is to pass image pathces to the network, as we don't want to learn any global information).
2) We are padding the original image to get the same size for output after convolution so that we can calculate the pixel wise loss.
3) Yes, no linear layers, just 2 convolution layers to learn gradients in X and Y direction.
4) Yes, Pytorch has many optimizations to speed up convolution operation on gpu's, one major optimization is using Toeplitz matrix to do convolution by using mtrix-matrix multiplication instead of naively using sliding window techniques to multiply kernels. More details (https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication)
5) To reduce the model size we can use smaller weight matrices maybe 3x3 or 2x2 to learn the filters.
Also if we wanna keep the same filter size, we can constrain the center column in Gx and center row in Gy to zeros, we can fix these parameters (don't require gradients) and can reduce the total parameters to from 50 (25+25) to 40 (20% reduction by using sparse matrices xP).
6) Well, having a clear loss limit for this kind of low parameter optimization is ambigous. We can check if the avergae delta loss value is more or less constant and the loss is asymptotically approaching 0, we can conclude the training.   
7) Benefits of deeper network:
    - For general img2img transformations, network can see different fields of view and can learn global and local correlations in image. (NOTE: Sobel operator is a local function, I don't think we need a deeper model for this, in theory we can train with patches of images too, no global information is required to learn this kernel.)
    - By making the network deeper we increase the network's capacity thus overparametrization can make networks task easier.
    - More generally for these img2img type of tasks we use UNET variants, which introduce a bottleneck as we deepen the network to compress and use only essential information to reconstruct the image. This also introdues learning problems which are mititgated by skip connections.  

## Extra Credit:

For generalization in the current setup we can just use a single weight matrix to learn the filter and apply that kernel on the image in forward function using *F.conv2d*. The limitation of this setup is the network won't learn and we might need a deeper network potentially with a bottleneck to learn any arbitray kernel/combination of kernels, alluding to the benefits of deep networks mentioned above. 
