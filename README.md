# Towards Context-aware Convolutional Network for Image Restoration

Fangwei Hao, Ji Du, Weiyun Liang, Jing Xu, Xiaoxuan Xu,


 ## Abstract
Image restoration (IR) is a long-standing task to recover a high-quality image from its corrupted observation. Recently, Transformer-based algorithms and some attention-based convolutional neural networks (CNNs) have presented promising results on several IR tasks. However, existing convolutional residual building modules for IR encounter limited ability to map inputs into high-dimensional and non-linear feature spaces, and their local receptive fields have difficulty in capturing long-range context information like Transformer. Besides, CNN-based attention modules for IR either face static abundant parameters or have limited receptive fields. To address the first issue, we propose an efficient residual star module (ERSM) that includes context-aware "star operation" (element-wise multiplication) to contextually map features into exceedingly high-dimensional and non-linear feature spaces, which greatly enhances representation learning. To further boost the extraction of contextual information, as for the second issue, we propose a large dynamic integration module (LDIM) which possesses an extremely large receptive field. Thus, LDIM can dynamically and efficiently integrate more contextual information that helps to further significantly improve the reconstruction performance. Integrating ERSM and LDIM into an U-shaped backbone, we propose a context-aware convolutional network (CCNet) with powerful context-aware ability for contextual high-dimensional mapping and abundant contextual information. Extensive experiments show that our CCNet with low model complexity achieves superior performance compared to other state-of-the-art IR methods on several IR tasks, including image dehazing, image motion deblurring, and image desnowing.

## Installation
For installing, follow these instructions:
~~~
conda install pytorch=1.11.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
