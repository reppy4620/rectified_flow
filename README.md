Rectified Flow
===

The Implementation of Rectified Flow model for study.

[https://arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)

・Following datasets are supported

- MNIST
- FASHION MNIST
- cifar10
- AFHQ cat

・Two types of losses are supported 

- Normal

$$
L(\theta) = ||X_1 - X_0 - v_\theta(X_t, t)||_2
$$

- ot
    - loss equation is same as Normal but distribution of $X_1$ and $X_0$ are different
    - $X_1$ and $X_0$ are sampled from joint distribution induced by optimal transport matrix.

# Dependencies
```
pip install torch torchvision torchaudio POT tqdm scipy einops datasets omegaconf
```

# Usage
I prepared various configs for aforementioned datasets, but there is only script for training MNIST with Normal loss.
You can create the script for other dataset or loss.


```
cd scripts
bash normal_mnist.sh
```
