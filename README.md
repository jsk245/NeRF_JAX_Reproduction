# NeRF_JAX_Reproduction
## Description
This code reproduces some of the results from the paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (https://arxiv.org/abs/2003.08934) 
using the JAX machine learning framework and Haiku library. The paper approaches the problem of novel view synthesis (creating images from new angles based on old ones)
by training MLP models that are specific to the scene in question. I partially trained two models, and results from these two models are included below. Since the training
requires a GPU, these models were trained on Google Colab, so I used significantly lower resolution images (downsampled from 1008x756 to 126x94) for faster convergence. 
Although I was only able to complete around 50k-65k steps out of the 150k I was planning to do during training due to usage limits, the model was able to create some
non-trivial results, but proper training would require more GPU hours and more RAM to handle the higher resolution data.

## Data
In this repo, I have included images corresponding to 2 scenes along with the parameters of the model trained on these two scenes. These images are downsampled versions of
the ones that can be downloaded from here: https://www.matthewtancik.com/nerf. I only trained this model on llff data, meaning that the data was all forward-facing,
and this data also came with camera information, which is required for this model to work. If camera information is not available, software such as COLMAP is required to
obtain the camera poses.

## Model
The NeRF approach to the task of novel view synthesis involves training MLP models that are specific to each scene. The inputs to the model are the xyz coordinates of a 
point in space and a 3d vector used to designate the angle that this point is being viewed from. After passing this data through 8 linear layers (each with ReLU
activations and 256 features), this model then returns a color (in rgb) and a volume density (you can think of this as essentially assigning a value to if an object
exists at this point and if it can be seen from the inputted direction).

In order to use this model to create an image, we first choose a camera origin, and we use different direction vectors to create rays originating from this camera that
point in the direction of the scene (each of these rays correspond to one pixel of the final image). Next, we use what the authors refer to as "coarse sampling", 
which means that we sample 64 approximately equally spaced points along the ray (we have bounds for the scene, so these are spaced between these bounds), 
and we input all of these points into the MLP model. After this, we use the volume densities, which can be interpreted as the points that can be seen from the camera,
to create a probability distribution along the ray that we sample from, which is what the authors refer to as "fine sampling". We sample an additional 128 points along the ray
based on these weights, which are mainly in areas that we expect to contribute to the end pixel, and put these into the MLP model as well. 
Finally, we use volume rendering** to combine all of the rgb values and volume densities corresponding to these points to create the final rgb value for the pixel.

** The paper includes the formula for volume rendering on pages 5-6. This equation involves how we transform the volume density to a nice number that affects the final pixel
color, and this is the actual value we use to create the probability distribution for fine sampling.

## Results
Here are images created from the two trained models from views not included in the dataset:

<img src="https://user-images.githubusercontent.com/93054906/212563317-75787dea-9a1c-4ae4-ae55-6d8fe3bf9b7a.png" width="256">
<img src="https://user-images.githubusercontent.com/93054906/212563444-2f8caec3-23b5-4a9e-b3f7-35828665f65f.png" width="256">

And here are videos created that only include camera poses not found in the dataset. The room was trained for 15k less steps than the flowers, and both were trained
for only about 1/3 of the proper number of steps, so when combined with the lower resolution images and smaller batch size used, that is likely why the lighting
at the edges is weird:

![video](https://user-images.githubusercontent.com/93054906/212563624-bdddd3df-2004-43e2-b825-1ac47f5a240c.gif)
![video(1)](https://user-images.githubusercontent.com/93054906/212563631-030d7055-e45b-4e90-8531-a221100f0742.gif)


## Implementation Details
I used the same number of layers and featuers per layer as the paper, with the same activations. However, I had to lower the resolution of the images as described above,
and I also had to use a smaller batch size (1024 vs 4096) due to limited RAM. Additionally, the learning rate used in the paper (5e-4) was too high for my implementation,
so I use 2e-4 instead. Finally, I used the Adam optimizer like the paper.

One line in my code that may be confusing is when I calculate the $T_i = exp(-\Sigma_{j=1}^{i-1}\sigma_j\delta_j)$, which is involved in how we calculate the final pixel color
during volume rendering.
The paper describes what each part of this is in more detail, so read that to actually understand what this is, but if you look at my code, I use jnp.exp and then jnp.cumprod
instead of jnp.cumsum and then jnp.exp. Since the volume density is calculated only using a ReLU activation layer (which is the $\sigma_j$ in this equation), it is unbounded
in the positive direction, so I used jnp.cumprod to avoid overflows.
