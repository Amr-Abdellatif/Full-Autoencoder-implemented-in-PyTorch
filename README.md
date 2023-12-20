# Full-Autoencoder-implemented-in-PyTorch

This Repo is about implementing a Full scale auto encoder in PyTorch

## Walkthrough

1. The code start by setting up device agnostic code , downloading MNIST Dataset and applying the transformations and pushing it to the Network, the training data is changed from x,y pair to x,x pair because we're trying to make the network learn through figuring out the features like selfsupervised learning so no labels are given.

2. The Network takes the 28*28 image size MNIST and compress the features in bottleneck to 2 feature representation and push it to back to the 28*28 feature representation again

3. This 2 feature representation will be used to visualize the features later in the code as more than 2 features will be so difficult to represent

4. Finally we show what the network was really be able to reconstruct the features back to the original image and compare them with the function `show encode decode`
