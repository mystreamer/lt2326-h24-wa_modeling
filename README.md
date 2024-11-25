# Assignment 2 - Wikiart peregrinations

## Part 0 - Documentation

## Part 1 - Fix class imbalance

For the class imbalance problem I opt for the method of oversampling, i.e. duplicating the images of the underrepresented classes and *additionally* I manipulate them in a specific way.

## Part 2 - Autoencoder and cluster representations

### Training the autoencoder

#### Architecture
We use a simple architecture consisting of identical layers for the Encoder as well as the Decoder. The layers of the decoder are in reverse order.

#### Measuring progress
To measure the progress we need to use a function that estimates how well the decoder reconstructs the original image from the latent space. A loss function that can capture this called **reconstruction loss function**. Here we use the Mean Squared Error as reconstruction loss.

## Bonus B