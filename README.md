# Assignment 2 - Wikiart peregrinations

## Part 0 - Documentation

In each section of this document I describe my observations along with the commands that are used to generate the models / images.

#### Configuration / CLI

The default configuration parameters can be found in the `config.json`. Each of the setting parameters can be adjusted by overriding them and passing them as command-line arguments to the script that one wants to run. By default the system will look for a file of the name `config.json` in the root directory. If this behaviour should be changed, all of the commands that follow can be immediately followed with the path to a different configurations file, e.g. `python train_autoencoder.py config_custom.json [--options...]`

## Part 1 - Fix class imbalance

For the class imbalance problem I opt for the method of oversampling, i.e. duplicating the images of the underrepresented classes. For the implementation I use generate weightings that "balance" out the imbalance and then use the [WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler). The weight calculation code can be found in the `WikiArtDataset` class under `modules.wikiart`.

## Part 2 - Autoencoder and cluster representations

### Training the autoencoder

#### Architecture
We use a simple architecture consisting of identical layers for the Encoder as well as the Decoder. The layers of the decoder are in reverse order.

#### Measuring progress
To measure the progress we need to use a function that estimates how well the decoder reconstructs the original image from the latent space. A loss function that can capture this called **reconstruction loss function**. Here we use the Mean Squared Error as reconstruction loss.

Running the command:
```
python train_autoencoder.py [--training_autoencoder.learning_rate] [--training_autoencoder.epochs]
```

Will train the autoencoder model and then save it in the `models` folder under the name `autoencoder.pth` if not specified otherwise.

### Generating the cluster representations

To generate the clusters I use Principal Component Analysis (PCA) for dimensionality reduction and K-menas to generate the clusters. In order for the K-means algorithm to perform one must choose an adequate K (number of clusters to be generated). Since we already know how many classes "should" be clustered I set k to the number of different art-style categories in our dataset.

Running the command:
```
python clustering.py [--clustering.k]
```
with optional k will yield generate an image in the `visualisations` folder (if the `visualisation_dir` option is not overriden). The filename is `clusters.png`.

#### Clustering results

My datapoints do not appear to cluster nicely, the individual points (projected down to 2d) of the classes seem to rather all belong to the same distribution. I do not know whether this is due to the short training time or whether there are other reasons.



## Part 3 - Generation / style transfer

For this task I used the AutoEncoder I constructed in Part 2, but also added two more components (expanded the model). Firstly, I added an StyleEmbedding module so that we can generate generalised representations of style in the latent AutoEncoder space, rather than image-specific representation. Second, we use a Discriminator module that comprises of a simple feedforward layer followed by Softmax activations. This discriminator module is tasked with "training" the style embeddings. The basic idea is to use the discriminator module make the style embeddings become generalised representations of the image-classes they are denoting.
The idea was primarily inspired from this repo-code / paper [found here](https://github.com/heejin928/How-Positive-Are-You-Text-Style-Transfer-using-Adaptive-Style-Embedding). The paper is not concerned with vision, but deals with textual style transfer.

Running the command:
```
python train_discriminator.py [--training_discriminator.learning_rate] [--training_discriminator.epochs]
```

Will train the autoencoder model and then save it in the `models` folder under the name `discriminator.pth` if not specified otherwise through the `discriminator_model_name`.

#### Results

Unfortunately, the resultant images appear to be more noise than anything else. The below example was generated when I tried to transfer the style of an image from *Analytical Cubism* to *Abstract Expressionism*. The source image is from the training data itself. Using data from the test set, the images always appear to come out fully black.

![GeneratedImg](data:image/jpeg;base64,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
)

To obtain your own results use:
```
python transfer_style.py [--optionals] target_image original_style target_style
```

For example:
```
python transfer_style.py ./images/test_input.jpg "Abstract_Expressionism" "Analytical_Cubism"
```

The resulting image can then be found in `./images/transferred_image.png` (if none of the paths deviate from the default).

## Bonus A+B

Not completed so far.