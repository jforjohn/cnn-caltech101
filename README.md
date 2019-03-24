# cnn-caltech101
This is a project about exploring Convolutional Neural Networks in an object classification use case using Caltech101 data set.

## Files description:

### cnn-caltech
* It reads the respective data from the folder *data_spit*.
* Takes 2 command line args:
  * the model index to load from the *model_pool* file
  * the folder name where it's going to save the results
* If the name of the folder contains the word *augment*, data augmentation is gonna applied

### model_pool
Naive way to specify different models and be able to switch from to the other easily by just specifying the index of each model. It returns the neural netword which was finally built.

### sb_caltech
Script for defining properties of the queue which each job will be submitted to.

### run
Wrapper script for running different models sequentially and creating the folders where the results will be stored. It takes 2 arguments, the starting and the last index of the models of the *model_pool* to be run sequentially through a for-loop.