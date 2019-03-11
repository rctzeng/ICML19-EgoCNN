# Ego-CNN

## Dependence
 * Tensorflow 1.0
 * NetworkX 2.0
 * Numpy 1.13, Matplotlib 2.1
 * optparse

## Step 1. Download and Preprocess Graph Classification Datasets
 (1) Download the dataset: `python download_dataset.py`
 (2) Preprocess the dataset for training: `python preprocess-dataset.py -n [dataset-name] -k [size of neighborhood] [-s relabel graph or not]`, in which it includes the settings like
     * How the Nodes are ordered in the graph?
     * How the K-Nearest Neighbors are chosen for each node in each graph?
     * Note: do not set `-s` for visualization exps to maintain the vertex correspondence with original graphs

## Step 2. Train Ego-CNN on specified datasets
 * run experiment on specified dataset: `python run-ego-cnn.py -n [dataset-name] -g [gpu-id] -f [gpu-fraction] -k [#neighbor of Ego-Convolution] -m [model-type]`
 * to reproduce our [paper(TBD)](),
     * settings for graph classification exp: `k=17` and `model-type='6L'`
     * settings for scale-free exp: `model-type='6L_SF` (with scale-free regularizer) and `model-type='2L'` (base model)
     * settings for visualization exp of synthetic compound datasets: `k=4` and `model-type='6L'`

## Step 3. Visualization of trained Ego-CNN
Option 1: visualize by Attention Layer + Transposed Convolution
 (1) Re-train the visualization model(which adds an Attention layer): `python run-visualization-model.py -n [dataset-name] -g [gpu-id] -f [gpu-fraction] -k [#neighbor of Ego-Convolution]`
 (2) Plot critical structures: `python plot-critical-structure-Attention.py -n [dataset-name] -g [gpu-id] -f [gpu-fraction] -k [#neighbor of Ego-Convolution] -t [threshold to select important neighborhoods]`

Option 2: visualize by Grad-CAM[ICCV'17] + Transposed Convolution
 * `python plot-critical-structure-GradCAM.py -n [dataset-name] -g [gpu-id] -f [gpu-fraction] -k [#neighbor of Ego-Convolution] -t [threshold to select important neighborhoods]`