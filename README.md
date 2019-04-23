# Ego-CNN
This is the repo for Distributed, Egocentric Representations of Graphs for Detecting Critical Structures (ICML 2019).

We study the problem of detecting critical structures using a graph embedding model. Existing graph embedding models lack the ability to precisely detect critical structures that are speciﬁc to a task at the global scale. In this paper, we propose a novel graph embedding model, called the Ego-CNNs, that detects precise critical structures efficiently. Our Ego-CNN can be jointly trained with a task model and help explain/discover knowledge for the task. We conduct extensive experiments and the results show that Ego-CNNs (1) can lead to comparable task performance as the state-of-the-art graph embedding models, (2) works nicely with CNN visualization techniques to illustrate the detected structures, and (3) is efficient and can incorporate with Scale-Free priors, which commonly occurs in social network datasets ,to further improve the training efﬁciency.

## Dependence
 * Python 3.6
 * Tensorflow 1.0
 * NetworkX 2.0
 * Numpy 1.13, Matplotlib 2.1
 * optparse

## To Reproduce Our Result On ICML'19

### Step 1. Download and Preprocess Graph Classification Datasets
Execute Command `python download_dataset.py` to download all the bioinformatic and social network datasets used in the paper.

## Step 2. Train Ego-CNN on specified datasets for specified tasks
To reproduce ...
 * Graph Classification Experiments: run `./execute-graph-classification-on-benchmarks.sh`
 * Effectiveness of Scale-Free Regularizer: run `./execute-graph-classification-on-benchmarks.sh`
 * Visualization on synthetic compounds: run `./execute-graph-classification-on-benchmarks.sh`

## Details
Each experiment contains (1) preprocess step, (2) training step and optionally (3) visualization step.
 1. Each graph dataset needs to be preporcessed by `python preprocess-dataset.py -n [dataset-name] -k [size of neighborhood] [-s relabel graph or not]`, where we already filled the required parameters in the [execute-graph-classification-on-benchmarks.sh](execute-graph-classification-on-benchmarks.sh).
 However, please note that there are still many options in processing the dataset such as
     * How to assign global ordering of nodes?
     * How K-Nearest Neighbors are chosen for each node?
 ... etc. For the details, please see the comments in [preproces-dataset.py](preproces-dataset.py).
 As our main focus is at detecting critical structures, We just choosed an okay setting to get comparable performance.
 2. Execute `python run-ego-cnn.py -n [dataset-name] -g [gpu-id] -f [gpu-fraction] -k [#neighbor of Ego-Convolution] -m [model-type]` with the below settings for each task:
  * Graph Classification Experiments are all with `k=17` (self + 16 nearest neighbors) and `model-type='6L'` (means 1 Patchy-San + 5 Ego-Convolution Layers)
  * Scale-Free Experiments use `model-type='6L_SF` (with scale-free regularizer) and `model-type='2L'` (base model)
  * Visualization Experiments of synthetic compound datasets: `k=4` and `model-type='6L'`
 3. The trained Ego-CNN can be visualized by many existing CNN techniques. Below are some examples:
   * Attention Layer + Transposed Convolution: run `python plot-critical-structure-Attention.py -n [dataset-name] -g [gpu-id] -f [gpu-fraction] -k [#neighbor of Ego-Convolution] -t [threshold to select important neighborhoods]`. This is the setting to plot the figures in our paper.
   * Grad-CAM[ICCV'17] + Transposed Convolution: run `python plot-critical-structure-GradCAM.py -n [dataset-name] -g [gpu-id] -f [gpu-fraction] -k [#neighbor of Ego-Convolution] -t [threshold to select important neighborhoods]`.
