# Project details

## Project title: BCPNNSim

## Project summary (abstract):

BCPNNSim2.0 is built on Bayesian Confidence Propagation Neural Networks (BCPNN), 
a model rooted in decades of research into computational brain science. 
Mainstream ML approaches rely on backpropagation and synchronized global computation. 
BCPNN, instead, employs localised memory access and parallel, 
asynchronous computation at the synapse level. This approach is inherently scalable 
and energy-efficient. This design enables unsupervised learning, 
continuous adaptation to streaming data, and robustness to untrained noise. 
Such features are critical for real-world applications. 

The latest version of the software introduces structural plasticity, 
enabling networks to rewire dynamically during learning, akin to the human brain. 
Moreover, It also supports spiking neural units, enhancing its capacity for 
temporal data processing and sequence learning. These capabilities position BCPNNSim2.0 
as a versatile tool for applications from associative memory modeling to advanced ML.


## Explain the scientific case of the project for which you intend to use the code(s):

NEED INPUT

## Keywords:

Computational brain science, Brain-like computing, Neural networks, Machine learning


## Proposal for civilian purposes
yes
## Is any part of the project confidential?
No


# Research fields

## research field title
PE6 Computer Science and Informatics

## Research field sub-title
NEED INPUT

## Research field share (%)

100%


## AI set of technologies selection

machine learning

## Please specify how does your project ensure ethical principles and addresses potential societal impacts associated with the development and deployment of AI technologies: Not sure
NEED INPUT

# Submission details

## Project duration
6 months or 12 months?

## Preferable starting date
NEED INPUT

## Industry involvement
No

## Public sector involvement
No

# Principal Investigator
## Personal information
NEED INPUT

# Contact Person and Team Members Information
NEED INPUT


# Partitions

## Partition name
LUMI-G

## Code(s) used
BCPNNSIM2.0

## Requested amount of resources (node hours)
4500

## Average number of processes/threads
1

## Average job memory (total usage over all nodes in GB)
40 HBM2 GB

## Maximum amount of memory per process/thread (MB)
40000

## Total amount of data to transfer to/from (GB)
10

## Justification of data transfer
Input/output data and data from GPU profiler

## Is I/O expected to be a bottleneck
No

## I/O libraries, MPI I/O, netcdf, HDF5 or other approaches
output data is written in binary format

## Frequency and size of data output and input
input/output data is quite small, on the order of GB and read from/written to disk only once for input/output 


## Number of files and size of each file in a typical production run
The number of output files is less than 10 and in total the size is on the order of GB

## Total storage required (GB)
50


# Code Details and Development

## Development of the code(s) description

The current code is implemented in C++, capable of unsupervised representation learning  
and it builds on the Bayesian Confidence Propagation Neural Network (BCPNN), 
which has earlier been implemented as abstract as well as 
biophysically detailed recurrent attractor neural networks. 
We developed a feedforward BCPNN model to perform representation learning 
by incorporating a range of brain-like attributes derived from 
neocortical circuits such as cortical columns, divisive normalization, 
Hebbian synaptic plasticity, structural plasticity, sparse activity, 
and sparse patchy connectivity.


The previous verison (https://github.com/nbrav/BCPNNSim-ReprLearn) 
is implemented with MPI for message passing and CUDA/HIP for GPU parallelization.
The current BCPNNSim code version 2.0 is ported to run on a single GPU without multi-GPU support. 
There is also lots of room for improving overall code structure and optimizing the code.


# Code details

## Name and version of the code
BCPNNSim2.0

## Webpage and other references

https://github.com/anderslan/BCPNNSim2.0P/tree/master

https://github.com/nbrav/BCPNNSim-ReprLearn


## Licensing model
MIT license

## Contact information of the code developers

Anders Lansner
ala@kth.se

# Your connection to the code (e.g. developer, collaborator to main developers, etc.)
main developer


# Scalability and performance

## Describe the scalability of the application and performance of the application
The code now only runs on a single GPU, and the performance on an AMD GPU is considerably worse compared to an NVIDIA GPU,
and we would like to optimize it

```
 resolution     CUDA runtime (s) HIP runtime (s)
  1000/1000      20.8            113.3
  6000/1000      94.4            550.3
  60000/10000    934.2           5492.4
```


## What is the target for scalability and performance

```
 resolution     runtime (s)
  1000/1000      20.8
  6000/1000      94.4
  60000/10000    934.2
```

We would like to have a version support multi-GPU as well with reasonable parallel efficiency


# Optimization of the work proposed

## Explain how the optimization work proposed will contribute to future Tier-0 projects
These enhancements will enable simulations of large-scale spiking brain models, potentially achieving real-time speeds on advanced HPC platforms

## Describe the impact of the optimization work proposed - is the code widely used; can it be used for other research projects and in other research fields with minor modifications; would these modifications be easy to add to the main release of the software?

The code BCPNNSim2.0 is used in an EU-funded project EXTRA-BRAIN and SeRC (Swedish e-Science Research Centre) funded Brain-IT projects.
All the improvments will be merged back to the main release and benifit the users from the other projects

## Describe the main algorithms and how they have been implemented and parallelized
NEED INPUT

# Performance

## Main performance bottlenecks
no multi-GPU support
bad performance on AMD GPU

## Describe possible solutions you have considered to improve the performance of the project

using explicit data copying from/to GPU on an AMD GPU

## Describe the application enabling/optimization work that needs to be performed to achieve the target performance
parallelize the code using MPI for multiple GPUs


## Which computational performance limitations do you wish to solve with this project
we would like to have a multi-GPU  version of the code
we would like to reach a reasonable performance on an AMD GPU


# Application Support Team (AST)

## Does your proposal require assistance from an AST on the selected partition(s)
yes


## Is your proposal a follow up of another submitted Epicure project to use EuroHPC quota for providing application support? 
No

