# ecoNetwork-OT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16103567.svg)](https://doi.org/10.5281/zenodo.16103567)

This repository holds the codebase for applying optimal transport to compare ecological networks, as documented in "Quantifying functionally equivalent species and ecological network dissimilarity with optimal transport distances." In particular, the repo contains code for operating over the synthetic networks.  

Refer to the Start-Up Guide for your first time working with this repository and Table of Content if you are looking to reference specific scripts and/or directories.

### Start-Up Instruction

The expected usage of this repository is on a local computing device using a virtual environment organized by Conda. Virtual environment essentially creates a space in your computer dedicated to software related to this repository. We additionally provide a file to specify the package versions for reproduction our results exactly. 

For first-time users, we recommend the lightweight miniconda distribution that can be found [here](https://docs.anaconda.com/free/miniconda/). If the user is on Mac, consider using [Homebrew](https://brew.sh/) to [install Miniconda](https://formulae.brew.sh/cask/miniconda). 

After conda is installed on your device, create a virtual environment from `environment.yml` by running 
```
conda env create -f environment.yml -p ./envs
```
which should create a folder in the top-level directory named `envs` with all the packages you need. 


To activate this virtual environment, you can then run the following command
``` 
conda activate ./envs
```

After that, you can start a Jupyter Notebook to run `simulation_studies.ipynb` with the following commands. 
```
jupyter notebook
```

### Table of Content

Here we summarize all files present in this repo and their purpose.
```
+-- results/
|   +-- permutation/ : stores output from perturbation_analysis.ipynb
|   +-- simulation/ : stores output from simulation_studies.ipynb
+-- conceptual_transport.ipynb : a notebook computing optimal transport plan on pairs of simple food webs, corresponds to Fig 1 and 2. 
+-- simulation_studies.ipynb : a notebook containing all experiments using topological graphs (appendix) and generative models (Section 3)
+-- perturbation_analysis.ipynb : a notebook to perturb webs and see how GW distance changes in response to node or edge removals
+-- got.py : class file for GraphOT and GraphOT_Factory, which are abstractions of measure (probabilistic) networks
+-- logger.py : class file for a logger object to document experiments 
+-- environment.yml
+-- .gitignore
+-- README.md 
```

### Advanced Notes 
There are a few subtle facts about this repository that will help the users navigate. 

1. The major set of results you can reproduce lie in `simulation_studies.ipynb`, which is divided into 3 main sections: (1) Graph Topology, which corresponds to Appendix A.7 and the left panel in Figure A.5, (2) Block Graphs, which correspond to the right panel in Figure A.5 and Figure A.6, (3) Cascade Models and (4) Niche Models, both of which compose Section 3 in the main text. `perturbation_analysis.ipynb` corresponds to Section 4.7 in the main text, but we use synthetic (erdos-renyi) graphs instead of the empirical webs. 

2. The expected workflow for computing OT distance is by first generating a [dictionary](https://www.geeksforgeeks.org/python-dictionary/) of (name:graph) pairings, where name is of datatype `string` and graph is of datatype `nx.Graph` or `nx.DiGraph`. With this dictionary, the user can create a `got.GraphOT_Factory` object and then run `got.compute_pairwise_dist(...)` to easily recover a matrix of optimal transport distances, and the corresponding transport plans. 

3. To better understand the `got` package, we recommend checking out the `conceptual_transport.ipynb` which showcases `got` package abstracts the definitions of the probability (e.g. uniform) and in-network distance (e.g. shortest path) for two graphs. The computation of GW distance here also showcases the integration of the POT library and our custom `got.py` file. The computation of OT distances for a set of network (described in 1) expands upon the procedures in this script, resulting in the `GraphOT_Factory.compute_pairwise_dist(...)` method. This `conceptual_transport.ipynb` is also where our Fig 1 and 2 transport plans are made. 

4. Our code `simulation_studies.ipynb` includes a reproduction of Figure A.6 with significantly fewer samples, to avoid the user from wondering why the script takes 30+ minutes to run on initial encounter. To reproduce the actual experiments, follow the notebook instruction to set the experiment parameter `SAMPLES` to 50 instead of 5. The exact run results corresponding to our manuscript are documented in `results/simulation/mds_vis/block-Path-Cycle-Star-Tree_2024-09-30_12:46:37`