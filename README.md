# Low-dimensional embeddings of high-dimensional data

This is a companion repository to our review paper [de Bodt & Diaz-Papkovich et al. 2025, Low-dimensional embeddings of high-dimensional data](https://arxiv.org/abs/2508.15929). All code is in Python. 

## Content of the repository

This repository is structured as follows:

- `./code`: contains all code files.
  - `./code/Plotting Wikipedia dim reduction.ipynb`: produces figures with 2-D embeddings of Simple English Wikipedia data. The embeddings must have been computed using `./code/wikipedia-compute-embeddings.py` beforehand. 
  - `./code/Wikipedia - create cluster names.ipynb`: defines cluster labels for Simple English Wikipedia data. 
  - `./code/citations.ipynb`: produces a figure showing the number of citations for popular manifold learning tools per year. 
  - `./code/genome-preprocessing-1.sh`: first step in pre-processing 1000 Genomes Project data. See below for more information. 
  - `./code/genome-preprocessing-2.py`: second step in pre-processing 1000 Genomes Project data. See below for more information. 
  - `./code/genomes-compute-embeddings.py`: computes 2-D embeddings of preprocessed 1000 Genomes Project data, and evaluates their quality.
  - `./code/genomes-figure.py`: produces a figure with 2-D embeddings of preprocessed 1000 Genomes Project data. The embeddings must have been computed using `./code/genomes-compute-embeddings.py` beforehand. 
  - `./code/kanton-compute-embeddings.py`: computes 2-D embeddings of preprocessed Kanton et al. data, and evaluates their quality.
  - `./code/kanton-figure.py`: produces a figure with 2-D embeddings of preprocessed Kanton et al. data. The embeddings must have been computed using `./code/kanton-compute-embeddings.py` beforehand. 
  - `./code/kanton-preprocess.py`: preprocesses Kanton et al. data. Raw data are downloaded if needed. 
  - `./code/metrics.ipynb`: produces a figure summarizing quality scores of 2-D embeddings of studied datasets. 
  - `./code/params.py`: defines main parameters employed by many scripts in `./code`. 
  - `./code/paths.py`: defines paths of data files and where to save results. 
  - `./code/tasic-compute-embeddings.py`: computes 2-D embeddings of preprocessed Tasic et al. data, and evaluates their quality.
  - `./code/tasic-figure.py`: produces a figure with 2-D embeddings of preprocessed Tasic et al. data. The embeddings must have been computed using `./code/tasic-compute-embeddings.py` beforehand. 
  - `./code/tasic-preprocess.py`: preprocesses Tasic et al. data. Raw data must have been downloaded beforehand. 
  - `./code/turtles.ipynb`: applies PCA on the turtles dataset from [(Jolicoeur and Mosimann, Growth 1960)](https://www.researchgate.net/profile/Alessandro-Giuliani-2/post/How-to-create-an-index-using-principal-component-analysis-PCA/attachment/61c86da9d248c650edbba126/AS%3A1105257056743426%401640525225582/download/sizeshapeold.pdf). The first two principal components are displayed. 
  - `./code/wikipedia-compute-embeddings.py`: computes 2-D embeddings of [Simple English Wikipedia data](https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings), and evaluates their quality. Data are downloaded if needed. 
  - `./code/utils`: contains a style file and functions used by scripts in `./code`. 
  - `./code/conda_virtual_env/low-dim_embs.yml`: specifies package versions that we employed. Run
  ```
  conda env create -f low-dim_embs.yml
  ```
  to create a `low-dim_embs` conda virtual environment with these package versions. Then, run
  ```
  pip install --user phate==1.0.11
  ```
  to install the version of the `phate` package that we used. 
- `./data`: contains all data files.
  - `./data/citations.csv`: data used by `./code/citations.ipynb`.
  - `./data/1000-Genomes-Project`: contains preprocessed 1000 Genomes Project data and metadata used by `./code/genomes-compute-embeddings.py` and `./code/genomes-figure.py`.
  - `./data/Kanton-et-al/human-409b2`: contains preprocessed Kanton et al. data and labels used by `./code/kanton-compute-embeddings.py` and `./code/kanton-figure.py`.
  - `./data/Simple-English-Wikipedia`: contains files produced by `./code/Wikipedia - create cluster names.ipynb`.
  - `./data/Tasic-et-al`: contains preprocessed Tasic et al. data and metadata used by `./code/tasic-compute-embeddings.py` and `./code/tasic-figure.py`. 
- `./figures`: contains all figures produced by scripts in `./code`.
- `./results`: contains all result files.
  - `./results/metrics.csv`: used by `./code/metrics.ipynb`.
  - `./results/embeddings`: contains all 2-D embeddings of preprocessed Tasic et al., preprocessed Kanton et al., Simple English Wikipedia and preprocessed 1000 Genomes Project data. The only exceptions consist in the first two principal components of preprocessed Tasic et al. and preprocessed Kanton et al. data, which are not saved as these datasets are preprocessed by PCA. Their respective first two principal components can be retrieved by running:
  ```
  import numpy as np
  X_pca_tasic = np.load('./data/Tasic-et-al/preprocessed-data.npy')[:,:2]
  X_pca_kanton = np.load('./data/Kanton-et-al/human-409b2/preprocessed-data.npy')[:,:2]
  ```
  - `./results/quality_scores`: contains quality scores computed for all 2-D embeddings of preprocessed Tasic et al., preprocessed Kanton et al., Simple English Wikipedia and preprocessed 1000 Genomes Project data. 

## Tasic et al. data pre-processing

This dataset is from [(Tasic et al., Nature 2018)](https://www.nature.com/articles/s41586-018-0654-5). Preprocessing is conducted as detailed in [(Kobak and Berens, Nature communications 2019)](https://www.nature.com/articles/s41467-019-13056-x) and its [companion repository](https://github.com/berenslab/rna-seq-tsne). 

## Kanton et al. data pre-processing

This dataset is from [(Kanton et al., Nature 2019)](https://www.nature.com/articles/s41586-019-1654-9). Preprocessing is conducted as detailed in [(Böhm et al., Journal of Machine Learning Research 2022)](https://www.jmlr.org/papers/v23/21-0055.html) and in [(Damrich et al., bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.04.26.590867v1.abstract). 

## 1000 Genome Projet data pre-processing

We use data from the 1000 Genomes Project. It can be obtained here:

* https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/

The file is `ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped`.

Pre-processing is done in two files. The first uses two versions of [PLINK](https://www.cog-genomics.org/plink/) to filter the input. The second is a python script that fills some missing data and parses data into something usable in numpy/pandas.

The first file is `genome-preprocessing-1.sh`. It first identifies regions of [linkage disequilibrium](https://en.wikipedia.org/wiki/Linkage_disequilibrium) using PLINK2. After that it filters the data for quality and restricts it to common variants (minor allele frequency > 0.05). It also filters some highly-variable regions of the genome such as the [HLA locus](https://en.wikipedia.org/wiki/Human_leukocyte_antigen). The genomic coordates are taken from the default list used in [flashPCA](https://github.com/gabraham/flashpca):
```
5 44000000 51500000 r1
6 25000000 33500000 r2
8 8000000 12000000 r3
11 45000000 57000000 r4
```

The second file is `genome-preprocessing-2.py`. This uses [sgkit](https://github.com/sgkit-dev/sgkit) to parse data.
