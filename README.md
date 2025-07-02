# Low-dimensional embeddings of high-dimensional data

This is a companion repository to our paper (de Bodt & Diaz-Papkovich et al. 2025, Low-dimensional embeddings of high-dimensional data). All code is in Python. 

## Content of the repository

This repository is structured as follows:

- `./code`: contains all code files.

-- `./wikipedia-compute-embeddings.py`: 

-- `./conda_virtual_env/low-dim_embs.yml`: indicates employed package versions. Run

```
conda env create -f low-dim_embs.yml
```

to create a `low-dim_embs` conda virtual environment with these package versions. Then, run

```
pip install --user phate==1.0.11
```

to install the version of the `phate` package that we used. 

-- `./utils`: 

--- `./utils/SQuaD_MDS.py`: 

--- `./utils/dr_quality.py`: 

--- `./utils/mpl_style.txt`: 

--- `./utils/plot_fcts.py`: 

--- `./utils/preprocess.py`: 

--- `./utils/run_embs.py`: 

- `./data`: contains all data files.

- `./figures`: contains all figures.

- `./results`: contains all result files.

## Tasic et al. data pre-processing

Todo

## Kanton et al. data pre-processing

Todo

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
