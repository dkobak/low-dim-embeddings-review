# Low-dimensional embeddings of high-dimensional data

## Genome pre-processing

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
