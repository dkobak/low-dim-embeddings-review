# This is the first step in pre-processing the 1KGP data
# We filter for LD and high-variability regions and work with common variants
# exclusion regions are from flashPCA (https://github.com/gabraham/flashpca)
# 5 44000000 51500000 r1
# 6 25000000 33500000 r2
# 8 8000000 12000000 r3
# 11 45000000 57000000 r4
# Genotype data are available here:
# https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/

# Note that this uses two versions of PLINK
# (PLINK2 for LD thinning and PLINK1.9 elsewhere)

# First thin for LD and then convert to a bed file
MASK_FILE=exclusion_regions_hg19.txt
VCF_FILE=ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz
BED_FILE=ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped

# Use PLINK2 for LD thinning
module load plink/2.00

plink2 \
--vcf ${VCF_FILE} \
--indep-pairwise 1000 50 0.1 \
--threads 16 \
--make-bed \
--out ${BED_FILE}

module unload plink/2.00
module load plink/1.90

OUT_FILE=ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped
PRUNE_IN=${BED_FILE}.prune.in

# Also make a VCF
plink \
--bfile ${BED_FILE} \
--recode vcf \
--maf 0.05 \
--mind 0.1 \
--geno 0.1 \
--hwe 1e-6 \
--threads 16 \
--extract ${PRUNE_IN} \
--exclude range ${MASK_FILE} \
--out ${BED_FILE}_VCF
