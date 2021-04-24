#CIRCNV

CIRCNV is a method can detect CNVs from NGS data, it transfers the read depth profile from a line shape to a circular shape 
via the polar coordinate transformation, and it test CNVs two times.

### Installation
* numpy 
* numba
* sklearn
* pandas
* pysam
* rpy2

command: pip install numpy numba sklearn pandas pysam rpy2


### R packages
* DNAcopy

command: 
>install.packages("BiocManager")
>BiocManager::install("DNAcopy")


### Input files
A bam file and a reference file
Reference file of simulation data is chr21 of hg37


### runnig command
python CIRCNV.py [bamfile] [reference] [output]


Finally, three files we will get, one is the mutation information of detection, and RD profiles berfore and after recovered
by estimated tumor purity.

