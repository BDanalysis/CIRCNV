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
* matplotlib
command: pip install numpy numba sklearn pandas pysam rpy2 matplotlib


### R packages
* DNAcopy

command: 
>install.packages("BiocManager")
>BiocManager::install("DNAcopy")


### Input files
A bam file and a reference file. Here, the reference sequence could be selected from the commonly used version, human genome 19. In addition, the bam file ,including header and read alignment section of the tumor sample. for more details, you can view test.bam by samtools view test.bam.


### runnig command
python CIRCNV.py [bamfile] [reference] [binsize] [output] [segCount] [k]

-bamfile: a bam file
-reference: the reference folder path
-binsize: the window size('1000' by default)
-segCount: the number of partitions to CBS('50' by default)
-k: the parameters of this algorithm('10' by default)

Example running:
python CIRCNV.py test.bam Users/reference 1000 Users/CIRCNV_result 50 10 


###Output files
After running, there will be three files, including mutation information of detection(test.bam.score.txt), the detail of CNV before recorrecting RD values(test.bam.result.txt) and after recorrecting RD values(test.bam.new_result.txt). However, if your bam file doesn't contain CNVtype of loss, there will be two output files, because the process of correcting RD values is for loss.


###Storage
It is depended of your input files, but theoretically, our algorithm need storage less than 4G.

Enjoy it!!!

