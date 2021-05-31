import numpy as np
import sys
import pysam
import os
from scipy.stats import norm
import gc
import pandas as pd
import scipy
from numba import njit
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from sklearn import preprocessing
import datetime
from sklearn.cluster import KMeans

from sklearn.metrics import euclidean_distances


def read_bam_file(filename):
    samfile = pysam.AlignmentFile(filename, "rb")  
    chrList = samfile.references  
    return chrList


def read_ref_file(filename, chr_num, ref):     
    # read reference file
    if os.path.exists(filename):
        print("Read reference file: " + str(filename))
        with open(filename, 'r') as f:
            line = f.readline()  
            for line in f:
                linestr = line.strip()
                ref[chr_num] += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
        
    return ref     
   
   
def Binning(ref, binSize, chrLen, filename):
    chrTag = np.full(23, 0)
    chrList = np.arange(23)
    maxNum = int(chrLen.max() / binSize) + 1
    InitRD = np.full((23, maxNum), 0.0)
    # read bam file and get bin rd
    print("Read bam file: " + str(filename))
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        idx = int(line.pos / binSize)
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                InitRD[int(chr)][idx] += 1
                chrTag[int(chr)] = 1
    chrList = chrList[chrTag > 0]
    chrNum = len(chrList)
    RDList = [[] for i in range(chrNum)]
    PosList = [[] for i in range(chrNum)]
    InitGC = np.full((chrNum, maxNum), 0)    
    pos = np.full((chrNum, maxNum), 0)
    
    # initialize bin_data and bin_head
    count = 0
    for i in range(len(chrList)):
        chr = chrList[i]
        binNum = int(chrLen[chr] / binSize) + 1
        for j in range(binNum):
            pos[i][j] = j
            cur_ref = ref[chr][j * binSize:(j + 1) * binSize]  
            N_count = cur_ref.count('N') + cur_ref.count('n')
            if N_count == 0:
                gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
            else:
                gc_count = 0
                InitRD[chr][j] = -1000000
                count = count + 1
            InitGC[i][j] = int(round(gc_count / binSize, 3) * 1000)   
     
        # delete
        cur_RD = InitRD[chr][:binNum]
        cur_GC = InitGC[i][:binNum]
        cur_pos = pos[i][:binNum]
        cur_RD = cur_RD / 1000
        index = cur_RD >= 0
        RD = cur_RD[index]
        GC = cur_GC[index]
        cur_pos = cur_pos[index]
        RD[RD == 0] = modeRD(RD)
        RD = gc_correct(RD, GC)
        PosList[i].append(cur_pos)
        RDList[i].append(RD)    
        
    del InitRD, InitGC, pos
    gc.collect()    
    return RDList, PosList, chrList


def modeRD(RD):
    RD = np.array(RD)
    index = RD > 0
    RD = RD[index]
    newRD = np.full(len(RD), 0)
    for i in range(len(RD)):
        newRD[i] = int(round(RD[i], 3) * 1000)
    count = np.bincount(newRD)
    mode = newRD[np.argmax(count)] / 1000
    return mode


def gc_correct(RD, GC):
    # correcting gc bias
    bincount = np.bincount(GC)
    global_rd_ave = np.mean(RD)
    for i in range(len(RD)):
        if bincount[GC[i]] < 2:
            continue
        mean = np.mean(RD[GC == GC[i]])
        RD[i] = global_rd_ave * RD[i] / mean
    return RD

     
def dis_matrix(RD_count):
    pos = np.array(range(1, len(RD_count)+1))
    nr_min = np.min(RD_count)
    nr_max = np.max(RD_count)
   
    newpos = (pos - min(pos) / max(pos) - min(pos)) * 2 * np.pi
    RD_count = (RD_count - min(RD_count))/(max(RD_count) - min(RD_count))
    RD_count = RD_count.astype(np.float)
    newpos = newpos.astype(np.float)
    rd = np.c_[newpos, RD_count] 
    temp = rd.copy()
    rd[:,0] = temp[:,1]*np.cos(temp[:,0])
    rd[:,1] = temp[:,1]*np.sin(temp[:,0])
    plot_polar(newpos,RD_count)
    dis = euclidean_distances(rd, rd)

    return dis, newpos      
    
@njit
def k_matrix(dis, k):
    min_matrix = np.zeros((dis.shape[0], k))
    for i in range(dis.shape[0]):
        sort = np.argsort(dis[i])
        min_row = dis[i][sort[k + 1]]
        for j in range(1, k + 1):
            min_matrix[i, j] = sort[j]
        dis[i][sort[1:(k + 1)]] = min_row
    return dis, min_matrix


@njit
def reach_density(dis, min_matrix, k):
    density = []
    for i in range(min_matrix.shape[0]):
        cur_sum = np.sum(dis[min_matrix[i], i])
        if cur_sum == 0.0:
            cur_density = 100
        else:
            cur_density = 1 / (cur_sum / k)
        density.append(cur_density)
    return density



def get_scores(density, min_matrix, binHead, k):
    scores = np.full(int(len(binHead)), 0.0)
    for i in range(min_matrix.shape[0]):
        cur_rito = density[min_matrix[i]] / density[i]
        cur_sum = np.sum(cur_rito) / k
        scores[i] = cur_sum
    return scores



def plot_polar(pos,data):
    r = data
    theta = pos
    ax = plt.subplot(111,projection='polar')
    c = ax.scatter(theta,r,c='black',cmap='hsv',alpha=0.8)   
    plt.show()



def scaling_RD(RD, mode): 
    posiRD = RD[RD > mode]
    negeRD = RD[RD < mode]
    if len(posiRD) < 50:
        mean_max_RD = np.mean(posiRD)
    else:
        sort = np.argsort(posiRD)
        maxRD = posiRD[sort[-50:]]
        mean_max_RD = np.mean(maxRD)

    if len(negeRD) < 50:
        mean_min_RD = np.mean(negeRD)
    else:
        sort = np.argsort(negeRD)
        minRD = negeRD[sort[:50]]
        mean_min_RD = np.mean(minRD)
    scaling = mean_max_RD / (mode + mode - mean_min_RD)
    for i in range(len(RD)):
        if RD[i] < mode:
            RD[i] /= scaling
    return RD



def Write_data_file(chr, seg_start, seg_end, seg_count, scores):
    """
    write data file
   
    """
    output = open(p_value_file, "w")
    output.write("chr" + '\t' + "start" + '\t' + "end" + '\t' + "read depth" + '\t' + "lof score" + '\n')
    for i in range(len(scores)):
        output.write(
            str(chr[i]) + '\t' + str(seg_start[i]) + '\t' + str(seg_end[i]) +
            '\t' + str(seg_count[i]) + '\t' + str(scores[i]) + '\n')


def calculating_CN(mode, CNVRD, CNVtype):

    CN = np.full(len(CNVtype), 0)
    index = CNVtype == 1
    lossRD = CNVRD[index]
    if len(lossRD) > 2:
        data = np.c_[lossRD, lossRD]
        del_type = KMeans(n_clusters=2, random_state=9).fit_predict(data)
        CNVtype[index] = del_type
        if np.mean(lossRD[del_type == 0]) < np.mean(lossRD[del_type == 1]):
            homoRD = np.mean(lossRD[del_type == 0])
            hemiRD = np.mean(lossRD[del_type == 1])
            for i in range(len(CN)):
                if CNVtype[i] == 0:
                    CN[i] = 0
                elif CNVtype[i] == 1:
                    CN[i] = 1
        else:
            hemiRD = np.mean(lossRD[del_type == 0])
            homoRD = np.mean(lossRD[del_type == 1])
            for i in range(len(CN)):
                if CNVtype[i] == 1:
                    CN[i] = 0
                elif CNVtype[i] == 0:
                    CN[i] = 1
        purity = 2 * (homoRD - hemiRD) / (homoRD - 2 * hemiRD)

        for i in range(len(CNVtype)):
            if CNVtype[i] == 2:
                CN[i] = int(2*CNVRD[i] / (mode * purity) - 2 * (1-purity) / purity)
    return CN


def boxplot(scores):
    four = pd.Series(scores).describe()
    Q1 = four['25%']
    Q3 = four['75%']
    IQR = Q3 - Q1
    upper = Q3 + 0.75 * IQR
    lower = Q1 - 0.75 * IQR
    return upper

def seg_RD(RD, binHead, seg_start, seg_end, seg_count):
    seg_RD = np.full(len(seg_count), 0.0)
    for i in range(len(seg_RD)):
        seg_RD[i] = np.mean(RD[seg_start[i]:seg_end[i]])
        seg_start[i] = binHead[seg_start[i]] * binSize + 1
        if seg_end[i] == len(binHead):
            seg_end[i] = len(binHead) - 1
        seg_end[i] = binHead[seg_end[i]] * binSize + binSize

    return seg_RD, seg_start, seg_end


def Read_seg_file(num_col, num_bin):
    """
    read segment file (Generated by DNAcopy.segment)
    seg file: col, chr, start, end, num_mark, seg_mean
    """
    seg_start = []
    seg_end = []
    seg_count = []
    seg_len = []
    with open("seg", 'r') as f:
        for line in f:
            linestrlist = line.strip().split('\t')
            start = (int(linestrlist[0]) - 1) * num_col + int(linestrlist[2]) - 1
            end = (int(linestrlist[0]) - 1) * num_col + int(linestrlist[3]) - 1
            if start < num_bin:
                if end > num_bin:
                    end = num_bin - 1
                seg_start.append(start)
                seg_end.append(end)
                seg_count.append(float(linestrlist[5]))
                seg_len.append(int(linestrlist[4]))
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    return seg_start, seg_end, seg_count, seg_len



def combiningCNV(seg_chr, seg_start, seg_end, seg_count, scores, upper, mode):

    index = scores > upper
    CNV_chr = seg_chr[index]
    CNVstart = seg_start[index]
    CNVend = seg_end[index]
    CNVRD = seg_count[index]

    type = np.full(len(CNVRD), 1)
    for i in range(len(CNVRD)):
        if CNVRD[i] > mode:
            type[i] = 2

    for i in range(len(CNVRD) - 1):
        if CNVend[i] + 1 == CNVstart[i + 1] and type[i] == type[i + 1]:
            CNVstart[i + 1] = CNVstart[i]
            type[i] = 0

    index = type != 0
    CNVRD = CNVRD[index]
    CNV_chr = CNV_chr[index]
    CNVstart = CNVstart[index]
    CNVend = CNVend[index]
    CNVtype = type[index]

    return CNV_chr, CNVstart, CNVend, CNVRD, CNVtype


def clusting(sign0):
    homo_RD = []
    hemi_RD = []
    with open(outfile, "r") as f:
        for line in f:
            linestrlist = line.strip().split('\t')
            if linestrlist[3] == "loss":
                if linestrlist[4] == str(sign0):
                    homo_RD.append(float(linestrlist[5]))
                else:
                    hemi_RD.append(float(linestrlist[5]))
    
    return homo_RD, hemi_RD


def calculating_ave_rfa(homo_RD, hemi_RD, ave_RD):
    rfa = []
    if(len(homo_RD) > 0):
        for i in range(len(homo_RD)):
            homo_rfa = float(1.0 - (homo_RD[i] / ave_RD))
            rfa.append(homo_rfa)
    if(len(hemi_RD) > 0):
        for i in range(len(hemi_RD)):
            hemi_rfa = float(2.0  * (ave_RD - hemi_RD[i]) / ave_RD)
            rfa.append(hemi_rfa)    
    
    ave_rfa = np.mean(rfa)
    return ave_rfa


def recover_RD(ave_RD, ave_rfa, scores, all_RD):
    new_RD = []
    for i in range(len(scores)):
        rp = (all_RD[i] - (1 - ave_rfa) * ave_RD) / ave_rfa
        new_RD.append(rp)
  
    new_RD = np.array(new_RD)
    return new_RD
        


def Write_CNV_File(chr, CNVstart, CNVend, CNVtype, CN, filename, CNVRD):
    """
    write cnv result file
    """
    output = open(filename, "w")
    for i in range(len(CNVtype)):
        if CNVtype[i] == 2:
            output.write("chr" + str(chr[i]) + '\t' + str(CNVstart[i]) + '\t' + str(
                CNVend[i]) + '\t' + str("gain") + '\t' + str(CN[i]) + '\t' + str(CNVRD[i]) + '\n')
        else:
            output.write("chr" + str(chr[i]) + '\t' + str(CNVstart[i]) + '\t' + str(
                CNVend[i]) + '\t' + str("loss") + '\t' + str(CN[i]) + '\t' + str(CNVRD[i]) +  '\n')
    output.close()


def Write_CNV_newFile(chr, CNVstart, CNVend, CNVtype, CN, filename):
    """
    write new cnv result file
    """
    output = open(filename, "w")
    for i in range(len(CNVtype)):
        if CNVtype[i] == 2:
            output.write("chr" + str(chr[i]) + '\t' + str(CNVstart[i]) + '\t' + str(
                CNVend[i]) + '\t' + str("gain") + '\t' + str(CN[i]) + '\n')
        else:
            output.write("chr" + str(chr[i]) + '\t' + str(CNVstart[i]) + '\t' + str(
                CNVend[i]) + '\t' + str("loss") + '\t' + str(CN[i]) + '\n')
    output.close()


# get params
starttime = datetime.datetime.now()
#bam = sys.argv[1]
bam = "test.bam"
refpath = sys.argv[2]
binSize = sys.argv[3]
outpath = sys.argv[4]
path = os.path.abspath('.')
segpath = path+str("/seg")
p_value_file = outpath + '/' + bam + ".score.txt"
outfile = outpath + '/' + bam + ".result.txt"
newoutfile = outpath + '/' + bam + ".new_result.txt"
col = sys.argv[5]
k = sys.argv[6]


ref = [[] for i in range(23)]
refList = read_bam_file(bam)
for i in range(len(refList)):
    chr = refList[i]
    chr_num = chr.strip('chr')
    if chr_num.isdigit():
        chr_num = int(chr_num)
        if(chr_num == 21):
            reference = "chr21.fa"
            ref = read_ref_file(reference, chr_num, ref)


chrLen = np.full(23, 0)
for i in range(1, 23):
    chrLen[i] = len(ref[i])
  

RDList, PosList, chrList = Binning(ref, binSize, chrLen, bam)
all_chr = []
all_RD = []
all_start = []
all_end = []
modeList = np.full(len(chrList), 0.0)

for i in range(len(chrList)):
    print("analyse " + str(chrList[i]))
    RD = np.array(RDList[i][0])
    pos = np.array(PosList[i][0])
    numbin = len(RD)
    modeList[i] = modeRD(RD)
    scalRD = scaling_RD(RD, modeList[i])
    print("segment count...")
    v = robjects.FloatVector(scalRD)
    m = robjects.r['matrix'](v, ncol=col)
    robjects.r.source("CBS_data.R")
    robjects.r.CBS_data(m, segpath)

    num_col = int(numbin / col) + 1
    seg_start, seg_end, seg_count, seg_len = Read_seg_file(num_col, numbin)
    seg_count = np.array(seg_count)

    seg_count = seg_count[:-1]
    seg_start = seg_start[:-1]
    seg_end = seg_end[:-1]

    seg_count, seg_start, seg_end = seg_RD(RD, pos, seg_start, seg_end, seg_count)
    all_RD.extend(seg_count)
    all_start.extend(seg_start)
    all_end.extend(seg_end)
    all_chr.extend(chrList[i] for j in range(len(seg_count)))


all_chr = np.array(all_chr)
all_start = np.array(all_start)
all_end = np.array(all_end)
all_RD = np.array(all_RD) 
for i in range(len(all_RD)):
    if np.isnan(all_RD[i]).any():
        all_RD[i] = (all_RD[i - 1] + all_RD[i + 1]) / 2

print("test CNV...")
dis, newpos = dis_matrix(all_RD)
dis, min_matrix = k_matrix(dis, k)
min_matrix = min_matrix.astype(np.int)
density = reach_density(dis, min_matrix, k)
density = np.array(density)
scores = get_scores(density, min_matrix, all_RD, k)
mode = np.mean(modeList)

Write_data_file(all_chr, all_start, all_end, all_RD, scores)
upper = boxplot(scores)
CNV_chr, CNVstart, CNVend, CNVRD, CNVtype = combiningCNV(all_chr, all_start, all_end, all_RD, scores, upper, mode)
CN = calculating_CN(mode, CNVRD, CNVtype)
Write_CNV_File(CNV_chr, CNVstart, CNVend, CNVtype, CN, outfile, CNVRD)



#cir
ave_RD = np.mean(all_RD)
homo_RD, hemi_RD = clusting(0)  
ave_rfa = calculating_ave_rfa(homo_RD, hemi_RD, ave_RD)
new_RD = recover_RD(ave_RD, ave_rfa, scores, all_RD)


if(ave_rfa != 0.0):

    print("retest CNV...")
    dis, newpos = dis_matrix(new_RD)
    dis, min_matrix = k_matrix(dis, k)
    min_matrix = min_matrix.astype(np.int)
    density = reach_density(dis, min_matrix, k)
    density = np.array(density)
    scores = get_scores(density, min_matrix, new_RD, k)
    mode = modeRD(new_RD)
    upper = boxplot(scores)
    CNV_chr1, CNVstart1, CNVend1, CNVRD1, CNVtype1 = combiningCNV(all_chr, all_start, all_end, new_RD, scores, upper, mode)
    CN1 = calculating_CN(mode, CNVRD1, CNVtype1)
    Write_CNV_newFile(CNV_chr1, CNVstart1, CNVend1, CNVtype1, CN1, newoutfile)



endtime = datetime.datetime.now()
print("running time: " + str((endtime - starttime).seconds) + " seconds")
