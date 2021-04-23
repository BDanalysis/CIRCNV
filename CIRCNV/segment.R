library(cghFLasso)
"segment" <- function(data){
   seg<-cghFLasso(data, FDR=0.01)
   segdata<-seg$Esti.CopyN
   out.file="seg.txt"
   write.table(segdata, file=out.file, sep="\t")
}