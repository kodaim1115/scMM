#Output data with most variable genes
setwd("~/proj/multimodal/")

library(Seurat)
library(data.table)
library(Matrix)

data_path <- "data/Seurat_v3"
filtered_path <- "data/Seurat_v4/filtered_RNA-seq"

#RNA
data <- Matrix::readMM(paste0(data_path,"/RNA-seq/RNA_count.mtx"))
dim(data)
gene_v3 <- as.matrix(fread(file=paste0(data_path,"/RNA-seq/gene.tsv"),
                           header=FALSE,sep="\t"))[,1]
length(gene_v3)

gene_v4 <- as.matrix(fread(file=paste0(filtered_path,"/gene.tsv"),
                        header=FALSE,sep="\t"))[,1]
length(gene_v4)

data_ <- rbind(data, rep(0,ncol(data)))
dim(data_)

id <- match(gene_v4, gene_v3)
sum(is.na(id)) #1238
id[is.na(id)] <- nrow(data_)

sel_data <- data_[id,]
dim(sel_data)

save_path <- "data/Seurat_v3_genes_selected/filtered_RNA-seq"
dir.create(save_path)
writeMM(sel_data,file=paste0(save_path,"/RNA_count.mtx"))

#Dummy data
Zeromat <- function(ni,nj=NULL) {
  if(is.null(nj)) nj <- ni
  return(as(sparseMatrix(i={},j={},dims=c(ni,nj)),"dgCMatrix"))
}

cite_dim <- 224
dummy <- matrix(50,nrow=cite_dim, ncol=ncol(data_))
dummy <- Matrix(dummy, sparse=TRUE)
save_path2 <- "data/Seurat_v3_genes_selected/filtered_CITE-seq"
dir.create(save_path2)
writeMM(dummy,file=paste0(save_path2,"/Protein_count.mtx"))
