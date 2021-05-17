#Output data with most variable genes
setwd("~/proj/multimodal/")

library(Seurat)
library(data.table)
library(Matrix)

data_path <- "data/Seurat_v3"

#RNA
data <- Matrix::readMM(paste0(data_path,"/RNA-seq/RNA_count.mtx"))
dim(data)

gene <- as.matrix(fread(file=paste0(data_path,"/RNA-seq/gene.tsv"),
                        header=FALSE,sep="\t"))[,1]

barcode <- as.matrix(fread(file=paste0(data_path,"/RNA-seq/barcode.tsv"),
                           header=FALSE,sep="\t"))[,1]

rownames(data) <- gene
colnames(data) <- barcode

d <- CreateSeuratObject(counts = data, 
                        project = "RNA", 
                        min.cells = 0, 
                        min.features = 0)

data <- NULL
d@assays$RNA@counts

d <- NormalizeData(d, normalization.method = "LogNormalize", scale.factor = 10000)
nfeatures <- 5000
d <- FindVariableFeatures(d, selection.method = "vst", nfeatures = nfeatures)

top10 <- head(VariableFeatures(d), 10)

plot1 <- VariableFeaturePlot(d)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

filtered_path <- paste0(data_path, "/filtered_RNA-seq")
dir.create(filtered_path)

filtered_count <- d@assays$RNA@counts[match(VariableFeatures(d), rownames(d@assays$RNA@counts)),]
dim(filtered_count) #5000 30672

filtered_barcode <- matrix(colnames(filtered_count), ncol=1)
dim(filtered_barcode) #30672 1

filtered_gene <- matrix(VariableFeatures(d), ncol=1)
dim(filtered_gene) #5000 1


writeMM(filtered_count,file=paste0(filtered_path,"/RNA_count.mtx"))

write.table(filtered_barcode,file=paste0(filtered_path,"/barcode.tsv"),sep="\t",row.names=FALSE,col.names=FALSE)

write.table(filtered_gene,file=paste0(filtered_path,"/gene.tsv"),sep="\t",row.names=FALSE,col.names=FALSE)

#Toy data
dir.create("data/Seurat_v3_toy/RNA-seq")
data <- readMM(file="data/Seurat_v3_toy/RNA-seq/RNA_count.mtx")
filtered_count <- data[match(VariableFeatures(d), rownames(d@assays$RNA@counts)),]
dim(filtered_count) #5000 10000

writeMM(filtered_count,file="data/Seurat_v3_toy/filtered_RNA-seq/RNA_count.mtx")
write.table(filtered_gene,file="data/Seurat_v3_toy/filtered_RNA-seq/gene.tsv",sep="\t",row.names=FALSE,col.names=FALSE)
