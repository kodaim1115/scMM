setwd("~/proj/multimodal/")

library(Seurat)
library(Signac)
library(EnsDb.Hsapiens.v86)
library(dplyr)
library(ggplot2)
library(data.table)
library(Matrix)

data_path <- "data/SHARE-seq"


#ATAC
count_a <- readMM(file=paste0(data_path, "/GSM4156597_skin.late.anagen.counts.mtx"))

peak <- fread(file=paste0(data_path, "/GSM4156597_skin.late.anagen.peaks.bed"),
                        header=FALSE, sep="\t")
barcode_a <- as.matrix(fread(file=paste0(data_path, "/GSM4156597_skin.late.anagen.barcodes.txt"),
                           header=FALSE))[,1]

df_peak <- data.frame(chr=peak$V1, start=peak$V2, end=peak$V3)
grange <- makeGRangesFromDataFrame(df_peak, seqnames.field="chr", 
                                    start.field="start",
                                    end.field="end")
peak <- paste0(peak$V1, ":", peak$V2, "-", peak$V3)

dim(count_a)
length(peak)
length(barcode_a)

rownames(count_a) <- peak
colnames(count_a) <- barcode_a

#RNA
count_r <- as.matrix(fread(file=paste0(data_path, "/GSM4156608_skin.late.anagen.rna.counts.txt"),
                           header=TRUE, drop="gene"))
count_r <- Matrix(count_r, sparse=TRUE)
gene <- as.matrix(fread(file=paste0(data_path, "/GSM4156608_skin.late.anagen.rna.counts.txt"),
                        header=TRUE, select="gene"))[,1]

barcode_r <- colnames(count_r)
barcode_r <- gsub(",", ".", barcode_r)

dim(count_r)
length(gene)
length(barcode_r)
rownames(count_r) <- gene
colnames(count_r) <- barcode_r

# match_r <- match(barcode_r, barcode_a)
# match_a <- match(barcode_a, barcode_r)
# match_r <- match_r[-which(is.na(match_r))]
# match_a <- match_a[-which(is.na(match_a))]

match <- match(barcode_r, barcode_a)
match <- match[-which(is.na(match))]
common_barcode <- barcode_a[match]
length(common_barcode)

#Select common cells
count_r <- count_r[,match(common_barcode, barcode_r)]
count_a <- count_a[,match(common_barcode, barcode_a)]

d <- CreateSeuratObject(counts=count_a, assay="ATAC")
d_r <- CreateAssayObject(counts=count_r)
d[["RNA"]] <- d_r
#save(d, file="data/Seurat_SHARE-seq.R")
load(file="data/Seurat_SHARE-seq.R")

#Get mt RNA percentage
d[["percent.mt"]] <- PercentageFeatureSet(d, assay="RNA", pattern="mt-")

#Filter RNA
DefaultAssay(d) <- "RNA"
d <- NormalizeData(d, normalization.method = "LogNormalize", scale.factor = 10000)
nfeatures <- 5000
d <- FindVariableFeatures(d, selection.method = "vst", nfeatures = nfeatures)

filtered_count <- d@assays$RNA@counts[match(VariableFeatures(d), rownames(d@assays$RNA@counts)),]
dim(filtered_count) #5000 34774

filtered_gene <- matrix(VariableFeatures(d), ncol=1)
dim(filtered_gene) 

filtered_path <- paste0(data_path, "/filtered_RNA-seq")
dir.create(filtered_path)

writeMM(filtered_count,file=paste0(filtered_path,"/RNA_count.mtx"))

write.table(common_barcode,file=paste0(filtered_path,"/barcode.tsv"),sep="\t",row.names=FALSE,col.names=FALSE)

write.table(filtered_gene,file=paste0(filtered_path,"/gene.tsv"),sep="\t",row.names=FALSE,col.names=FALSE)


#Filter ATAC
DefaultAssay(d) <- "ATAC"
d <- FindTopFeatures(d, min.cutoff = 'q75')
filtered_count <- d@assays$ATAC@counts[match(VariableFeatures(d), rownames(d@assays$ATAC@counts)),]
dim(filtered_count)

filtered_peak <- matrix(VariableFeatures(d), ncol=1)
dim(filtered_peak) 

filtered_path <- paste0(data_path, "/filtered_ATAC-seq")
dir.create(filtered_path)

writeMM(filtered_count,file=paste0(filtered_path,"/ATAC_count.mtx"))

write.table(common_barcode,file=paste0(filtered_path,"/barcode.tsv"),sep="\t",row.names=FALSE,col.names=FALSE)

write.table(filtered_peak,file=paste0(filtered_path,"/peak.tsv"),sep="\t",row.names=FALSE,col.names=FALSE)

