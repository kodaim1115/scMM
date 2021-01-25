library(data.table)
library(ggplot2)
library(RColorBrewer)

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

PlotEmbCluster <- function(lat_emb,cluster){
  color <- gg_color_hue(length(unique(cluster)))
  
  uniq_clus <- sort(unique(cluster))
  loc <- c()
  for(i in 1:length(uniq_clus)){
    loc[i] <- min(which(cluster == uniq_clus[i]))
  }
  
  lat_emb <- data.frame(UMAP1=lat_emb[,1],UMAP2=lat_emb[,2],
                        Cluster=factor(cluster, levels=1:length(unique(cluster))))
  
  gg <- ggplot(lat_emb,  aes(x = UMAP1, y = UMAP2, color = Cluster)) +
    geom_point(size = 0.1) +
    ggtitle("Clustering") + 
    theme_classic() +
    scale_color_manual(values = color) +
    guides(color = guide_legend(override.aes = list(size = 3), ncol = 2)) +
    annotate("text", x=lat_emb[,1][loc], y=lat_emb[,2][loc], label=as.character(uniq_clus), size=3)
  return(gg)
}

#By each dimension
PlotEmbSelectDim <- function(lat_emb, lat, dim){
  gg <- list()
  df <- data.frame(UMAP1=lat_emb[,1],UMAP2=lat_emb[,2])
  scale <- scale(lat[,dim])
  color <- scale
  df[,"color"] <- color
  gg <- ggplot(df,  aes(x=UMAP1, y=UMAP2, color=color)) +
    geom_point(size = 0.1) +
    ggtitle(paste0("Dimension ", dim)) +
    theme_classic() +
    theme(legend.title=element_blank()) +
    scale_color_gradientn(colours = colorRampPalette(rev(brewer.pal(n = 11, name = "Spectral")))(50))
  return(gg)
}

#Function for Spearman correlation test
CorTest <- function(x,y, alpha){
  res <- cor.test(x, y, method="spearman")
  if(res$p.value < alpha){
    res <- res$estimate
  }else{
    res <- 0
  }
  return(res)
}

AssociatedGene <- function(dimension){
  i <- dimension
  lim <- 0
  up_id <- which(cor_up_g[[i]] > lim)
  down_id <- which(cor_down_g[[i]] < -lim)
  
  up_name <- gene_up[[i]][up_id]
  down_name <- gene_down[[i]][down_id]
  
  up_c <- cor_up_g[[i]][up_id]
  down_c <- cor_down_g[[i]][down_id]          
  
  length(up_name)
  length(down_name)
  
  up_show <- 10
  down_show <- 10
  if(length(up_name) > up_show){
    dum_up <- rep(NA, length(up_name))
    dum_up[1:up_show] <- up_name[1:up_show]
  }else{
    dum_up <- up_name
  }
  if(length(down_name) > down_show){
    dum_down <- rep(NA, length(down_name))
    dum_down[1:down_show] <- down_name[1:down_show]
  }else{
    dum_down <- down_name
  }
  
  df <- data.frame(Correlation=abs(c(up_c, rev(down_c))),
                   Gene=c(dum_up, rev(dum_down)),
                   updown=c(rep("up", length(up_name)), rep("down", length(down_name))),
                   id=c(rev(1:length(up_name)), c(-1:-length(down_name))))
  
  gg <- ggplot(df, aes(x=id, y=Correlation, color=updown)) +
    theme_bw() +
    geom_point(size=0.5) +
    geom_vline(xintercept=0, linetype="dashed") +
    ggtitle(paste0("Dimension ", i, " associated genes")) +
    xlab("Genes") +
    scale_color_manual(breaks = c("up", "down"),
                       values=c("red", "blue")) +
    theme(legend.position="none", 
          axis.text.x=element_blank()) +
    ggrepel::geom_text_repel(data=df, mapping=aes(label=df[,"Gene"]),size=3, segment.alpha=0.5)
}

AssociatedPeak <- function(dimension){
  i <- dimension 
  lim <- 0
  up_id <- which(cor_up_p[[i]] > lim)
  down_id <- which(cor_down_p[[i]] < -lim)
  
  up_name <- peak_up[[i]][up_id]
  down_name <- peak_down[[i]][down_id]
  
  up_c <- cor_up_p[[i]][up_id]
  down_c <- cor_down_p[[i]][down_id]          
  
  length(up_name)
  length(down_name)
  
  up_show <- 5
  down_show <- 5
  if(length(up_name) > up_show){
    dum_up <- rep(NA, length(up_name))
    dum_up[1:up_show] <- up_name[1:up_show]
  }else{
    dum_up <- up_name
  }
  if(length(down_name) > down_show){
    dum_down <- rep(NA, length(down_name))
    dum_down[1:down_show] <- down_name[1:down_show]
  }else{
    dum_down <- down_name
  }
  
  df <- data.frame(Correlation=abs(c(up_c, rev(down_c))),
                   Gene=c(dum_up, rev(dum_down)),
                   updown=c(rep("up", length(up_name)), rep("down", length(down_name))),
                   id=c(rev(1:length(up_name)), c(-1:-length(down_name))))
  
  gg <- ggplot(df, aes(x=id, y=Correlation, color=updown)) +
    theme_bw() +
    geom_point(size=0.5) +
    geom_vline(xintercept=0, linetype="dashed") +
    ggtitle(paste0("Dimension ", i, " associated peaks")) +
    scale_color_manual(breaks = c("up", "down"),
                       values=c("red", "blue")) +
    theme(legend.position="none", 
          axis.text.x=element_blank()) +
    xlab("Peaks") +
    ggrepel::geom_text_repel(data=df, mapping=aes(label=df[,"Gene"]), size=2.7, segment.alpha=0.5) 
  
}

AssociatedProtein <- function(dimension){
  i <- dimension
  lim <- 0
  up_id <- which(cor_up_p[[i]] > lim)
  down_id <- which(cor_down_p[[i]] < -lim)
  
  up_name <- protein_up[[i]][up_id]
  down_name <- protein_down[[i]][down_id]
  
  up_c <- cor_up_p[[i]][up_id]
  down_c <- cor_down_p[[i]][down_id]          
  
  length(up_name)
  length(down_name)
  
  up_show <- 9
  down_show <- 9
  if(length(up_name) > up_show){
    dum_up <- rep(NA, length(up_name))
    dum_up[1:up_show] <- up_name[1:up_show]
  }else{
    dum_up <- up_name
  }
  if(length(down_name) > down_show){
    dum_down <- rep(NA, length(down_name))
    dum_down[1:down_show] <- down_name[1:down_show]
  }else{
    dum_down <- down_name
  }
  
  df <- data.frame(Correlation=abs(c(up_c, rev(down_c))),
                   Gene=c(dum_up, rev(dum_down)),
                   updown=c(rep("up", length(up_name)), rep("down", length(down_name))),
                   id=c(rev(1:length(up_name)), c(-1:-length(down_name))))
  
  gg <- ggplot(df, aes(x=id, y=Correlation, color=updown)) +
    theme_bw() +
    geom_point(size=0.5) +
    geom_vline(xintercept=0, linetype="dashed") +
    ggtitle(paste0("Dimension ", i, " associated proteins")) +
    xlab("Proteins") +
    scale_color_manual(breaks = c("up", "down"),
                       values=c("red", "blue")) +
    theme(legend.position="none", 
          axis.text.x=element_blank()) +
    ggrepel::geom_text_repel(data=df, mapping=aes(label=df[,"Gene"]),segment.alpha=0.5)
  
}


#Create pseudobulk data by aggregating per cluster
CreateAgg <- function(data, cluster){
  #Aggregate clusters 
  agg <- matrix(0,length(unique(cluster)),ncol(data))
  for(i in c(1:length(unique(cluster)))){
    id <- which(cluster==i)
    tmp <- data[id,]
    agg[i,] <- colSums(tmp)
  }
  sum <- rowSums(agg)
  agg <- agg/sum #Normalize by total counts per gene
  return(agg)
}

