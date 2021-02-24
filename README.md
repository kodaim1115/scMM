# scMM: Mixture-of-experts multimodal deep generative model for single-cell multiomics analysis

![figure](https://github.com/kodaim1115/scMM/blob/master/overview.png)

scMM is a novel deep generative model-based framework for the extraction of interpretable joint representations and cross-modal generation for single-cell multiomics data (e.g. transcriptome & chromatin accessibility, transcriptome & surface proteins). It is based on a mixture-of-experts multimodal deep generative model and achieves end-to-end learning by modeling raw count data in each modality based on different probability distributions.

`colab_tutorial.ipynb` shows how to run scMM using GPU on Google Colab.
For the tutorial, we use toy data generated from CITE-seq (single-cell transctiptome & surface protein) data for bone marrow mononuclear cell (BMNC) including randomely subsampled 15,000 cells ([Stuart and Butler et. al., 2018](https://www.cell.com/cell/fulltext/S0092-8674(19)30559-8#%20)). Most varaible 5000 genes were selected for transcriptome data.

RNA and protein count matrix should be stored in folder named `RNA-seq` and `CITE-seq` accomapnied with feature information stored in `gene.tsv` and `protein.tsv`, respectively. Also, single-cell barcode stored in `barcode.tsv` should be included. When running on chromatin accessibility data, name folder as `ATAC-seq` and feature file as `peak.tsv`. For example, folder structure looks like:
```
data/BMNC
     |---RNA-seq
     |   |---RNA_count.mtx
     |   |---gene.tsv
     |   |---barcode.tsv
     |---CITE-seq
         |---Protein_count.mtx
         |---protein.tsv
         |---barcode.tsv
```

Tutorial on downstream analysisfor scMM outputs can be found at `R/tutorial.R`. 
Vignette is available [here](http://htmlpreview.github.io/?https://github.com/kodaim1115/test/blob/master/tutorial.html).
Codes were adopted from the [MMVAE repository](https://github.com/iffsid/mmvae). 

Check out our [preprint](https://www.biorxiv.org/content/10.1101/2021.02.18.431907v1.full) for more details on the methods. 


