# scMM: Mixture-of-experts multimodal deep generative model for single-cell multiomics analysis


`colab_tutorial.ipynb` shows how to run scMM using GPU on Google Colab.
For the tutorial, we use CITE-seq (single-cell transctiptome & surface protein) data for bone marrow mononuclear cell (BMNC) including randomely subsampled 15,000 cells. 

RNA and protein count matrix should be stored in folder named `RNA-seq` and `CITE-seq` accomapnied with feature information stored in `gene.tsv` and `protein.tsv`, respectively. Also, single-cell barcode stored in `barcode.tsv` should be included. When running on chromatin accessibility data, name folder as `ATAC-seq` and feature file as `peak.tsv`.

Tutorial on downstream analysisfor scMM outputs can be found at `R/tutorial.R`. 
Vignette is available [here](http://htmlpreview.github.io/?https://github.com/kodaim1115/test/blob/master/tutorial.html).

