from .vae_atac import ATAC as VAE_atac
from .vae_rna import RNA as VAE_rna
from .vae_protein import Protein as VAE_protein

from .mmvae_rna_atac import RNA_ATAC as VAE_rna_atac
from .mmvae_rna_protein import RNA_Protein as VAE_rna_protein


__all__ = [VAE_rna, VAE_atac, VAE_protein, 
           VAE_rna_atac, VAE_rna_protein]
