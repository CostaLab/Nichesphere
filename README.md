# Nichesphere

Nichesphere is an sc-verse compatible Python library which allows the user to find differentially colocalised cellular niches based on cell type pairs colocalisation probabilities in 
different conditions. Cell type pair colocalisation probabilities are obtained from deconvoluted spatial transcriptomics / PIC-seq data (probaabilities of finding each cell type in each spot 
/ multiplet) as the probabilities to find each cell type pair in the same spot/multiplet. 

It also offeres the possibility to look at differentially communicating cellular niches based on Ligand-Receptor pairs expression data, such as results from CrossTalkeR [ref.].

Moreover, Nichesphere combines colocalisation and communication networks via communication-weighted colocalisation based distances to find differentially interacting niches.
