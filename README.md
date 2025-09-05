# M2Diff
M2Diff: A Multi-modality Mamba Diffusion Model for Medical Image Segmentation

We propose an innovative multi-modality mamba diffusion Model (M2Diff) for 3D medical image segmentation. Our basic assumption is to use a U-shaped Independent Feature Encoder (UIFE) to extract the global semantic features of each modality during the diffusion process and conditionally adjust them with the backbone model. At the same time, in order to effectively capture the 3D spatial structure and context information, we propose a Global Context Feature Fusion Block (GCFFB), which can effectively capture global features while reducing the number of parameters by fusing Mamba and Large Kernel. Finally, the inherent features extracted from different modalities by UIFE and the predicted results are added to the cross-modality fusion denoising module (CMFDM) in the form of conditional encoding to perform the final integration of multi-modality images. 


## Dataset 
We conducted empirical experiments on six publicly accessible datasets, ISLES'15 (SISS and SPES), ISLES'22, BraTS2020, FeTS2021 and BraTS2023 dataset 
to evaluate the proposed network and modules' performance and utility.


## Code
The paper is being submitted and the full code will be published soon.
