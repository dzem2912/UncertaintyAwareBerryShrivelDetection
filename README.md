# Interpretable and Uncertainty-Aware Deep Learning for Grapevine Berry Shrivel Detection

Code and (pending release) dataset accompanying the paper:

> DŽ. Rožajac, S. Schweng, M. Griesser, A. Forneck, J. Del Ser, A. Holzinger.
> **Interpretable and Uncertainty-Aware Deep Learning for Grapevine Berry Shrivel Detection.**

Grapevine berry shrivel (BS) is a ripening disorder whose symptoms are mainly internal and not
reliably reflected in berry color, size or density, making it hard to diagnose from a photograph
alone — even for a trained grapevine physiologist. This repository provides the code used to:

- benchmark eight ImageNet-pretrained architectures on BS classification,
- extend the three best-performing models with an Evidential Deep Learning (EDL) output that
  reports a per-sample uncertainty score alongside each prediction, and
- generate LayerCAM explanations and compare them against a domain-expert annotation study.

## Dataset

- **1,664** close-up RGB grape-cluster images (**778** healthy, **886** BS-affected)
- Collected during the **2025** ripening season in commercial vineyards in **Lower Austria and
  Burgenland**, Austria
- Ground truth assigned by a trained grapevine physiologist, following the cluster-scale symptom
  descriptions of Griesser et al. (2012)
- Acquisition was **not standardized** across viewing angle or illumination

> **Availability:** 
> The dataset is available for download via the following link:

## Method summary

**Classification backbones** (fine-tuned end-to-end, ImageNet-pretrained): ResNet-18, ResNet-34,
ResNet-50, DenseNet-121, EfficientNet-B0, MobileNet-V3-Large, ConvNeXt-Tiny, ViT-B/16.

**Table 1 — test-set performance, all eight architectures**

| Architecture | Params | Bal. Acc. | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 4.01 M | 0.913 | 0.913 | 0.913 | 0.913 | 0.062 |
| MobileNet-V3-Large | 4.20 M | 0.917 | 0.917 | 0.917 | 0.917 | 0.075 |
| DenseNet-121 | 6.96 M | 0.901 | 0.901 | 0.901 | 0.901 | 0.068 |
| ResNet-18 | 11.18 M | 0.906 | 0.914 | 0.906 | 0.908 | 0.076 |
| ResNet-34 | 21.29 M | 0.913 | 0.913 | 0.913 | 0.913 | 0.072 |
| ResNet-50 | 23.51 M | 0.913 | 0.913 | 0.913 | 0.913 | 0.066 |
| **ConvNeXt-Tiny** | 27.82 M | **0.966** | **0.967** | **0.966** | **0.966** | **0.027** |
| ViT-B/16 | 85.80 M | 0.869 | 0.875 | 0.869 | 0.871 | 0.114 |

ConvNeXt-Tiny is selected as the primary model on balanced accuracy and Brier score.

**Evidential Deep Learning.** ResNet-18, EfficientNet-B0 and ConvNeXt-Tiny are each extended with
an EDL output layer (Sensoy et al., 2018).

**Table 2 — base vs. EDL, with and without Platt scaling**

| Model | Variant | Bal. Acc. | Precision | F1 | Brier | ECE |
|---|---|---|---|---|---|---|
| ResNet-18 | Base | 0.906 | 0.914 | 0.908 | 0.076 | 0.055 |
| | Base + Platt | 0.909 | 0.942 | 0.898 | 0.070 | 0.031 |
| | EDL | 0.914 | 0.890 | 0.909 | 0.068 | 0.057 |
| | EDL + Platt | 0.914 | 0.890 | 0.909 | 0.067 | 0.043 |
| EfficientNet-B0 | Base | 0.913 | 0.913 | 0.913 | 0.062 | 0.046 |
| | Base + Platt | 0.922 | 0.927 | 0.915 | 0.058 | 0.013 |
| | EDL | 0.909 | 0.933 | 0.899 | 0.059 | 0.055 |
| | EDL + Platt | 0.918 | 0.912 | 0.912 | 0.055 | 0.022 |
| ConvNeXt-Tiny | Base | 0.966 | 0.973 | 0.966 | 0.027 | 0.009 |
| | Base + Platt | 0.967 | 0.973 | 0.964 | 0.028 | 0.031 |
| | EDL | 0.963 | 0.964 | 0.960 | 0.032 | 0.054 |
| | EDL + Platt | 0.966 | 0.973 | 0.964 | 0.031 | 0.048 |

ECE computed on the BS-class probability with M = 10 quantile bins.

**Explainability.** LayerCAM (Jiang et al., 2021) is used to generate per-image, per-class
attention heatmaps, compared against a study with three domain experts who independently
diagnosed and annotated a 48-image subset of the held-out test set.


All experiments in the paper were run on a single NVIDIA Quadro RTX 4000 GPU, 20 epochs max,
batch size 32, early stopping (patience 7) on validation accuracy.

## Citation


## Funding

This research was funded by the Gesellschaft für Forschungsförderung (GFF) Lower Austria as part
of the FTI-strategy Lower Austria 2027 through project FTI23-A-014, and in part by the Austrian
Science Fund (FWF) [10.55776/PAT9471624]. J. Del Ser acknowledges funding support from the Basque
Government through the consolidated research group MATHMODE (IT1866-26).

## License

## Contact
