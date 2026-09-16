---
layout: post
title: 02-04-00 Applications and recent developments
chapter: "02"
order: 4
owner: nglelinh
lang: en
categories:
- chapter02
lesson_type: optional
---

Data types, numerical summaries, and plots in this chapter look elementary until a pipeline treats them as afterthoughts. Modern computer-vision and tabular systems fail first on *what was measured*, *how it was summarized*, and *what the picture hid*. This optional note maps those three skills onto data-centric work from about 2021–2025. Required theory is unchanged; the goal is to see the same measurement scales and summaries inside dataset design, label cleaning, and embedding plots.

## Dataset design as descriptive statistics at scale

Gadre, Ilharco, Fang, and colleagues (*DataComp*, [NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html); [arXiv:2304.14108](https://arxiv.org/abs/2304.14108); [datacomp.ai](https://www.datacomp.ai)) freeze the training code of a CLIP model and *vary the training set*. A 12.8-billion image–text pool (CommonPool) is filtered by simple, Chapter 02-style rules: English text, CLIP similarity scores, deduplication, and related quality cuts. Their best subset, DataComp-1B (1.4B pairs), trains a ViT-L/14 to 79.2% ImageNet zero-shot—3.7 points above OpenAI’s CLIP ViT-L/14 at matched compute. The lesson is blunt: mean similarity, frequency of language tags, and a histogram of scores are not “EDA homework.” They *are* the intervention. A later language-model analogue is DataComp-LM (Jeffrey Li, Fang, Smyrnis, Ivgi, and colleagues, NeurIPS 2024; [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)). When you compute a median or drop outliers in Pandas, you are running a tiny DataComp filter.

## Label errors and robust summaries

A nominal class label is a measurement scale. If 5% of ImageNet tags are wrong, the sample mean of “accuracy” is a contaminated location estimate—the same sensitivity the mean-versus-median example in the numerical-measures lesson already showed. Northcutt, Jiang, and Chuang (*Confident Learning*, [JAIR 2021](https://doi.org/10.1613/jair.1.12125)) estimate the joint of noisy labels $$\tilde{Y}$$ and latent clean labels $$Y$$ from out-of-sample predicted probabilities, then flag likely errors. The open-source [cleanlab](https://docs.cleanlab.ai/) package is the production form. sklearn’s `RobustScaler` (median and IQR instead of mean and standard deviation) is the continuous analogue: choose a summary that matches the scale and the contamination you actually have.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

rng = np.random.default_rng(0)
x = rng.normal(size=(200, 1))
x[0] = 40.0  # one broken sensor reading
print("mean / sd:", StandardScaler().fit(x).mean_[0], StandardScaler().fit(x).scale_[0])
print("median / IQR-scale:", RobustScaler().fit(x).center_[0], RobustScaler().fit(x).scale_[0])
```

The printed mean jumps; the median barely moves. That is not a new formula. It is why data-centric papers talk about *cleaning* before *fitting*.

## Pictures that lie: embeddings as modern misleading graphs

Anscombe’s quartet in the visualization lesson already showed four datasets with matching means, variances, and correlations and four different shapes. The 2020s version is a two-dimensional t-SNE or UMAP of a 2,000-gene count matrix. Chari and Pachter (*PLOS Computational Biology*, 2023; [doi:10.1371/journal.pcbi.1011288](https://doi.org/10.1371/journal.pcbi.1011288)) show that collapsing hundreds of dimensions to two *necessarily* distorts distances and neighborhoods, and that biologists then treat the scatter as if it were the data. The Chapter 02 moral survives: a plot is a *statistic*. If the statistic is a nonlinear embedding, report what it preserves and what it invents. Prefer targeted views (a gene, a cluster score) over one “all-in-one” map.

## Mixed types and a prior over tables

Hollmann, Müller, Eggensperger, and Hutter (*TabPFN*, [ICLR 2023](https://iclr.cc/virtual/2023/poster/12113); [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)) train a Transformer once on synthetic tables drawn from a prior that includes mixed numeric features and simple causal graphs, then classify a *new* small table in a forward pass. The 2025 Nature follow-up extends the same idea. You do not need the architecture to use the statistical point: tabular work starts by declaring which columns are nominal, ordinal, or ratio—the first lesson of this chapter—and a method that ignores that declaration is not “end-to-end,” it is misspecified.

## What to carry forward

When a paper says “we filtered Common Crawl by CLIP score,” “we used cleanlab on the labels,” or “do not cluster on the UMAP,” it is speaking Chapter 02. Later probability chapters add sampling models; they do not replace the discipline of type, summary, and picture.

## Sources

1. S. Y. Gadre et al., “DataComp: In search of the next generation of multimodal datasets,” NeurIPS 2023. [abstract](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html) · [arXiv:2304.14108](https://arxiv.org/abs/2304.14108)
2. J. Li et al., “DataComp-LM: In search of the next generation of training sets for language models,” NeurIPS 2024. [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)
3. C. G. Northcutt, L. Jiang, and I. L. Chuang, “Confident Learning: Estimating Uncertainty in Dataset Labels,” *JAIR* 70, 2021. [DOI](https://doi.org/10.1613/jair.1.12125) · [cleanlab](https://docs.cleanlab.ai/)
4. T. Chari and L. Pachter, “The specious art of single-cell genomics,” *PLOS Comput. Biol.* 19(8), 2023. [DOI](https://doi.org/10.1371/journal.pcbi.1011288)
5. N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter, “TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second,” ICLR 2023. [ICLR](https://iclr.cc/virtual/2023/poster/12113) · [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
6. scikit-learn, `RobustScaler`. [docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
