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

It is easy to skim Chapter 02—nominal vs. ordinal scales, mean vs. median, boxplots, Anscombe’s quartet—and assume it is just introductory warm-up before the “real” math starts. Then you step into a modern AI lab and discover that some of the biggest leaps in vision, language, and bioinformatics from 2021 to 2025 did **not** come from inventing a new neural architecture. They came from **descriptive statistics at scale**: filtering billions of noisy web pairs, catching corrupted class labels, and exposing misleading 2D visualizations!

This optional note maps Chapter 02’s three foundational habits—*what was measured*, *how it was summarized*, and *what the picture hid*—onto four landmark data-centric systems: **DataComp**, **Confident Learning (`cleanlab`)**, **t-SNE/UMAP distortion**, and **TabPFN**.

---

## Background Knowledge Map: Why Descriptive Statistics Drives Data-Centric AI

Three core statistical ideas sit beneath every modern data-curation pipeline.

### 1. Measurement Scales & Data Curation (Nominal, Ordinal, Ratio)
Every column in a table carries a **measurement scale** that dictates which operations are meaningful (averaging ZIP codes is nonsense; thresholding similarity scores is valid). **Data-centric AI** treats dataset filtering—deduplication, language tags, score histograms—as a primary scientific intervention rather than a throwaway preprocessing step.
- **Core questions:** How much model accuracy comes from better data curation versus bigger architectures?
- **Real-world reach:** Web-scale multimodal pretraining (DataComp, DataComp-LM), medical registries, and tabular foundation models (TabPFN).

### 2. Robust Location & Scale Estimators (Breakdown Point)
The sample **mean** and **standard deviation** have a **breakdown point of 0%**—a single corrupted outlier can drag them to infinity! By contrast, the **median** has a **50% breakdown point** and the **interquartile range (IQR)** has a **25% breakdown point**. The exact same robustness logic extends to categorical labels when 3–10% of human annotations are wrong.
- **Core questions:** Which summaries stay stable under sensor glitches or label noise? How can out-of-sample probabilities flag mislabeled rows?
- **Real-world reach:** `RobustScaler`, `cleanlab` label auditing, financial outlier filtering, and industrial IoT pipelines.

### 3. Visual Statistics & Dimensionality-Reduction Distortion
Just as **Anscombe’s quartet** proved that identical means, variances, and correlations can hide four radically different scatterplots, modern nonlinear embeddings (**t-SNE**, **UMAP**) are *statistics* that inevitably distort global distances when squashing 2,000 dimensions onto a flat 2D screen.
- **Core questions:** Which geometric features (local neighbors vs. global distances) does a visualization preserve, and which does it fabricate?
- **Real-world reach:** Single-cell genomics, LLM embedding explorers, and exploratory anomaly detection.

---

## Dataset design as descriptive statistics at scale

Gadre, Ilharco, Fang, and colleagues (*DataComp*, [NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html); [arXiv:2304.14108](https://arxiv.org/abs/2304.14108); [datacomp.ai](https://www.datacomp.ai)) froze the training code of a CLIP model and varied **only the training data**. Starting from a 12.8-billion image–text pool (CommonPool), they applied Chapter 02-style descriptive filters: English-language tags, CLIP similarity quantiles, deduplication, and aspect-ratio bounds. Their curated subset, **DataComp-1B** (1.4B pairs), trained a ViT-L/14 to **79.2%** ImageNet zero-shot accuracy—**+3.7 points** ahead of OpenAI’s CLIP ViT-L/14 at matched compute! A language-model counterpart, **DataComp-LM** (Li et al., NeurIPS 2024; [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)), confirmed the same rule for LLMs. When you inspect a histogram and trim junk rows in Pandas, you are running a miniature DataComp filter.

## Label errors and robust summaries

A nominal class label is a measurement scale. If 5% of ImageNet tags are wrong, the sample mean of “accuracy” is a contaminated location estimate—the discrete cousin of a corrupted sensor reading. Northcutt, Jiang, and Chuang (*Confident Learning*, [JAIR 2021](https://doi.org/10.1613/jair.1.12125)) estimate the joint distribution of noisy labels $$\tilde{Y}$$ and latent clean labels $$Y$$ from out-of-sample predicted probabilities, flagging likely mislabeled examples inside the open-source [cleanlab](https://docs.cleanlab.ai/) library.

---

### Worked Example 1: Mean/SD vs. Median/IQR Under One Corrupted Sensor Reading

Suppose five temperature readings are cleanly recorded as $$\{10,\; 11,\; 12,\; 13,\; 14\}$$, and then a single glitch corrupts $$14 \to 100$$, giving $$\{10,\; 11,\; 12,\; 13,\; 100\}$$. Compute both summaries by hand:

| Summary Statistic | Clean data $$\{10,11,12,13,14\}$$ | Corrupted data $$\{10,11,12,13,100\}$$ | Shift |
|-------------------|-----------------------------------|----------------------------------------|-------|
| **Mean** $$\bar{x}$$ | $$12.0$$ | $$\frac{146}{5} = \mathbf{29.2}$$ | **+143% jump!** |
| **Std. Dev.** $$s$$ (population) | $$\sqrt{2} \approx 1.41$$ | $$\sqrt{1253.76} \approx \mathbf{35.41}$$ | **25× explosion!** |
| **Median** | $$12.0$$ | **$$12.0$$** | **0% change!** |
| **IQR** ($$Q_3 - Q_1$$) | $$13 - 11 = 2.0$$ | $$13 - 11 = \mathbf{2.0}$$ | **0% change!** |

One broken entry out of five completely destroys `StandardScaler` (mean and SD), while `RobustScaler` (median and IQR) does not flinch!

---

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

rng = np.random.default_rng(0)
x = rng.normal(size=(200, 1))
x[0] = 40.0  # one broken sensor reading
print("mean / sd:", StandardScaler().fit(x).mean_[0], StandardScaler().fit(x).scale_[0])
print("median / IQR-scale:", RobustScaler().fit(x).center_[0], RobustScaler().fit(x).scale_[0])
```

The printed mean jumps; the median barely moves. That is why data-centric papers talk about *cleaning* before *fitting*.

---

### Worked Example 2: Flagging a Label Error with Confident Learning by Hand

Suppose a cat-vs-dog classifier produces out-of-sample predicted probabilities on 4 images labeled **Cat** ($$\tilde{y} = \text{Cat}$$):
- Image 1: $$P(\text{Cat}) = 0.92,\; P(\text{Dog}) = 0.08$$
- Image 2: $$P(\text{Cat}) = 0.85,\; P(\text{Dog}) = 0.15$$
- Image 3: $$P(\text{Cat}) = 0.78,\; P(\text{Dog}) = 0.22$$
- Image 4: $$P(\text{Cat}) = 0.09,\; P(\text{Dog}) = 0.91$$

How does **Confident Learning** decide whether Image 4 is actually a mislabeled Dog?
1. Compute the **class confidence threshold** $$t_{\text{Dog}}$$ = average predicted $$P(\text{Dog})$$ across all images labeled Dog (suppose $$t_{\text{Dog}} = 0.82$$).
2. Compare Image 4’s out-of-sample $$P(\text{Dog}) = 0.91 > t_{\text{Dog}} = 0.82$$!
3. Because Image 4’s self-confidence for its *given* label ($$0.09$$) is tiny while its confidence for **Dog** clears the Dog threshold ($$0.91 > 0.82$$), `cleanlab` places Image 4 into the off-diagonal cell $$(\tilde{Y}=\text{Cat},\; Y^*=\text{Dog})$$ of the confident joint matrix and flags it for human review!

---

## Pictures that lie: embeddings as modern misleading graphs

Anscombe’s quartet already showed four datasets with identical means, variances, and correlations yet four radically different shapes. The 2020s version is a 2D **t-SNE** or **UMAP** scatter plot of a 2,000-gene count matrix. Chari and Pachter (*PLOS Computational Biology*, 2023; [doi:10.1371/journal.pcbi.1011288](https://doi.org/10.1371/journal.pcbi.1011288)) proved mathematically and empirically that crushing thousands of dimensions into two *necessarily* warps global distances and cluster sizes—yet analysts routinely read distances between two UMAP islands as if they were real biological distances! The Chapter 02 moral survives intact: **every plot is a statistic**. Report what it preserves and what it distorts, and never run clustering directly on a 2D t-SNE/UMAP projection.

## Mixed types and a prior over tables

Hollmann, Müller, Eggensperger, and Hutter (*TabPFN*, [ICLR 2023](https://iclr.cc/virtual/2023/poster/12113); [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)) trained a Transformer once on synthetic tables drawn from a structural prior over mixed numeric/categorical features and causal graphs, allowing it to classify a brand-new small table in a single forward pass. The statistical takeaway is crystal clear: tabular modeling always begins by declaring which columns are **nominal**, **ordinal**, or **ratio**—ignoring measurement scales is not “end-to-end,” it is misspecified.

---

### Deep-Dive Explanation: Four Habits of Data-Centric Statistical Hygiene (Read Slowly)

Whenever you inherit a messy real-world dataset—whether 500 rows or 5 billion—run through these four checks slowly before touching a complex model.

#### Step 1: Audit every column’s measurement scale first
Separate nominal IDs/categories from ordered ranks and continuous ratios. Encoding a nominal category (`city_id = 1, 2, 3`) as a numeric integer silently injects fake arithmetic ($$1 + 2 = 3$$) into linear and distance-based models.

#### Step 2: Check breakdown sensitivity before standardizing
Compare mean vs. median and SD vs. IQR for every numeric feature. If the mean and median diverge sharply, a handful of outliers or heavy tails will distort ordinary standardization—switch to `RobustScaler` or a log transform (`log1p`).

#### Step 3: Audit the target labels with out-of-sample probabilities
Never assume training labels are ground truth. Train a simple cross-validated baseline and inspect the top-20 rows with the largest loss or lowest predicted probability on their given label—you will often uncover systematic labeling bugs worth more than weeks of model tuning.

#### Step 4: Treat 2D embeddings (t-SNE / UMAP) as exploratory sketches, not metric maps
Use t-SNE/UMAP to spot local clusters or data glitches, then verify every hypothesis back in the **original feature space** (using actual distances, gene expression levels, or summary statistics) before drawing scientific conclusions.

---

## What to carry forward

When a paper says “we filtered Common Crawl by CLIP score,” “we used cleanlab on the labels,” or “do not cluster on the UMAP,” it is speaking Chapter 02. Later probability chapters add sampling models; they do not replace the discipline of type, summary, and picture.

## Video tham khảo

- [StatQuest: t-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM) — StatQuest with Josh Starmer, English. **xem trước.** t-SNE keeps local neighborhoods and bends global distances — the Chari–Pachter warning this note calls a modern misleading graph.
- [UMAP Dimension Reduction, Main Ideas!!!](https://www.youtube.com/watch?v=eN0wFzBA4Sc) — StatQuest with Josh Starmer, English. UMAP is the same kind of nonlinear embedding; the two-dimensional picture is not the original data in the embedding section.
- [GenAI and Datacomp: Creating the Largest Public Multimodal Dataset in Academia](https://www.youtube.com/watch?v=ba-aqPF6xuw) — AI Council, English. DataComp filters an image–text pool with scores and frequencies — descriptive statistics as the intervention in the dataset-design section.

## Sources

1. S. Y. Gadre et al., “DataComp: In search of the next generation of multimodal datasets,” NeurIPS 2023. [abstract](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html) · [arXiv:2304.14108](https://arxiv.org/abs/2304.14108)
2. J. Li et al., “DataComp-LM: In search of the next generation of training sets for language models,” NeurIPS 2024. [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)
3. C. G. Northcutt, L. Jiang, and I. L. Chuang, “Confident Learning: Estimating Uncertainty in Dataset Labels,” *JAIR* 70, 2021. [DOI](https://doi.org/10.1613/jair.1.12125) · [cleanlab](https://docs.cleanlab.ai/)
4. T. Chari and L. Pachter, “The specious art of single-cell genomics,” *PLOS Comput. Biol.* 19(8), 2023. [DOI](https://doi.org/10.1371/journal.pcbi.1011288)
5. N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter, “TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second,” ICLR 2023. [ICLR](https://iclr.cc/virtual/2023/poster/12113) · [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
6. scikit-learn, `RobustScaler`. [docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
