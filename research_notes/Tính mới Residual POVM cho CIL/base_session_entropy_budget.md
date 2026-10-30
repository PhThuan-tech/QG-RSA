# Base-session entropy budget: penalizing von Neumann entropy S(rho_bar) at task 0 for forward compatibility in CIL

Scope: the novelty of penalizing S(rho_bar), with rho_bar = sum_c p_c rho_c (rho_c = second-moment matrix of L2-normalized features of class c), at task 0 of exemplar-free CIL using a pre-trained ViT plus an adapter. The aim is that base classes occupy fewer dimensions and later classes can use the complement.

Verification legend:
- [V] Checked this session: the abstract or page was fetched, or a search snippet from the primary page was read.
- [M] From prior knowledge, with an arXiv ID I am fairly confident of but did not fetch this session. Re-check before citing.

---

## Q1. Which CIL / FSCIL papers regularize the base session for forward compatibility, what do they regularize, and what gains do they report?

### Takeaway
"Reserve embedding space for future classes at the base session" is an established line of work: FACT (CVPR'22), NC-FSCIL (ICLR'23), OrCo (CVPR'24), CPPF (arXiv'23), and above all PRL (NeurIPS'24). PRL squeezes the base-class embedding distribution in non-exemplar CIL, which is the same motivation as the proposed idea. None of these papers states the objective as the von Neumann or spectral entropy of the class-mixture second-moment matrix. They use prototypes, margins, contrastive terms, or fixed ETF/orthogonal targets instead.

### Cited Findings
- **FACT** (Zhou, Wang, Ye, Ma, Pu, Zhan; CVPR 2022). ForwArd Compatible Training "reserv[es] embedding space for future new classes", using virtual prototypes that squeeze known-class embeddings and leave room for new ones. It reports SOTA on FSCIL benchmarks. [V] — [arXiv 2203.06953](https://arxiv.org/abs/2203.06953)
- **PRL, Prospective Representation Learning for Non-Exemplar CIL** (Wuxuan Shi, Mang Ye; NeurIPS 2024). This is the closest prior work. In the base phase it "compresses the embedding distribution of current classes to leave room for future classes" (module name: Preemptive Embedding Squeezing, PES). In incremental phases, Prototype-Guided Representation Update pushes new-class features away from saved old prototypes in a latent space. The paper calls it plug-and-play and reports that it beats SOTA NECIL methods on several benchmarks. [V for abstract and module names. I could not read the PDF text, so the exact PES loss form, the backbone, and the ablation numbers are not verified.] — [NeurIPS poster](https://neurips.cc/virtual/2024/poster/94608); [Proceedings PDF](https://proceedings.nips.cc/paper_files/paper/2024/file/01ecd39ca49ddecc5729ca996304781b-Paper-Conference.pdf); [Slides](https://nips.cc/media/neurips-2024/Slides/94608.pdf)
- **CPPF, "Combining Past, Present and Future: A Self-Supervised Approach for CIL"** (arXiv 2311.08764). Its Embedding Space Reserving module uses cluster centers as anchors, squeezes known-class embeddings away from a reserved group of centers with a margin loss, and uses a hyperparameter λ to split space between chosen and reserved regions. [V, search snippet] — [arXiv 2311.08764](https://arxiv.org/pdf/2311.08764)
- **NC-FSCIL** (Yang, Yuan, Li, Lin, Torr, Tao; ICLR 2023, notable top-25%). It pre-assigns a fixed simplex-ETF classifier that covers the whole label space (base plus all future sessions) and pulls features to their assigned prototypes. Space for future classes is therefore reserved geometrically from the start. Papers with Code lists average accuracy of 67.50 (CIFAR-100), 67.28 (CUB-200) and 67.82 (miniImageNet). [V] — [arXiv 2302.03004](https://export.arxiv.org/abs/2302.03004); [PwC](https://cs.paperswithcode.com/paper/neural-collapse-inspired-feature-classifier)
- **OrCo** (Ahmed, Kukleva, Schiele; CVPR 2024). Base-session pretraining uses supervised plus self-supervised contrastive losses and orthogonal pseudo-targets. The OrCo loss "maximizes margins and reserves space for the following incremental data". Reports SOTA on mini-ImageNet, CIFAR100 and CUB. [V] — [arXiv 2403.18550](https://arxiv.org/abs/2403.18550)
- **CwD, "Mimicking the Oracle: An Initial Phase Decorrelation Approach for CIL"** (Shi et al., CVPR 2022). This is a base-session regularizer that goes in the opposite direction (see Q3). Models trained on few classes show per-class covariance dominated by the top eigenvalues ("long and narrow" class regions). CwD makes each class scatter more uniformly (a flatter eigenspectrum) to mimic the joint/oracle model. Gain: about +1% to 3% on top of SOTA CIL methods. The supplement says why this helps "is still yet to be explored". [V] — [arXiv 2112.04731](https://arxiv.org/abs/2112.04731); [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Shi_Mimicking_the_Oracle_An_Initial_Phase_Decorrelation_Approach_for_Class_CVPR_2022_paper.html)
- Other base-session or forward-compatible FSCIL works, not fetched this session [M]:
  - CEC: Zhang et al., CVPR 2021, "Few-Shot Incremental Learning with Continually Evolved Classifiers", arXiv 2104.03047. Decouples the backbone from a graph-adapted classifier and uses pseudo-incremental tasks in the base session.
  - LIMIT: Zhou et al., TPAMI 2022, arXiv 2203.17030. Samples fake multi-phase tasks from base data.
  - ALICE: Peng et al., ECCV 2022, "Few-shot Class-Incremental Learning from an Open-Set Perspective", arXiv 2208.00147. Angular-penalty loss plus class/data augmentation in the base session to learn more transferable features.
  - SAVC: Song et al., CVPR 2023, "Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for FSCIL", arXiv 2304.00426. Creates virtual classes in the base session with contrastive learning.
  - "Forward Compatible Training for Large-Scale Embedding Retrieval Systems": Ramanujan et al., CVPR 2022, arXiv 2112.02805. Despite the name, this is about embedding backward/forward compatibility in retrieval, not CIL.

### Inferences
- The idea of "making base classes take less space so future classes fit" is anticipated in motivation by FACT (2022), PRL (2024), CPPF (2023) and OrCo (2024). PRL in particular is NECIL, base-phase squeezing, plug-in, and highly likely to be raised by reviewers.
- The operationalization is different. The existing works squeeze via prototype/margin/contrastive terms or fixed targets. None, as far as I found, uses a spectral or entropic measure of how many dimensions the base-class mixture occupies (S(rho_bar), effective rank, nuclear norm). That operationalization is the only defensible novelty.
- Note the sign conflict. FACT/PRL squeeze, while CwD flattens the per-class spectrum. Both report gains. The two are compatible only if you distinguish intra-class spread (CwD wants it isotropic) from the span of the whole base mixture (FACT/PRL want it compact). S(rho_bar) mixes the two (see Q2).

### Gaps
- The exact PES loss in PRL, its backbone (ResNet-18 from scratch vs. pretrained ViT), and its ablation gains. The PDF text could not be extracted. Must be read before claiming any difference.
- I did not check whether any pretrained-ViT/adapter CIL paper (L2P/DualPrompt/EASE/APER line) regularizes base-session rank or entropy. None surfaced in the searches.
- CEC/ALICE/SAVC/LIMIT reported numbers were not verified this session.

---

## Q2. Is von Neumann / matrix / spectral entropy, nuclear norm, effective rank, or coding rate used as a regularizer in representation learning and in continual learning? (MCR², ReduNet, i-CTRL)

### Takeaway
Yes, widely, but almost always to increase entropy or rank (anti-collapse, uniformity, transferability). VNE (CVPR'23) is the direct precedent for a von Neumann entropy regularizer on the feature autocorrelation matrix, and it supports both directions (increase or decrease). MCR² maximizes the coding rate of the whole set while compressing each class. i-CTRL notes that this "fills the feature space" and is therefore ill-suited to naive incremental learning. That observation is the closest prior statement of the motivation behind the proposed idea.

### Cited Findings
- **VNE** (Kim, Kang, Hwang, Shin, Rhee; CVPR 2023, pp. 3799–3810). Regularizes the von Neumann entropy of the representation autocorrelation matrix, i.e. the Shannon entropy of its normalized eigenvalues, bounded in [0, log d], added to the task loss with a coefficient at light overhead. Claims theoretical links to rank, disentanglement and isotropy. Applied to domain generalization, meta-learning, SSL and generative models. Code: github.com/jaeill/CVPR23-VNE. [V] — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_VNE_An_Effective_Method_for_Improving_Deep_Representation_by_Manipulating_CVPR_2023_paper.html); [arXiv 2304.01434](https://arxiv.org/abs/2304.01434)
  - [M, not verified this session] My recollection is that VNE uses both VNE+ (increase entropy) and VNE− (decrease entropy) depending on the task, which would mean decreasing feature entropy as a regularizer is itself already published. I also recall that it does not cover continual learning. Check the paper's tables.
  - Key overlap: the autocorrelation of normalized features (1/N) Σ z zᵀ is exactly rho_bar with empirical class priors p_c. So S(rho_bar) is VNE's quantity, computed per-batch over the base classes.
- **Matrix Information Theory for SSL / Matrix-SSL** (Zhang, Tan, Yang, Huang, Yuan; ICML 2024, PMLR 235:59897–59918). Defines a matrix entropy that generalizes von Neumann entropy beyond unit-trace PSD matrices, and a matrix KL tr(P log P − P log Q − P + Q). Builds on maximum-entropy encoding (unifying SimSiam, Barlow Twins, MEC) and adds covariance-matrix alignment. Reports gains on ImageNet linear evaluation and MS-COCO transfer. [V] — [PMLR](https://proceedings.mlr.press/v235/zhang24bi.html); [arXiv 2305.17326](https://arxiv.org/abs/2305.17326)
- **"Information Flow in SSL"** (Tan et al., ICML 2024) uses matrix mutual information and joint entropy to analyze Barlow Twins and spectral contrastive learning. [V, search snippet] — [summary page](https://liner.com/review/information-flow-in-selfsupervised-learning)
- **MCR²** (Yu, Chan, You, Song, Ma; NeurIPS 2020). The objective is ΔR = R(Z) − Σ_c (n_c/n) R(Z_c), with R(Z) = ½ log det(I + (d/nε²) Z Zᵀ). It has an expansion term (whole-set coding rate, maximized) and a compression term (per-class rate, minimized), giving "between-class discriminative and within-class diverse" representations. [V, abstract page; formula from M] — [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html)
- **"Incremental Learning via Rate Reduction"** (Wu et al., CVPR 2021, arXiv 2011.14593). Applies the rate-reduction/ReduNet construction to incremental learning. [V, existence; details not read] — [arXiv 2011.14593](https://arxiv.org/abs/2011.14593)
- **i-CTRL, "Incremental Learning of Structured Memory via Closed-Loop Transcription"** (Tong, Dai, Wu, Li, Yi, Ma; ICLR 2023).
  - States that maximizing rate reduction spreads features over the whole feature space, which makes plain MCR² unsuitable for naive incremental learning because the number of classes grows in a fixed feature space.
  - Uses a fixed-capacity encoder/decoder minimax (CTRL) with a constraint that keeps old-class memory consistent.
  - Generative-replay style. Beats prior generative replay on MNIST, CIFAR-10 and ImageNet-50.
  - [V] — [arXiv 2202.05411](https://arxiv.org/abs/2202.05411); [code](https://github.com/tsb0601/i-CTRL)
- **AdaGauss, "Task-recency bias strikes back"** (exemplar-free CIL, NeurIPS 2024). Applies an anti-collapse loss L_AC = −(1/S) Σ min(a_i, 1), where a_i are the Cholesky diagonals of the minibatch feature covariance, to the first task and all later tasks. This makes the covariance full-rank: on ImagenetSubset the eigenvalue ratio max/min falls from about 1.2×10⁵ to about 84, and rank = 64 = S. Replacing L_AC with shrinkage lowers CIFAR-100 (T=10) last accuracy from 46.1% to 40.2% and ImagenetSubset from 51.1% to 46.7%. Main backbone is ResNet18; there is a ViT-S appendix, but L_AC's effect on ViT is not isolated. [V] — [arXiv 2409.18265](https://arxiv.org/html/2409.18265v2)
- [M] Effective rank as an SSL quality metric: RankMe (Garrido et al., ICML 2023, arXiv 2210.02885) and α-ReQ (Agrawal et al., NeurIPS 2022). Higher effective rank correlates with better downstream transfer. VICReg/Barlow Twins (variance/covariance or redundancy-reduction terms) also explicitly push toward high rank. Matrix-based Rényi entropy: Sanchez Giraldo, Rao, Principe, IEEE TIT 2015.

### Inferences
- In quantum terms, S(rho_bar) = χ + Σ_c p_c S(rho_c), where χ is the Holevo quantity (an upper bound on class information extractable by any POVM). Penalizing S(rho_bar) shrinks both terms: the intra-class spread S(rho_c), and the inter-class distinguishability χ that classification needs. MCR² does the opposite of the first half (it maximizes total rate and minimizes per-class rate). An "entropy budget" that penalizes S(rho_bar) − λ·χ, i.e. effectively Σ p_c S(rho_c) plus a cap on S(rho_bar), would be closer to MCR² compression with a capped expansion term. This decomposition is my derivation, not from a source. It suggests the raw S(rho_bar) penalty fights discrimination unless it is paired with a CE/χ term.
- The novelty margin over VNE is thin. VNE already regularizes the von Neumann entropy of the autocorrelation of features. The proposed changes are: (a) the class-mixture interpretation with priors p_c, (b) applying it only at task 0, (c) motivating it as reserving room for future classes in EFCIL with a PTM and adapter. That is an application novelty, not a methodological one.

### Gaps
- Not verified: whether VNE explicitly reports a decreasing-entropy variant and on which tasks.
- No paper found that uses von Neumann, matrix or spectral entropy (as opposed to anti-collapse or rate terms) as a base-session regularizer in CIL. This is absence of evidence from about 12 searches, not proof.
- Details of Wu et al. 2021 (Incremental Learning via Rate Reduction) not read.

---

## Q3. Does lower or higher rank of base representations help or hurt later incremental tasks? (plasticity loss, rank collapse)

### Takeaway
The dominant evidence favors higher-rank or more uniform base features for later tasks.
- CwD: a flatter intra-class spectrum at the initial phase gives +1–3%.
- AdaGauss: forcing full rank gives about +4–6 points in EFCIL.
- SSL rank metrics and the plasticity literature: rank collapse correlates with poor transfer and plasticity.

Squeeze-based methods (FACT, PRL, OrCo) also report gains, but they compress inter-class placement or margins, not the dimensionality of the feature space. A global entropy or rank penalty directly contradicts CwD and AdaGauss evidence and risks the very collapse those papers fix.

### Cited Findings
- CwD (CVPR 2022): with few classes, per-class top eigenvalues dominate. The oracle (all classes) has a flatter spectrum. Regularizing toward uniform scatter at the initial phase gives +1–3% on top of SOTA CIL. [V] — [arXiv 2112.04731](https://arxiv.org/abs/2112.04731)
- AdaGauss (NeurIPS 2024): feature-extractor dimensionality collapse causes task-recency bias in EFCIL, and anti-collapse (full rank) gives a 4.4–5.9 point gain in last accuracy. [V] — [arXiv 2409.18265](https://arxiv.org/html/2409.18265v2)
- Dohare et al., "Loss of plasticity in deep continual learning", Nature 2024 (DOI 10.1038/s41586-024-07711-7). Standard deep learning gradually loses plasticity in continual settings, down to shallow-network level. The cause is linked to dead units, growing weights and loss of unit diversity. Continual backpropagation (re-initializing a small fraction of less-used units) maintains plasticity. [V via secondary pages; the paper's effective-rank analysis was not verified this session] — [Amii summary](https://www.amii.ca/research/loss-of-plasticity-in-deep-continual-learning); [arXiv 2306.13812](https://arxiv.org/abs/2306.13812)
  - [M] My recollection is that the Nature paper reports effective rank of the representation decreasing alongside plasticity loss. Verify before citing.
- Lyle et al., "Understanding Plasticity in Neural Networks" (ICML 2023). Plasticity loss is "deeply connected to changes in the curvature of the loss landscape" and often occurs without saturated units. Feature rank is only sometimes correlated with plasticity, so it is not a reliable single explanation. [V] — [PMLR](https://proceedings.mlr.press/v202/lyle23b.html)
- [M] Lyle, Rowland, Dabney, "Understanding and Preventing Capacity Loss in RL" (ICLR 2022, arXiv 2204.09560): feature rank decline correlates with capacity loss, and the InFeR regularizer mitigates it. Kumar et al., "Implicit Under-Parameterization" (ICLR 2021, arXiv 2010.14498): effective-rank collapse of features in TD learning hurts performance.
- 2025–26 preprints:
  - "Why Do Neural Networks Forget" (arXiv 2603.04580) frames forgetting as representational collapse measured by eRank. [V, snippet; preprint]
  - "Heads collapse, features stay" (arXiv 2512.07400, reported as ICLR 2026) finds that small replay buffers give rank-deficient covariances yet keep feature separability, so the main problem is the head/classifier. [V, snippet]
  - These conflict on how much collapse matters. — [arXiv 2603.04580](https://arxiv.org/html/2603.04580v1); [arXiv 2512.07400](https://arxiv.org/html/2512.07400v1)

### Inferences
- The plasticity literature concerns long sequences with full-network training. With a frozen pretrained ViT and only an adapter trained at task 0, the backbone keeps its rank. The penalty would only shape the adapter output, and EFCIL methods with frozen PTMs often freeze the adapter after task 0 (APER-style). In that regime the relevant question is not plasticity but whether the frozen feature space represents novel classes well. CwD/AdaGauss/RankMe evidence says that compressing it hurts linear and prototype separability of unseen classes.
- A space-reservation story is more consistent with the evidence if the penalty targets only the base-class mean or prototype subspace (inter-class span), or only the adapter's residual update, and leaves the pretrained feature spectrum intact. A global S(rho_bar) penalty does neither.
- Practical risk for prototype/NCM or Gaussian (FeCAM/AdaGauss-style) classifiers: a lower-rank base covariance gives ill-conditioned covariances, which is exactly the failure AdaGauss documents.

### Gaps
- No direct experiment found that lowers base-session feature rank on purpose and measures later-session accuracy with a pretrained ViT. This is the empirical gap any submission would need to fill. Expect a negative or neutral result unless it is designed carefully.
- Dohare et al.'s exact rank findings not verified.

---

## Q4. Verdict

### Takeaway
**Partially anticipated, with a weak novelty margin.**
- The motivation (squeeze base-class embeddings to reserve room for future classes) is anticipated by FACT (CVPR'22), CPPF (2023), OrCo (CVPR'24) and especially PRL (NeurIPS'24, non-exemplar CIL, base-phase "Preemptive Embedding Squeezing").
- The regularizer (von Neumann entropy of the feature autocorrelation/density matrix) is anticipated by VNE (CVPR'23). Related matrix-entropy and coding-rate objectives exist in Matrix-SSL (ICML'24) and MCR²/i-CTRL (NeurIPS'20 / ICLR'23). i-CTRL explicitly argues that space-filling objectives are unsuitable for incremental learning.
- What remains new: the specific combination "penalize S(Σ_c p_c rho_c) only at task 0 of EFCIL with a pretrained ViT plus adapter". The quantum-information framing (Holevo decomposition) is a presentational novelty.
- Direction risk: the strongest CIL evidence (CwD CVPR'22, AdaGauss NeurIPS'24) and the SSL and plasticity literature (RankMe, Lyle, Dohare) point the other way. Higher or more uniform rank at the base session helps later tasks.

### Cited Findings
- Closest anticipations:
  - PRL — [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/94608)
  - FACT — [arXiv 2203.06953](https://arxiv.org/abs/2203.06953)
  - VNE — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_VNE_An_Effective_Method_for_Improving_Deep_Representation_by_Manipulating_CVPR_2023_paper.html)
  - i-CTRL — [arXiv 2202.05411](https://arxiv.org/abs/2202.05411)
  - OrCo — [arXiv 2403.18550](https://arxiv.org/abs/2403.18550)
  - NC-FSCIL — [arXiv 2302.03004](https://export.arxiv.org/abs/2302.03004)
- Contradicting evidence:
  - CwD — [arXiv 2112.04731](https://arxiv.org/abs/2112.04731)
  - AdaGauss — [arXiv 2409.18265](https://arxiv.org/html/2409.18265v2)
  - Lyle 2023 — [PMLR](https://proceedings.mlr.press/v202/lyle23b.html)

### Inferences
- If the idea is pursued, it must be positioned against PRL and VNE explicitly. It should also be reformulated so it does not conflict with CwD/AdaGauss: for example, penalize only the span of the class means (the Holevo-like inter-class part, capped) or the adapter residual. Avoid the full mixture entropy, and keep each class's covariance well-conditioned.
- A cheap diagnostic before any claim: measure the effective rank / S(rho_bar) of frozen ViT features on base vs. future classes. If future-class means already lie largely outside the base span (likely, given that pretrained ViTs are high-rank), there is little to "reserve".

### Gaps
- PRL's loss and numbers, VNE's decreasing variant, and the Dohare rank analysis remain unverified and should be read before final positioning.
