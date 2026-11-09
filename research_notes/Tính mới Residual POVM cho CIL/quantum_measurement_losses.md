# Quantum measurement formalism (POVM / PGM / Helstrom / Born rule) as a classifier or training objective, and its use in continual learning: novelty check for "Residual-POVM training"

Proposed method under review: exemplar-free CIL with a pre-trained ViT + adapter. Old classes are stored as density matrices rho_c built from feature mean + covariance. Old-class POVM elements E_c are fixed, PGM-style. New-class elements live inside the residual R = I - sum_old E_c, as E_j = R^{1/2} M_j R^{1/2} with sum_j M_j = I, so completeness holds by construction. Training loss is -log Tr(E_y rho_x), with rho_x = |psi(f(x))><psi(f(x))| (amplitude-encoded feature).

Search coverage: about 16 search/fetch calls on arXiv, NeurIPS proceedings, ACL Anthology mirrors, Springer, and university repositories. Items marked **[UNVERIFIED]** come from background knowledge and were not confirmed in this session.

## Q1. Which works use POVM / Born-rule probabilities Tr(E rho) as a classifier or differentiable loss for (deep) learning models?

### Takeaway
There is a substantial literature on Born-rule and density-matrix classifiers. Three groups stand out: (a) Helstrom and PGM classifiers from the Cagliari group (Sergioli/Giuntini) and the Brussels group (Cruzeiro et al.); (b) trainable density-matrix layers from the Bogotá group (González et al.); (c) quantum-inspired NLP (CNM, NNQLM, Born-rule text classification). Each of these fixes the POVM in closed form from the data (PGM/Helstrom) or learns states/projectors freely. None builds a POVM incrementally inside a residual operator.

### Cited Findings
- **Sergioli, Giuntini, Freytes, "A new quantum approach to binary classification", PLOS ONE 2019 (e0216224).** Introduces the Helstrom Quantum Centroid (HQC) classifier. It is binary and inspired by the Helstrom measurement on density-matrix-encoded data (class centroids as density matrices), and reportedly beat 12 standard classifiers on 14 datasets. Performance improves with more copies of the state. — [IDEAS/RePEc record](https://ideas.repec.org/a/plo/pone00/0216224.html); [UNICA IRIS](https://iris.unica.it/handle/11584/270628); [HQC package (amplitude or stereographic encoding)](https://pypi.org/project/HQC/)
- **"Pretty Good Measurement for Radiomics", arXiv 2603.00223, Int. J. Theor. Phys. 65:161 (2026), Sergioli/Giuntini group.** Multi-class supervised classification with the PGM as an "operator-valued decision rule derived from quantum state discrimination". Each class is an encoded mixed state, and a single POVM handles all classes (no one-vs-rest). The PGM is built in closed form, not trained by gradient. — [arXiv 2603.00223](https://arxiv.org/abs/2603.00223); [HTML](https://arxiv.org/html/2603.00223v1)
- **"Multi-class classification based on quantum state discrimination" (2023)**, the earlier general framework from the same line, with classification functions interpretable as fuzzy sets. — [SEDICI record](https://repositoriosdigitales.mincyt.gob.ar/vufind/Record/SEDICI_13abe22ed2d2671533912195baaf9039)
- **QSMOTE-PGM / kPGM for imbalanced data, arXiv 2512.16960** builds on the kernelized PGM. — [arXiv 2512.16960](https://arxiv.org/abs/2512.16960v1)
- **Zambrini Cruzeiro, De Mol, Massar, Pironio, "Quantum-inspired classification based on quantum state discrimination", arXiv 2303.15353 (Mar 2023, preprint).** Classical quantum-inspired classifiers based on state discrimination, trained by SDP or an LP relaxation. A PGM classifier, kernelized via Gram matrices of dot products (kPGM-C), "generally outperforms the other quantum-inspired classifiers" on MNIST and MNIST-1D. — [arXiv abs](https://arxiv.org/abs/2303.15353); [PDF](https://arxiv.org/pdf/2303.15353)
- **Lloyd, Schuld, Ijaz, Izaac, Killoran, "Quantum embeddings for machine learning", arXiv 2001.03622 (2020).** Trains the *embedding* (a quantum feature map) so that class states separate in Hilbert space ("quantum metric learning"). The measurement is then fixed: the Helstrom measurement is optimal for trace (l1) distance, and the fidelity/overlap measurement for Hilbert-Schmidt distance. Fidelity-classifier empirical risk = -HS distance between class-averaged states. This is conceptually the closest "train features for a measurement" precedent, but it is binary, a quantum circuit, and not incremental. — [arXiv 2001.03622](https://arxiv.org/abs/2001.03622)
- **González et al., "Learning with density matrices and random features", Quantum Machine Intelligence 2022 (arXiv 2102.04394).** Density matrices for density estimation, classification, and regression. The "quantum measurement classification" (QMC) model represents the joint input-output distribution as a density matrix and is differentiable/trainable. — [Springer](https://link.springer.com/article/10.1007/s42484-022-00079-9); [arXiv](https://arxiv.org/abs/2102.04394v3)
- **"Quantum Measurement Classification with Qudits", arXiv 2107.09781 (DMKDE/DMKDC).** Density-matrix kernel density classification, with density matrices that can be trained by SGD. — [arXiv PDF](https://arxiv.org/pdf/2107.09781)
- **González, Ramos-Pollán, Gallego-Mejia, "Kernel Density Matrices for Probabilistic Deep Learning", arXiv 2305.18204; Quantum Machine Intelligence 7(2):94 (2025)** (earlier title reportedly "Quantum Kernel Mixtures for Probabilistic Deep Learning"). Density matrices in an RKHS used as differentiable deep-learning components for discriminative and generative modelling. — [arXiv](https://arxiv.org/abs/2305.18204v2); [Drexel record](https://researchdiscovery.drexel.edu/esploro/outputs/journalArticle/Kernel-density-matrices-for-probabilistic-deep/991022197436104721)
- **Guidotti & Ferrara, "Text Classification with Born's Rule", NeurIPS 2022 (pp. 30990–31001).** Text is treated as a superposition of words, and classification uses the Born rule for the probability that a document "collapses" into a class. There is an explicit version and a version embedded in a neural network (package `bornrule`). — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c88d0c9bea6230b518ce71268c8e49e0-Abstract.html)
- **Li, Wang, Melucci, "CNM: An Interpretable Complex-valued Network for Matching", NAACL 2019, pp. 4139–4148 (Best Explainable NLP Paper).** Words are pure states and sentences are mixed states (density matrices) in a complex space. Matching uses trainable measurement projectors whose outcome probabilities feed the network, trained end to end. — [ACL Anthology N19-1420](https://preview.aclanthology.org/menus/N19-1420/); [arXiv 1904.05298](https://ar5iv.labs.arxiv.org/html/1904.05298)
- NNQLM (Zhang et al. 2018a) embeds words as unit vectors and sentences as real density matrices, as cited inside CNM. — [CNM arXiv](https://ar5iv.labs.arxiv.org/html/1904.05298). **[UNVERIFIED venue]** Commonly cited as Zhang, Niu, Su, Wang, Ma, Song, "End-to-End Quantum-like Language Models with Application to Question Answering", AAAI 2018. This search did not confirm the venue.
- Garvin, Kondratyev, Lipton, Paini, "Density Matrix Classifier" (SSRN 2024): each class is a density matrix via a quantum feature map, and the nearest class is chosen by Frobenius distance. — [SSRN 5014387](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5014387)
- LegoQ (arXiv 2607.28970) learns density-matrix (matrix-state) representations for hyperspectral classification, with density-matrix prototypes scored by fidelity. — [arXiv PDF](https://arxiv.org/pdf/2607.28970)

### Inferences
- **"Born-rule probability as a classifier output" is clearly anticipated.** It appears in HQC, PGM classifiers, QMC/KDM, CNM, and Born-rule text classification. Training a deep network end to end through Tr(E rho) also has precedent (CNM, KDM/QMC, Lloyd et al.'s trained embeddings). So "-log Tr(E_y rho_x) as the loss" cannot be claimed as novel on its own.
- **"PGM from class density matrices" is also anticipated** (Sergioli/Giuntini 2026; Cruzeiro et al. 2023). In those works the PGM is computed once from training data. It is not a fixed sub-POVM for old classes with trainable elements for new classes.
- The specific combination is not found in any of these: PGM elements for old classes, a trainable sub-POVM confined to the residual I - sum E_old, and a feature extractor trained against that partial POVM.

### Gaps
- I did not read full texts for CNM, QMC, or KDM to confirm the exact loss form (log of Born probability vs. cross-entropy over softmax of Born probabilities). Prior work using a Born probability inside cross-entropy is likely enough that the loss alone should not be a novelty claim.
- The Quantum Probability Driven Network (QPDN) and other quantum-inspired NLP models (QMSA, QMN/QMNN for conversational emotion, and others) were not separately verified in this session.

## Q2. Is PGM, the Helstrom bound, fidelity, or trace distance used for continual / incremental / few-shot learning or to prevent class overlap?

### Takeaway
Two quantum continual-learning lines exist, plus one very recent quantum class-incremental paper (Aug 2026). None uses a PGM/Helstrom POVM or completeness to allocate measurement space to new classes. The closest is "Quantum Incremental Learning with Mixed State Prototypes", which uses mixed-state class prototypes, Hilbert-Schmidt distance via SWAP test, and exemplar replay.

### Cited Findings
- **Wu, Zhou, Deng, Jiang, Cheong, Pedrycz, "Quantum Incremental Learning with Mixed State Prototypes", arXiv 2608.10464 (11 Aug 2026, cs.AI).** The closest work found.
  - Each class has a rank-K mixed-state prototype rho_c = sum_i w_{c,i} U_c|i><i|U_c^† (4 qubits, K=7, softmax weights).
  - Classification uses squared Hilbert-Schmidt distance ||rho(x) - rho_c||_F^2, with overlaps estimated by SWAP test. Logit_c = O_c(x) - 1/2 sum w^2.
  - **No POVM and no completeness constraint.** No PGM, no Helstrom, and no residual subspace. Fidelity appears only as a baseline.
  - **Uses exemplar replay** (640-sample memory).
  - Backbone: ResNet-18 → linear → amplitude encoding into 8 qubits → QCNN. Not a pre-trained ViT with an adapter.
  - It notes that basis-readout quantum classifiers need expanded measurements or readout rules as the label space grows. That is the same problem the residual-POVM idea addresses, but the paper solves it with prototypes rather than POVM growth.
  - — [arXiv HTML](https://arxiv.org/html/2608.10464); [PDF](https://arxiv.org/pdf/2608.10464)
- **Jiang, Lu, Deng, "Quantum Continual Learning Overcoming Catastrophic Forgetting", Chinese Physics Letters 39(5):050303 (2022), arXiv 2108.02786.** Shows catastrophic forgetting in quantum classifiers and mitigates it with EWC. It is regularization-based, with no measurement-space allocation. — [arXiv](https://arxiv.org/pdf/2108.02786); [CPL](https://cpl.iphy.ac.cn/en/article/pdf/preview/10.1088/0256-307X/39/5/050303.pdf)
- **"Rethinking Quantum Continual Learning with Quantum Fisher Information", arXiv 2607.16030 (2026).** QEWC weights parameters by QFI, motivated by the bound CFI(POVM) ≤ QFI for any fixed POVM. POVMs appear only in the Fisher-information argument, not as a classifier structure. — [arXiv PDF](https://arxiv.org/pdf/2607.16030)
- "Quantum continual learning of quantum data realizing knowledge..." (arXiv 2203.14032): a quantum-data continual-learning paper. Details not read. — [arXiv PDF](https://arxiv.org/pdf/2203.14032)
- Experimental quantum continual learning on superconducting qubits was reported (ResearchGate listing; details not verified). — [ResearchGate](https://www.researchgate.net/publication/399507880_Experimental_demonstration_of_quantum_continual_learning_with_superconducting_qubits)
- **Li, Xiao, Cao, Wu, Ji, "Quantum-Gated Task-interaction Knowledge Distillation for Pre-trained Model-based Class-Incremental Learning" (QKD), arXiv 2604.11112, listed as accepted to CVPR 2026.** This is pre-trained-model CIL with adapters, the same setting as the proposed method. The abstract mentions only "quantum gating" for task-relevance weighting and distillation between adapters. It does not mention POVMs, density matrices, or the Born rule. — [arXiv abs](https://arxiv.org/abs/2604.11112)
- Few-shot: no density-matrix/POVM few-shot method was found. The nearest hits are LegoQ (fidelity prototypes, not few-shot) and quantum-enhanced prototypical networks for malware (Sci. Rep. 2026), which do not use measurement formalism. — [LegoQ](https://arxiv.org/pdf/2607.28970); [Sci. Rep.](https://www.nature.com/articles/s41598-026-45738-0)

### Inferences
- **The application of PGM/Helstrom/completeness to class-incremental learning appears unoccupied.** The quantum-CL papers found use EWC/QFI regularization (2108.02786, 2607.16030), HS-distance prototypes plus replay (2608.10464), or quantum-named gating (2604.11112).
- **2608.10464 is the must-cite, must-compare work.** It is "quantum CIL with density-matrix class prototypes". The proposed method can be framed as differing on four axes: (i) POVM with completeness vs. HS-distance logits; (ii) exemplar-free (stats only) vs. replay; (iii) covariance-based rho_c built from stored statistics vs. circuit-learned prototypes; (iv) a pre-trained ViT with an adapter on classical hardware vs. a QCNN.
- **2604.11112 (QKD, CVPR 2026) must be cited and differentiated** because it shares the PTM-CIL setting and "quantum" branding. Reviewers will ask about it.

### Gaps
- I did not read the full text of 2604.11112 to rule out density-matrix or measurement components in its gating. Its exemplar use is also unconfirmed.
- I did not read the last ~15% of 2608.10464 (appendix/related work). It may cite further quantum-CIL works.

## Q3. Does any work grow a POVM incrementally (adding elements in the residual I - sum E) or use Naimark dilation to add classes?

### Takeaway
No learning paper was found that grows a POVM by placing new elements in the residual I - sum E_old, or uses Naimark dilation to add classes. The physics literature has iterative Naimark-extension constructions, but these build a dilation of a given POVM, not new outcomes.

### Cited Findings
- Dalla Pozza and Paris describe an iterative Naimark-extension construction: effects are placed in a block of a larger matrix, orthogonality to previously built projectors is imposed, and positivity and idempotence are enforced. It is a dilation of a fixed POVM, not the addition of outcomes for new classes. — [Emergent Mind summary of Naimark measurement](https://www.emergentmind.com/topics/naimark-quantum-measurement) (secondary source; primary paper not fetched)
- Canonical vs. minimal Naimark dilations and the joint measurability of POVMs are covered. — [arXiv 1404.1477](https://arxiv.org/pdf/1404.1477); [arXiv 2112.06784](https://arxiv.org/pdf/2112.06784)
- PAC-learnability of POVM hypothesis classes (arXiv 2308.12304) is theory only, with no incremental growth. — [arXiv 2308.12304](https://arxiv.org/pdf/2308.12304)
- Classical analog (not quantum): Adam-NSCL (Wang, Li, Sun, Xu, CVPR 2021) trains in the null space of the uncentered feature covariance of previous tasks (via SVD), with no stored data. — [mlanthology](https://mlanthology.org/cvpr/2021/wang2021cvpr-training); [arXiv 2103.07113](https://www.arxiv.org/abs/2103.07113)

### Inferences
- **The residual parameterization appears novel as a learning mechanism.** It is E_j = R^{1/2} M_j R^{1/2}, with fixed old-class effects and a learned sub-POVM inside R so that completeness holds by construction. Mathematically it is a standard result: any POVM on the support of R composed with R^{1/2} gives effects summing to R. So the novelty is in its use for CIL, not in the operator algebra.
- **Reviewers will see the conceptual link to null-space / orthogonal-subspace CL methods** (Adam-NSCL, GPM [UNVERIFIED here: Saha, Garg, Roy, "Gradient Projection Memory for Continual Learning", ICLR 2021]). "New classes take the unclaimed subspace" is close in spirit to "new tasks update in the null space of old feature covariance". The difference is that those methods project *gradients/weights*, while the proposed method partitions the *output measurement* (class-probability mass) via completeness. That distinction should be made explicitly.
- **Note: R = I - sum E_old is generally full rank, not a projector** (PGM effects are not projectors). So the "unclaimed subspace" is soft. New-class mass in directions where old effects are near 1 is suppressed by R^{1/2}, not forbidden. Claims should be phrased as "soft residual", not as an orthogonal complement.

### Gaps
- I did not search Google Scholar or Semantic Scholar directly (no access). A Scholar search for "sequential POVM", "incremental quantum measurement learning", and "open-set POVM / reject element" is recommended before submission. In particular, a "null/reject" POVM element E_? = I - sum E_c for open-set recognition may exist in quantum-inspired work.

## Q4. Is there a quantum continual-learning paper that uses measurement-based class allocation?

### Takeaway
None found. The quantum CL/CIL papers identified (2108.02786, 2203.14032, 2607.16030, 2608.10464, 2604.11112) use EWC/QFI regularization, prototypes plus replay, or gating. None allocates measurement (POVM) space to classes.

### Cited Findings
- 2608.10464 uses distance-to-prototype logits with an auxiliary MLP head on diagonal measurement probabilities p_k = Tr[rho(x)|k><k|]. That is a fixed computational-basis readout, not a class-allocating POVM. — [arXiv HTML](https://arxiv.org/html/2608.10464)
- 2607.16030 uses POVMs only to motivate QFI over CFI. — [arXiv PDF](https://arxiv.org/pdf/2607.16030)
- 2108.02786 uses EWC on quantum classifiers. — [arXiv](https://arxiv.org/pdf/2108.02786)

### Inferences
- The "measurement-based class allocation" angle is open as far as this search reached.

### Gaps
- The search did not cover Chinese-language venues or workshop papers (e.g., QTML, IEEE QCE) exhaustively.

## Q5. How does each close work differ from Residual-POVM training? Overall verdict

### Takeaway
**Verdict: PARTIALLY ANTICIPATED (individual components exist), but the combination is NOVEL as far as this search reached.** The residual-constrained, completeness-by-construction POVM for exemplar-free CIL, built from stored mean/covariance states and used to train a deep feature extractor, was not found in any prior work.

### Comparison table (from the findings above)

| Work | Born/POVM classifier | Training-time loss through Tr(E rho)? | Incremental / CL? | Exemplar-free? | Uses covariance for rho_c? | Completeness by construction? |
|---|---|---|---|---|---|---|
| Sergioli, Giuntini, Freytes, PLOS ONE 2019 (HQC) | Helstrom, binary | No (closed-form) | No | n/a | Centroid density matrix | Yes (2-outcome Helstrom) |
| PGM for Radiomics, IJTP 2026 / arXiv 2603.00223 | PGM, multi-class | No (closed-form) | No | n/a | Class mixed states (averages of encoded data) | Yes (PGM sums to I on support) |
| Cruzeiro et al., arXiv 2303.15353 | PGM/kPGM + SDP/LP classifiers | SDP/LP, not deep backprop | No | n/a | Gram/kernel | Yes for PGM |
| Lloyd et al., arXiv 2001.03622 | Helstrom / fidelity | Yes, trains the embedding | No | n/a | Class-averaged states | Fixed binary measurement |
| González et al. QMC (QMI 2022) / KDM (QMI 2025) | Density-matrix measurement | Yes, differentiable | No | n/a | Learned density matrices | Not the focus |
| CNM (NAACL 2019), NNQLM | Trainable measurement projectors | Yes, end to end | No | n/a | Sentence density matrices | Not enforced as class POVM |
| Guidotti & Ferrara, NeurIPS 2022 | Born rule | Yes (NN variant) | No | n/a | No | No |
| Wu et al., arXiv 2608.10464 | HS-distance to mixed prototypes (SWAP) | Yes | **Yes (CIL)** | **No (replay)** | No (circuit-learned) | **No** |
| Li et al., QKD, arXiv 2604.11112 (CVPR 2026) | "Quantum gating" only | n/a | **Yes (PTM-CIL)** | Unclear | No | No |
| Jiang et al., CPL 2022; QEWC arXiv 2607.16030 | Quantum classifier + EWC/QFI | n/a | Yes (CL) | Regularization | No | No |
| Adam-NSCL, CVPR 2021 (classical) | No | n/a | Yes | Yes | **Yes (feature covariance)** | n/a (null-space gradient projection) |

### Inferences
- **Not novel in isolation:**
  - Born-rule/POVM probabilities as a classifier (HQC, PGM, Born-rule text, CNM).
  - Backpropagating through Tr(E rho) (CNM, QMC/KDM, Lloyd et al.).
  - PGM built from class density matrices (Sergioli/Giuntini; Cruzeiro et al.).
  - Quantum-flavored CIL in general (2608.10464; 2604.11112).
  - Using old-feature covariance to protect old classes without exemplars (Adam-NSCL; FeCAM [UNVERIFIED here: Goswami et al., "FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning", NeurIPS 2023]).
- **Apparently novel:**
  1. Freezing old-class POVM effects (PGM from stored mean+covariance states) and parameterizing new-class effects as R^{1/2} M_j R^{1/2} inside the residual, so completeness holds by construction across tasks.
  2. Using -log Tr(E_y rho_x) under that partial POVM as a training objective. The loss automatically penalizes mass in old-class effects, a measurement-theoretic replacement for exemplar replay or logit-bias correction.
  3. Doing this exemplar-free on a pre-trained ViT + adapter.
- **Recommended positioning:**
  - Cite 2608.10464 as the nearest quantum-CIL work. It differs by using HS distance with no POVM/completeness and needing replay.
  - Cite 2604.11112 as the PTM-CIL work with quantum branding but no measurement formalism.
  - Cite Sergioli/Giuntini (HQC 2019, PGM 2026) and Cruzeiro 2023 as the PGM-classifier lineage. Their PGM is static and non-incremental.
  - Cite Lloyd et al. 2020 as "train features for a fixed optimal measurement".
  - Cite CNM / Born-rule text / QMC/KDM for differentiable Born-rule losses.
  - Cite Adam-NSCL (and GPM) as classical "residual subspace" analogs, and explain that the proposed method constrains the output measurement rather than gradients.
- **Risk:** a reviewer may argue the method reduces to a constrained classical classifier. With rank-one rho_x, Tr(E rho_x) = psi^T E psi is a quadratic form. A PGM with a residual is then a particular normalized quadratic discriminant, close to Mahalanobis/FeCAM-style scoring. The novelty argument should rest on the completeness/residual allocation mechanism and on empirical gains, not on the quantum vocabulary.

### Gaps
- Not verified this session: GPM (ICLR 2021), FeCAM (NeurIPS 2023), NNQLM (AAAI 2018), and the primary Hausladen & Wootters (1994) PGM reference.
- No Google Scholar/Semantic Scholar/OpenReview sweep was possible. A final check on OpenReview (ICLR 2025–2026) for "POVM" + "continual" is advisable.
