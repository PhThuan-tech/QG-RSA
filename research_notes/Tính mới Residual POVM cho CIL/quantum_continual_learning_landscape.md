# Quantum and Quantum-Inspired Continual / Class-Incremental Learning: Landscape and Reviewer Expectations

Scope: 2019 to 2026. Research done 2026-10-30 with web search and arXiv abstract/HTML pages. Items marked **[UNVERIFIED]** come from memory or from secondary snippets and were not checked against a primary page in this session.

## Q1. Which papers address quantum continual learning, catastrophic forgetting in VQCs/QNNs, or hybrid quantum-classical CL? Datasets, scale, baselines, claimed results

### Takeaway
Quantum continual learning (QCL) is a small, physics-venue literature. It uses toy scales: a handful of qubits, MNIST or permuted MNIST, binary or few-task streams, and quantum-state data. Its baselines are naive sequential training plus EWC or GEM, ported from 2017-era classical CL. None of these papers is evaluated on ImageNet-R, ImageNet-A, CIFAR-100 B0Inc10 or CUB against PTM-based CIL methods. I found no paper that uses quantum kernels, PGM or density-matrix class states for continual learning.

### Cited Findings
- **Jiang, Lu, Deng (Tsinghua), "Quantum Continual Learning Overcoming Catastrophic Forgetting"**, arXiv:2108.02786 (Aug 2021), published in *Chinese Physics Letters* 39(5):050303 (2022).
  - Shows that variational quantum classifiers suffer catastrophic forgetting, using a fixed VQC trained sequentially on MNIST and then permuted MNIST.
  - Applies EWC, using local Fisher/geometric information, to mitigate the forgetting.
  - Simulations ran in Yao.jl.
  - Claim: forgetting "can be overcome in certain situations."
  - Sources: [arXiv](https://arxiv.org/abs/2108.02786v1); [CPL](https://cpl.iphy.ac.cn/article/doi/10.1088/0256-307X/39/5/050303); [CPL supplement](https://cpl.iphy.ac.cn/fileup/0256-307X/PDF/0256-307X-2022-39-5-050303-Suppl.pdf)
- **Situ, Lu, Pan, Li**, first titled "Continual learning of quantum state classification with gradient episodic memory" (arXiv:2203.14032, Mar 2022), later retitled "Quantum continual learning of quantum data realizing knowledge backward transfer". Published in *Physica A* vol. 620 (2023).
  - A VQC trained with GEM on quantum-state classification tasks.
  - Simulations show GEM beats EWC and gives positive backward transfer.
  - Sources: [arXiv](https://arxiv.org/abs/2203.14032v2); [RePEc/Physica A](https://ideas.repec.org/a/eee/phsmap/v620y2023ics0378437123003345.html)
- **Zhang, Lu, ..., Deng, Song (Zhejiang/Tsinghua), "Quantum continual learning on a programmable superconducting processor"**, arXiv:2409.09729 (Sep 2024). The arXiv journal reference is *npj Quantum Information* (2026).
  - Hardware experiment: three sequential tasks, two real-image recognition tasks and one quantum-state classification task.
  - With EWC, average accuracy exceeds 92.3%.
  - On quantum-engineered data, the quantum classifier beats a classical feedforward net with a comparable number of parameters.
  - Source: [arXiv](https://arxiv.org/abs/2409.09729)
- **Jain, "CQural: A Novel CNN based Hybrid Architecture for Quantum Continual Machine Learning"**, arXiv:2305.09738 (2023).
  - A quantum layer on top of a CNN, tested on MNIST and CIFAR-10 subsets.
  - Claims the hybrid forgets less than the classical baseline.
  - No peer-reviewed venue verified.
  - Sources: [arXiv](https://arxiv.org/abs/2305.09738); [ADS](https://ui.adsabs.harvard.edu/abs/2023arXiv230509738J/abstract)
- **Chen & Zhang, "Intrinsic preservation of plasticity in continual quantum learning"**, arXiv:2511.17228 (Nov 2025; v2 Jun 2026). Preprint only.
  - Claims QNNs keep plasticity over long continual-learning horizons, in supervised and RL settings, while MLPs lose it.
  - Attributes this to unitary/compact-manifold constraints that stop weights and gradients from growing.
  - The abstract gives no numbers. Search snippets mention permuted MNIST.
  - Source: [arXiv](https://arxiv.org/abs/2511.17228)
- **Hsu, Lin, Li, Chen, Kuo, "Rethinking Quantum Continual Learning with Quantum Fisher Information"**, arXiv:2607.16030 (Jul 2026). Preprint.
  - Proposes QEWC, which replaces classical Fisher information with quantum Fisher information in EWC.
  - Tested on sequential binary image and quantum-phase classification.
  - Claims better noise robustness under depolarizing noise.
  - Baselines are only naive sequential training and CFI-EWC. The abstract gives no numbers.
  - Source: [arXiv](https://arxiv.org/abs/2607.16030)
- **Zhu & Li, "QCL-IDS: Quantum Continual Learning for Intrusion Detection with Fidelity-Anchored Stability and Generative Replay"**, arXiv:2601.21318 (Jan 2026). Preprint.
  - Uses a fidelity-based anchor term that constrains decision changes on past data, which is conceptually close to a "fidelity to old class states" regularizer.
  - Datasets: UNSW-NB15 and CICIDS2017, in a three-stage stream.
  - Reports mean attack-F1 of 0.941 / 0.944 with forgetting 0.005 / 0.004. Sequential fine-tuning gets F1 of about 0.80 with forgetting of about 0.13.
  - Sequential fine-tuning is the only baseline named.
  - Source: [arXiv](https://arxiv.org/abs/2601.21318)
- Search for "quantum kernel" plus continual/incremental learning returned no dedicated paper.
  - The closest work is the VQC/EWC line above and the QKD gating paper (see Q2).
  - Source: search results summarised at [arXiv 2108.02786](https://arxiv.org/abs/2108.02786v1) and [emergentmind 2604.11112](https://www.emergentmind.com/papers/2604.11112)

### Inferences
- All native-QCL papers use toy scales and weak CL baselines: EWC, GEM, fine-tuning. None compares against replay-free PTM methods such as SimpleCIL/APER, EASE or MoS.
- A paper that plugs PKQ, density-matrix class states or a PGM head into a ViT+adapter CIL pipeline would therefore be the first of its kind at realistic CIL scale.
- The QCL physics community and the CIL/CV community barely overlap. Related-work sections need both.
- The fidelity-anchor idea (QCL-IDS) and QFI-EWC (Hsu et al.) are the closest conceptual neighbours to "fidelity/PGM-based class-state retention". Cite them to pre-empt novelty objections.

### Gaps
- Exact qubit counts and per-task accuracies for 2409.09729, 2511.17228 and 2607.16030 were not extracted. Only abstracts were read.
- I could not confirm that the Situ et al. Physica A article number matches arXiv:2203.14032 beyond the RePEc listing.
- I did not search Chinese-language venues, or IEEE QCE / QTML proceedings, for further QCL papers. More short papers may exist there.

## Q2. Quantum-inspired (classically simulated) methods on standard CIL benchmarks with PTMs: any that beat strong PTM baselines?

### Takeaway
There is a direct precedent: **QKD (CVPR 2026)**, a simulated shallow PQC gate inside a ViT-B/16 + adapter CIL pipeline. It reports state-of-the-art results on CIFAR-100, CUB, ImageNet-A, ImageNet-R and VTAB. Its gains over EASE/MoTE are often small (IN-R +0.03 average accuracy). Its strongest compared baselines are EASE and MoTE; MoS, TUNA, SSIAT and RSIAT are absent. It has no quantum-advantage or simulability discussion, only a "replace the quantum gate with MLP / cosine / attention" ablation. Separately, the PGM/Helstrom "quantum-inspired classifier" line (Sergioli et al.) exists but works only on small tabular data, with no CIL.

### Cited Findings
- **Li, Xiao, Cao, Wu, Ji, "Quantum-Gated Task-interaction Knowledge Distillation for Pre-trained Model-based Class-Incremental Learning" (QKD)**, arXiv:2604.11112 (13 Apr 2026), accepted to CVPR 2026 as a poster.
  - **Method:**
    - A shallow PQC with Ry angle encoding, trainable Ry layers and a CNOT chain.
    - Inputs are SVD-reduced to 12 dimensions. Qubit sweep: {3, 6, 9, 12, 15}.
    - Readout is the fidelity |<psi|phi_i>|^2 between the sample state and per-task states, followed by a softmax with tau = 1.
    - The gate weights distillation from old adapters to new ones.
    - No simulator is named and there are no hardware runs.
    - It mentions exponential concentration of fidelity kernels only in related work.
  - **Setup:** ViT-B/16-IN21K. Splits: CIFAR B0-Inc10, CUB B0-Inc20, IN-A B0-Inc10, IN-R B0-Inc20, VTAB B0-Inc10.
  - **Results (average acc. / last acc.):**

    | Benchmark | QKD | Best baseline |
    |---|---|---|
    | CIFAR B0-Inc10 | 94.08 / 90.20 | MoTE 93.06 / 88.98 |
    | CUB B0-Inc20 | 93.54 / 90.12 | MoTE 91.83 / 86.77 |
    | IN-A B0-Inc10 | 65.77 / 55.56 | MoTE 63.00 / 49.00 |
    | IN-R B0-Inc20 | 81.79 / 77.67 | EASE 81.76 / 75.30 |
    | VTAB B0-Inc10 | 95.81 / 92.20 | MoTE 93.56 / 85.94 |

  - **Classical-replacement ablation (Table 4, average acc.):**
    - IN-R: MLP 80.25, Cosine 80.32, Attention 80.08, Quantum 81.79.
    - CUB: MLP 92.63, Cosine 92.72, Attention 92.57, Quantum 93.54.
    - Parameter counts are similar (about 1.3 to 1.7M; Attention is about 10.8M).
    - Quantum latency is about 22.8 ms/img versus 13.2 ms for MLP on IN-R.
  - There is a possible table inconsistency: the claimed VTAB final-accuracy gain does not match the listed numbers.
  - Sources: [arXiv abs](https://arxiv.org/abs/2604.11112); [arXiv HTML](https://arxiv.org/html/2604.11112); [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/39787)
- **Sergioli and collaborators: Helstrom Quantum Centroid and PGM quantum-inspired classifiers.**
  - Each pattern is encoded as a density matrix. Classification uses Helstrom (binary) or the pretty-good measurement (multiclass).
  - Using more "quantum copies" improves accuracy.
  - Giuntini, Holik, Park, Freytes, Blank, Sergioli (2021 preprint) compared against 11 classical classifiers and report "surprisingly good performance".
  - A kernelized PGM (kPGM) addresses the cost in high dimensions.
  - Sources: [UniCa Quantum-like classifier 2021](https://iris.unica.it/bitstream/11584/352278/6/Quantum_like_Classifier_2021.pdf); [kPGM arXiv 2512.16960](https://www.arxiv.org/pdf/2512.16960); [NITheCS seminar abstract](https://nithecs.ac.za/events/all-events/a-quantum-inspired-approach-to-supervised-multi-class-classification/2026-07-31-nithecs-qsun-seminar.pdf/@@display-file/file)
  - Details are from search snippets and were not independently verified.

### Inferences
- QKD sets the bar and the template.
  - A CV venue (CVPR) accepted a classically simulated quantum component in PTM-CIL with no advantage claim.
  - The justification was empirical SOTA plus a same-budget classical-substitute ablation.
  - A new submission must (a) cite and compare with QKD, and (b) exceed QKD's ablation rigor. Useful additions: an entanglement-free or product-state variant, an RBF/cosine kernel with tuned bandwidth, and multiple seeds with confidence intervals.
- QKD's IN-R margin over EASE (+0.03 average accuracy) is within seed noise. Reviewers aware of this may discount "quantum" gains of under 1 point unless they come with multiple seeds and significance tests.
- RSIAT (CVPR 2026) appears to be absent from QKD's tables. A head-to-head against RSIAT plus QKD would be novel.
- PGM/Helstrom heads over density-matrix class states have a quantum-inspired classification precedent, but not in CIL or with PTM features. Cite Sergioli et al. for the PGM head's lineage.

### Gaps
- I could not confirm whether QKD released code, or whether its numbers used multiple seeds.
- No other quantum-inspired PTM-CIL paper turned up in two searches. There may be others in ICCV/NeurIPS 2025-2026 that were not found.
- I did not verify a peer-reviewed venue for the Sergioli PGM multiclass paper. A *Quantum Information Processing* or *Int. J. Theor. Phys.* placement is possible but **[UNVERIFIED]**.

## Q3. Dequantization, classical simulability and benchmarking positions: what do they recommend for fair classical baselines?

### Takeaway
The community consensus is that QML claims must survive strong, tuned classical baselines and "de-quantum" ablations. Those ablations include no entanglement, classical kernels of the same geometry, and RFF/classical surrogates. Many trainable or concentration-free quantum models are suspected to be classically simulable. Small-scale "quantum beats classical" results are treated as likely positivity bias.

### Cited Findings
- **Tang, "A quantum-inspired classical algorithm for recommendation systems"**, STOC 2019 (arXiv:1807.04271). **[UNVERIFIED in this session; from memory]**
  - Started "dequantization": classical sample-and-query algorithms that match some QML linear-algebra speedups up to polynomial factors.
- **Schuld & Killoran, "Is quantum advantage the right goal for quantum machine learning?"**, *PRX Quantum* (2022), arXiv:2203.01340. The volume/article number (believed to be 3, 030101) is **[UNVERIFIED]**.
  - Argues that ML may be one of the hardest places to show practical advantage.
  - Urges moving past the "tunnel vision of quantum advantage" toward understanding quantum models, for example via their kernel connection.
  - Source: [arXiv](https://arxiv.org/pdf/2203.01340)
- **Huang, Broughton, Mohseni, Babbush, Boixo, Neven, McClean, "Power of data in quantum machine learning"**, *Nature Communications* 12 (2021), arXiv:2011.01938.
  - Introduces the **geometric difference** g between quantum and classical kernels, computed classically via SVD of the N x N kernel matrices.
  - If g is small, a classical model is guaranteed to perform similarly or better.
  - g should be checked against a *suite of optimized classical models*.
  - Introduces **projected quantum kernels** to counter the weaknesses of fidelity kernels.
  - States: "with the help of data, classical machine learning models can be competitive with quantum models even if they are tailored to quantum problems."
  - Sources: [arXiv](https://arxiv.org/pdf/2011.01938); [Google Research](https://research.google/pubs/power-of-data-in-quantum-machine-learning/)
- **Thanasilp, Wang, Cerezo, Holmes, "Exponential concentration in quantum kernel methods"**, *Nature Communications* 15 (2024), DOI 10.1038/s41467-024-49287-w, arXiv:2208.11060.
  - Kernel values concentrate exponentially in qubit number.
  - Four causes: embedding expressivity, global measurements, entanglement, noise.
  - Concentrated kernels need exponentially many shots, and the learned model becomes data-independent.
  - Trainable embeddings via kernel alignment are also susceptible.
  - Sources: [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11189509/); [arXiv](https://arxiv.org/abs/2208.11060v2)
- **Cerezo et al., "Does provable absence of barren plateaus imply classical simulability?"**, *Nature Communications* 16, 7907 (2025), arXiv:2312.09121.
  - Argues that many barren-plateau-free PQC losses live in small, polynomial subspaces.
  - Such losses are therefore often classically simulable, possibly after collecting some data from the device.
  - The published version is more hedged than v1 and lists caveats such as smart initialisation and models outside its assumptions.
  - Sources: [arXiv](https://arxiv.org/abs/2312.09121v3); [Strathprints](https://strathprints.strath.ac.uk/93944)
- **Sweke et al., "Potential and limitations of random Fourier features for dequantizing quantum machine learning"**, *Quantum* 9, 1640 (Feb 2025), arXiv:2309.11647.
  - Gives necessary and sufficient conditions under which RFF efficiently dequantizes variational QML regression.
  - Sources: [arXiv](https://arxiv.org/abs/2309.11647v3); [DOAJ](https://doaj.org/article/87c8109da60d4dd582fb1313938566a8)
  - Follow-up: Sweke, Shin, Gil-Fuster, "Kernel-based dequantization of variational QML without Random Fourier Features" (arXiv:2503.23931). It shows the relevant kernels can often be evaluated exactly and classically.
  - Source: [search result](https://sinoxiv.napstic.cn/article/21378768); the arXiv ID was not opened.
- **Bowles, Ahmed, Schuld (Xanadu), "Better than classical? The subtle art of benchmarking quantum machine learning models"**, arXiv:2403.07059 (Mar 2024). The arXiv page lists no journal reference. Published-venue status is **[UNVERIFIED]**.
  - **Study:** 12 QML models on 160 datasets from 6 binary tasks, with up to 18 qubits simulated.
  - **Results:**
    - Out-of-the-box classical models (SVC, MLP, CNN) outperform the quantum classifiers. The top model in every family was classical.
    - Hybrid models perform about the same as their classical hosts.
    - Removing entanglement often gives equal or better results.
  - **Recommendations:**
    - Use strong out-of-the-box classical baselines. Do not bother matching parameter counts.
    - Run wide hyperparameter searches for *both* sides and report them.
    - Test non-entangling or classically simulable variants.
    - Avoid downscaled or binary MNIST.
    - Ask narrow, testable questions rather than "does quantum beat classical?"
    - Watch for positivity bias. Reporting the best of many quantum variants made a published quantum mean of 0.74 beat a classical 0.65, although the true means favoured classical.
  - **Literature audit (arXiv 2018-2023 papers matching "quantum machine learning" + "outperform"):**
    - About 40% claimed quantum beats classical.
    - About 50% claimed an improvement to a QML method.
    - Only about 4% reported that quantum did not outperform classical.
  - The five design questions it poses:
    1. Do quantum components in hybrids help, or just mimic the classical parts they replace?
    2. What makes data reuploading work?
    3. What distances do quantum kernels define?
    4. Why do polynomial features fail?
    5. Why do quantum models struggle on linearly separable data?
  - Sources: [arXiv abs](https://arxiv.org/abs/2403.07059v2); [arXiv HTML](https://arxiv.org/html/2403.07059v2); [PennyLane blog](https://pennylane.ai/blog/2024/03/benchmarking-near-term-quantum-algorithms-is-an-art/)
- **Schnabel & Roth (Fraunhofer IPA), "Quantum Kernel Methods under Scrutiny: A Benchmarking Study"**, arXiv:2409.04406, published in *Quantum Machine Intelligence* (2025).
  - Benchmarks more than 20,000 fidelity and projected quantum kernel models, using 9 encoding circuits with up to 15 qubits, on SVC, KRR and SVR.
  - A 2026 review cites it for the finding that bandwidth, feature scaling and regularization dominate performance, and that earlier advantage claims may not hold under fair comparison.
  - Sources: [arXiv](https://arxiv.org/html/2409.04406v2); [Fraunhofer](https://publica.fraunhofer.de/entities/publication/21cfeb4e-fc6a-4920-83e2-a0a1fb912a75); [review arXiv 2604.07896](https://arxiv.org/pdf/2604.07896)

### Inferences
These points apply to the researcher's PQK, density-matrix and PGM components.
- **Projected quantum kernels behave like bandwidth-scaled Gaussian kernels.** Thanasilp et al. and Schnabel & Roth show that bandwidth/scaling dominates. Reviewers will ask for an RBF/Laplacian kernel on the same features with a tuned bandwidth, and ideally Huang's geometric difference g.
- **Amplitude-encoded density matrices with a PGM head are classically computable linear algebra.** For d-dim features, rho_c is d x d, and PGM uses rho^{-1/2} rho_c rho^{-1/2}. This puts them in the "quantum-inspired / dequantized" category by construction. The honest framing is that they are **quantum-information-theoretic inductive biases**, not quantum advantage. This is consistent with Schuld & Killoran's call to move past advantage.
- **Minimal fair baselines.** In line with Bowles et al., these would show whether the "quantum" ingredient matters:
  - Covariance/second-moment class prototypes with an LDA- or Mahalanobis-style head. This is the classical analogue of density-matrix class states, and RanPAC/FeCAM-style heads already exist in PTM-CIL.
  - Whitened nearest-class-mean as the analogue of PGM.
  - An RBF kernel at matched bandwidth as the analogue of PQK.
  - Entanglement-free / product-state variants.
  - Equal hyperparameter-tuning budgets on both sides.
  - Multiple seeds with confidence intervals.
- **The researcher's own finding is a publishable "careful benchmark" result.** The repo memory notes that "quantum not much more effective than RSIAT". This is the kind of evidence Bowles et al. say is under-reported (about 4%).

### Gaps
- I did not verify Gil-Fuster, Eisert, Bravo-Prieto, "Understanding quantum machine learning also requires rethinking generalization", *Nature Communications* (2024). It is relevant to generalization claims. **[UNVERIFIED]**
- No published paper specifically dequantizes PGM heads or density-matrix class prototypes. The claim that these equal classical covariance/whitening heads is my inference and needs a short derivation in the paper.
- I did not read the second half of the Bowles et al. HTML (only the first ~100k characters), so any additional recommendations there are missing.

## Q4. Are negative-result / "quantum does not help" papers accepted? Which venues are receptive?

### Takeaway
Yes, in quantum-specialist venues. Large-scale skeptical benchmarks and simulability/concentration theory appear in *Nature Communications*, *Quantum*, and *Quantum Machine Intelligence*. Bowles et al. is highly cited, but its journal status is unverified. At CV venues, the accepted quantum-inspired CIL paper (QKD, CVPR 2026) was positive and SOTA-framed. I found no verified example of a negative quantum result accepted at CVPR, ICCV or NeurIPS main track.

### Cited Findings
- **Nature Communications:**
  - Thanasilp et al. 2024 (kernel concentration) — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11189509/)
  - Cerezo et al. 2025 (BP-free implies simulable?) — [arXiv](https://arxiv.org/abs/2312.09121v3)
  - Huang et al. 2021 ("classical can be competitive with data") — [Google Research](https://research.google/pubs/power-of-data-in-quantum-machine-learning/)
- **Quantum (journal):** Sweke et al. 2025, RFF dequantization — [DOAJ](https://doaj.org/article/87c8109da60d4dd582fb1313938566a8)
- **Quantum Machine Intelligence (Springer):** Schnabel & Roth 2025, a benchmarking study of over 20,000 quantum kernels — [Fraunhofer](https://publica.fraunhofer.de/entities/publication/21cfeb4e-fc6a-4920-83e2-a0a1fb912a75)
- **PRX Quantum:** Schuld & Killoran 2022, a perspective questioning advantage as the goal — [arXiv](https://arxiv.org/pdf/2203.01340)
- **npj Quantum Information:** Zhang et al. (hardware QCL) — [arXiv](https://arxiv.org/abs/2409.09729)
- **Chinese Physics Letters / Physica A:** the first QCL papers (Jiang et al.; Situ et al.) — [CPL](https://cpl.iphy.ac.cn/article/doi/10.1088/0256-307X/39/5/050303); [RePEc](https://ideas.repec.org/a/eee/phsmap/v620y2023ics0378437123003345.html)
- **CVPR 2026:** QKD, a quantum-inspired, classically simulated PTM-CIL method. Accepted on SOTA numbers plus a classical-substitute ablation, with no advantage discussion — [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/39787)
- **Positivity-bias evidence:** only about 4% of surveyed QML "outperform" papers report a negative result — [Bowles et al.](https://arxiv.org/html/2403.07059v2)

### Inferences
There are three realistic routes for a "quantum components in RSIAT-style CIL" paper:
1. **CV venue (CVPR / ICCV / ECCV / NeurIPS main).**
   - Frame the work as a quantum-information-inspired head/regularizer with a SOTA or near-SOTA result.
   - The bar is QKD-style ablations plus stronger ones (RSIAT and QKD baselines, classical covariance/RBF analogues, seeds).
   - A pure negative result is unlikely to land here.
2. **Quantum-ML venue (Quantum Machine Intelligence, Quantum, or the QTML / IEEE QCE conferences).**
   - Frame the work as the first realistic-scale benchmark of PQK, density-matrix and PGM components in PTM-CIL.
   - Report honestly where quantum components match or lose to classical analogues, and give simulability arguments.
   - This is the most receptive venue type for negative or mixed results, given Schnabel & Roth's placement.
3. **TMLR.** Its stated criterion is correctness and interest rather than novelty or SOTA, which suits a careful mixed-result study. **[UNVERIFIED policy wording; check the TMLR site]**

Two further points:
- The NeurIPS / ICML QML workshops are suitable for a short version. Specific 2025-2026 workshop names were not verified.
- The QKD precedent cuts both ways. It shows CV reviewers accept "quantum" components without advantage proofs. It also means a new paper's novelty must go beyond "a simulated PQC in PTM-CIL".

### Gaps
- I did not verify whether Bowles et al. 2403.07059 has since been published in a journal or conference.
- I did not retrieve TMLR's acceptance-criteria text, or calls for papers for QTML, IEEE QCE or NeurIPS QML workshops in 2025-2026.
- I did not find OpenReview reviews of quantum-in-CV submissions showing what reviewers actually asked for. Reviewer expectations above are inferred from the benchmarking literature and the QKD ablation design.
