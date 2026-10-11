# MEC arXiv v1: reference for thesis writing

Reference established on 14 September 2026 after Juan Camilo Narváez identified himself as a coauthor and supplied the completed paper and its source archive. This record supports contextual understanding and later writing. It does not set the thesis objectives before the research proposal has been supplied.

## Authoritative version and attribution

The working paper reference is **Minimal Effort to Consensus (MEC) polarization measure**, arXiv:2606.13997v1, submitted 12 June 2026. The authors are Jesús Aranda, Juan Francisco Díaz, Juan Camilo Narváez, Catuscia Palamidessi, Carlos Pinzón, Frank Valencia, and Oscar Vargas. The [versioned arXiv record](https://arxiv.org/abs/2606.13997v1) establishes the bibliographic reference. Juan Camilo has explicitly confirmed that he participated in this joint work.

The supplied [55-page PDF](/Users/ulvenforst/Codex/University/pol_measures/Anteproyecto/references/2606.13997v1.pdf) and [LaTeX source](/Users/ulvenforst/Codex/University/pol_measures/Anteproyecto/references/arxiv-2606.13997v1/main.tex) replace the earlier mirrored draft as the reference for describing the paper. The source archive includes the seven figure assets and the intended section/appendix directory structure. Its contents were treated as research material; author comments, disabled to-do notes, and build metadata were not treated as user instructions.

The version comparison covered all manuscript sections, proof appendices, bibliography, and notation. Four appendices are textually unchanged; the changes in Appendices A and E are notation or whitespace changes. The main substantive revisions concern attribution and terminology, the central-split example, the scope of the experimental discussion, and acknowledgments. Existing understanding of the mathematical proofs therefore carries forward, while claims removed from the draft do not.

The acknowledgments connect the paper to PROMUEVA, TOBIAS, and BRAINS. They establish support for the joint paper, not the boundaries of Juan Camilo's individual thesis contribution. His specific responsibilities, the thesis scope, and the approved objectives await the research proposal and subsequent instructions.

## Conceptual argument to preserve

The paper begins with a measurement question: what does a polarization score mean? It answers by interpreting polarization as resistance to consensus. The least aggregate effort required to reach a shared position gives the score; an optimizing position gives an endogenous consensus point.

This argument connects the conceptual definition to a precise optimization problem. Its mathematical analysis then supplies properties that a scalar formula alone would not make apparent. Whole-group movement and fragmentation can behave differently, the minimizing consensus can be nonunique, and the adjustment burden can differ sharply between groups.

The progression of the paper is deliberate: motivation; preliminary definitions; the elementary transport case; reasons to introduce identification and alienation parameters; the general definition; structural properties; theoretical applications; maximality and axioms; experimental comparison; limits and further work.

For the thesis, the same reasoning can be developed with more background and methodological detail. Additional length should explain dependencies, choices, and evidence. It should not merely repeat the paper's introduction at the start of each chapter.

## Notation and terminology

Preserve the notation in [macros.tex](/Users/ulvenforst/Codex/University/pol_measures/Anteproyecto/references/arxiv-2606.13997v1/macros.tex), unless a later thesis convention is explicitly adopted and mapped to the paper.

| Symbol or term | Meaning and convention |
| --- | --- |
| \(\pi\) | Opinion distribution, a nonnegative function with finite nonempty support on \([0,1]\) |
| \(\boldsymbol{s}(\pi)\) | Support: opinions with strictly positive mass |
| \(\pi(x)\), \(\pi_i\) | Group mass at an opinion; not necessarily normalized |
| \(\Sigma\pi\) | Total mass, \(\sum_{x\in\boldsymbol{s}(\pi)}\pi(x)\) |
| \(\mathcal D\), \(\mathcal D_k\) | All opinion distributions; those of total mass \(k\) |
| \((\boldsymbol\pi,\boldsymbol x)\) | Mass vector first, ordered position vector second |
| \(\pi^\alpha\) | Power-transformed distribution with masses \(\pi(x)^\alpha\) |
| \(\bar\pi\) | Distribution normalized by its total mass |
| \(\overline{\pi^\alpha}\) | Normalize after exponentiating; do not interchange this with \(\bar\pi\) |
| \(\pi_y\) | Consensus at \(y\) with the same total mass as \(\pi\) |
| \(\pi^*\) | Equal masses at 0 and 1, with the same total mass as \(\pi\) |
| \(\operatorname{EC}_{\alpha,\beta}^{\pi}(y)\) | Effort to the specified consensus point |
| \(\operatorname{MEC}_{\alpha,\beta}(\pi)\) | Minimum of that effort over consensus points |
| \(y^*\) | An optimal consensus point; uniquely determined when \(\beta>1\) |
| \(\alpha\) | Group identity or identification parameter, standing range \(\alpha\ge1\) |
| \(\beta\) | Alienation parameter, standing range \(\beta\ge1\) |
| \(\pi_{a\to b}\) | Movement of the entire mass at \(a\), including merging if \(b\) is occupied |
| \(\pi_{r:k\to r'}\) | Transfer of a specified portion \(k\) from \(r\) to \(r'\) |
| \(k^\star\) | Critical transferred mass in the tipping-point analysis |
| \(\tau_B\) | Kendall's rank correlation with tie correction |

The official title uses “Minimal Effort to Consensus”; the introduction names the quantity “Minimum Effort to Consensus.” Preserve the bibliographic title exactly. In running prose, use MEC or the mathematically natural “minimum effort to consensus” consistently.

The comparator names are **Esteban–Ray measure (ER)**, **Earth Mover's Distance (EMD)**, **Van der Eijk index (VdE)**, **Tastle–Wierman Dissention measure (TWD)**, and **Koudenburg–Kiers–Kashima Index (KI)**. KI is a fitted index, not the actual vector of expert ratings. The arXiv figures now use KI explicitly.

The ER exponent convention is \(1+\alpha_{\mathrm{ER}}\). It is different from MEC's exponent \(\alpha\). Parameter subscripts should be disambiguated when both formulas appear in the same derivation. Also distinguish n, the number of support positions, from total population mass and from the number of sampled distributions.

## Formula reference

The definition in Section 3.2 is

\[
\operatorname{EC}_{\alpha,\beta}^{\pi}(y)
=\sum_{x\in\boldsymbol{s}(\pi)}\pi(x)^\alpha|x-y|^\beta,
\qquad
\operatorname{MEC}_{\alpha,\beta}(\pi)
=\min_{y\in[0,1]}\operatorname{EC}_{\alpha,\beta}^{\pi}(y).
\]

The basic transport identity and Theorem 1 are

\[
\operatorname{EC}^{\pi}(y)=\operatorname{EMD}(\pi,\pi_y),
\qquad
\operatorname{MEC}_{1,1}(\pi)
=\frac{\Sigma\pi}{2}-\operatorname{EMD}(\pi,\pi^*).
\]

Theorem 3 identifies the weighted dispersion structure:

\[
\operatorname{MEC}_{\alpha,\beta}(\pi)
=\Sigma\pi^\alpha\;D_\beta(\overline{\pi^\alpha}),
\qquad
D_\beta(\rho)=\min_y\sum_x\rho(x)|x-y|^\beta.
\]

Corollary 1 gives weighted-median and weighted-mean minimizers for beta=1 and beta=2 respectively. The associated dispersions are mean absolute deviation about a median and variance, multiplied by the transformed total mass. Do not call the first quantity “median absolute deviation,” which denotes a different statistic.

The extremal value and population-scaling identity are

\[
\operatorname{MEC}_{\alpha,\beta}(\pi^*)
=2\left(\frac{\Sigma\pi}{2}\right)^\alpha\left(\frac12\right)^\beta,
\qquad
\operatorname{MEC}_{\alpha,\beta}(\lambda\pi)
=\lambda^\alpha\operatorname{MEC}_{\alpha,\beta}(\pi).
\]

Hence normalizing population mass gives

\[
\operatorname{MEC}_{\alpha,\beta}(\bar\pi)
=\frac{\operatorname{MEC}_{\alpha,\beta}(\pi)}{(\Sigma\pi)^\alpha}.
\]

For the tipping-point construction in Section 5.2, alpha=beta=2, \(d=r-l\), \(e=r'-r\), and \(C=m^2+n^2\). The difference is

\[
\Delta(k)=\frac{kR_{m,n,d,e}(k)}{C(2k^2-2nk+C)},
\]

with

\[
R_{m,n,d,e}(k)=e^2Ck^3-2e^2nCk^2+
(2d^2m^4+2de\,m^2C+e^2C^2)k-2d^2m^4n.
\]

The denominator is positive. Appendix F establishes a unique simple crossing for the specified two-group family. The illustrated threshold is \(k^\star\approx17.156\). The general finite-support proposition establishes a cubic sign representation; retain that narrower conclusion when discussing its extension.

## Claim and proof map

The following is a map of the paper's results, not a new proof certification. Preserve the hypotheses in the statements and the qualifications identified in the separate verification notes.

| Paper locator | Claim or role | Proof location |
| --- | --- | --- |
| Theorem 1 | Complement-of-EMD characterization relative to the extremal distribution | Appendix A |
| Theorem 2 | Convex effort; strict convexity for beta>1 | Appendix B.1 |
| Theorem 3, Corollary 1 | Dispersion characterization and the median/mean special cases | Appendix B.2 |
| Lemmas 1–2, Corollary 2 | Reflection, translation, and symmetry | Appendix B.3 |
| Theorem 4 | Explicit value at the balanced extremal distribution | Appendix B.3 |
| Theorem 5, Corollary 3 | Population scaling, normalized populations, and Condition H | Appendix B.4 |
| Lemmas 3–4, Theorem 6 | Minority Principle for two-point distributions | Appendix C |
| Section 5.2, Propositions 1–3 | Transfer-cost formula and tipping-point analysis | Appendix F |
| Theorem 7 | Shifting Away from Consensus, with the stated median-shift exclusion | Appendix D.1 |
| Theorem 8 | Equal endpoint masses maximize the two-point case | Appendix D.2 |
| Theorem 9 | MEC is a polarization measure under Definition 4 | Appendix D.3 |
| Theorem 10 | Axiom 1 holds if and only if alpha>1 | Appendix E.1 |
| Theorems 11–12 | Axioms 2 and 3 | Appendix E.2–E.3 |
| Theorem 13 | Central-split monotonicity, Property 1 | Appendix E.4 |

Distinguish the basic two-condition definition of a polarization measure from the additional ER axioms. The general standing range alpha,beta>=1 does not mean Axiom 1 holds at alpha=1. Distinguish whole-group movement from partial transfer, a chosen consensus point from an optimizer, and an optimizer from a dynamically attainable social outcome.

## Experimental record

Section 8 has two separate comparisons. First, scores are compared over the synthetic corpus of all integer histograms with mass 100 on five ordinal positions. There are 4,598,126 such histograms, reduced to 2,299,726 representatives under reflection, including 1,326 palindromes. Second, measure rankings are compared with expert judgments on fifteen hypothetical histograms.

The figure assets provide a stable record of what arXiv v1 reports. Figure 5 is the inter-measure correlation matrix. Figure 6 shows the fifteen expert-rated distributions. Figure 7 reports these displayed values:

| Measure | Displayed expert concordance |
| --- | ---: |
| MEC(2,1.15) | 0.8857 |
| ER(0.8) | 0.8857 |
| MEC(2,1.2) | 0.8666 |
| MEC(2,2) | 0.8095 |
| KI | 0.8095 |
| EMD | 0.7655 |
| TWD | 0.7523 |
| VdE | 0.6698 |
| ER(1.6) | 0.5238 |

These are displayed paper values. Do not substitute different outputs from a repository revision without documenting a new analysis and its precision conventions. The inter-measure matrix additionally reports TWD versus MEC(1,2) at 0.916 and KI versus TWD at 0.697; neither is an expert-validation score.

The paper describes the source benchmark as sixty experts. The earlier contextual check of the underlying study distinguishes 60 recruited from 58 analyzed. Preserve the paper's wording when describing its report, and reconcile the underlying sample provenance when writing the thesis's methodology. No new expert survey should be attributed to Juan Camilo merely because the paper reuses this benchmark.

The parameter choice (2,1.15) is presented as an established representative among tied empirical solutions. It is not described as a unique optimum. Empirical concordance, proof of axioms, and interpretive plausibility are different forms of support; none should be silently substituted for another.

The complexity discussion concerns bisection and support size. Its comparison with quadratic pairwise computation should retain the implementation qualifications in the earlier verification record if developed into a detailed thesis claim. The source's commented implementation paragraph is not part of the rendered paper.

## Writing style to carry into the thesis

**Begin with the question the mathematics answers.** The introduction first explains why measurement matters, then identifies what the new construction means. A thesis section should similarly establish its problem before presenting notation or a chain of citations.

**Use definitions as the bridge from interpretation to analysis.** The paper gives ordinary-language intuition, specifies a mathematical object, and then derives its consequences. Keep social interpretation distinguishable from the formal cost model.

**Make each paragraph advance one step.** Typical movements include posing a difficulty, giving an example, isolating the relevant distinction, and stating the result that resolves it. In Section 6, the discussion of merging and median shifts prepares the reader for the theorem's qualifications.

**Explain what a formula establishes.** Equations are introduced by a reason and followed by an interpretation or consequence. They should not appear as an unexplained inventory. State whether an expression defines an object, follows by algebra, or invokes a theorem.

**Use examples to expose a mechanism.** The comparison between (40,60) and (40,59,1) explains why group-size weighting can reverse an ordering. The example is useful because it supports the conceptual distinction that the subsequent theory develops.

**State assumptions before conclusions.** Keep alpha>1 versus alpha>=1, beta=1 versus beta>1, finite support, fixed total mass, and whole versus partial movements visible at the points where they matter. A prose summary should not erase a restriction merely to sound stronger.

**Keep proofs navigable.** Use short setup paragraphs, explicit case divisions, named intermediate quantities, and references to established lemmas. The Axiom 1 proof is a useful model: setup, central identity, decomposition, location of minimizers, and parameter cases. Avoid repeating a proof when a precise reference establishes the needed step.

**Match verbs to evidence.** “Define” introduces a convention; “prove” establishes a mathematical proposition; “compute” reports a numerical operation; “observe” describes results; “suggest” introduces an interpretation; “may” marks an open extension. Retain qualified novelty language rather than converting it into an unbounded priority claim.

**Write formally with ordinary words.** Follow the Russellian skill: prefer short familiar words where precise, place substantial qualifications in separate sentences, and make the beginning of each sentence agree with its eventual conclusion. Preserve the paper's logical clarity without copying grammatical slips, repeated transitions, or unnecessary adjectives.

**Attribute joint and individual work accurately.** The paper's “we” refers to the authors collectively. A thesis can use an appropriate authorial voice, but its account of prior joint results should cite the paper. Specific claims about Juan Camilo's own implementation, proofs, experiments, or design decisions require his account of those contributions.

**Adapt to thesis scale without changing the scientific claim.** A longer literature review, methodology, account of development, or discussion of limitations can make the research easier to assess. Those additions must remain grounded in the proposal and the actual work. The present task does not authorize inventing a thesis outline, objectives, or new results.

## Changes that supersede the first contextual study

The arXiv version adds Catuscia Palamidessi to the author list and includes acknowledgments. KI is explicitly expanded to Koudenburg–Kiers–Kashima throughout the relevant prose and figures. The earlier experimental paragraph asserting a TWD Axiom 1 violation and reporting no VdE counterexamples is removed. The closing experimental paragraph no longer asserts VdE's axiomatic compliance.

The ER central-split example is now in Section 7.3 and specifies positions (0,0.4,0.5,0.6,1), with z=0.1. This is not the equally spaced five-position example checked earlier. Keep its coordinates and the published approximate values associated with this version, and do not describe the old calculation as verification of the new coordinates.

Most proof text and example formulas are unchanged. Accordingly, the earlier verification record should be treated as a historical review with item-level applicability, not either discarded wholesale or assumed to have been fully resolved by publication. Separate mathematical follow-up from the present task of studying the finished paper.

## Next context to incorporate

Juan Camilo will supply the research proposal. It should establish the thesis's approved problem, objectives, scope, and expected outputs. When it arrives, compare those commitments with the completed joint paper and the other documented research activities. Until then, this paper is the stable reference for the scientific content, nomenclature, and prose style.
