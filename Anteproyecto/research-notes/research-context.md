# AVISPA research context: MEC and the surrounding computational work

**Version notice:** This document records the initial study of the mirrored draft and the inspected repository snapshot. For the completed paper supplied subsequently by Juan Camilo Narváez, start with [the arXiv v1 thesis reference](arxiv-v1-thesis-reference.md). That reference supersedes this document wherever the manuscript versions differ. Repository observations remain tied to the commit below.

Study date: 14 September 2026. This is a working research record for future thesis development, not a thesis draft or an attribution of individual contributions.

## Evidence and coverage

I read the nine manuscript sections, six proof appendices, notation file, main LaTeX file, and bibliography supplied under `sources/`. I also extracted the 55-page PDF and visually inspected its experimental figures and surrounding text on pages 25–27. The manuscript source files remain unchanged.

The source mentions a public implementation in a commented paragraph. I inspected a separate reference copy of [pol_measures](https://github.com/ulvenforst/pol_measures/tree/1efd4fe3f083e478cb91a6f3ca078cad01386a80), including its MEC implementation, normalization, bisection helper, comparison pipeline, expert data, parameter-search report, and selected extension and timing artifacts. This snapshot is commit `1efd4fe3f083e478cb91a6f3ca078cad01386a80`, dated 28 May 2026. It must not automatically be treated as the version that generated the manuscript figures. The PDF's creation metadata is 8 May 2026; that is not evidence of publication or submission.

Primary external checks concerned the original Esteban–Ray axioms and the provenance of the Koudenburg benchmark. This was a focused contextual review, not a systematic literature review. The current evidence does not establish the complete history of AVISPA or PROMUEVA, the student's individual contributions, or the final thesis requirements.

## Research problem and organizing idea

The central problem is how to quantify polarization in a distribution of opinions while giving the resulting number an explicit interpretation. The manuscript proposes resistance to consensus as that interpretation. A distribution is more polarized when reaching a common opinion requires greater minimum aggregate effort under the specified cost model.

The construction produces two mathematical objects: a minimum cost and a set of cost-minimizing consensus positions. This connects measurement to an optimization problem. It differs conceptually from a pairwise antagonism index, even when both indices rank familiar examples similarly.

The primary research question can therefore be reconstructed as follows: can minimum effort to consensus define a mathematically tractable polarization measure that respects specified axioms and agrees with expert assessments? This wording is my synthesis of the manuscript, not a recovered original thesis objective.

The main methodological sequence is conceptual motivation, formal definition, structural analysis, analysis of opinion movements, axiomatic evaluation, numerical implementation, and comparative empirical evaluation. These parts support different conclusions: proofs establish behavior under assumptions; computational experiments establish outcomes for specified inputs; expert agreement assesses correspondence with a particular benchmark.

## Mathematical setting

An opinion distribution is a function \(\pi:[0,1]\to\mathbb R_{\ge0}\) with finite nonempty support. At a support point \(x_i\), the positive mass \(p_i=\pi(x_i)\) represents the group holding that opinion. Its total mass is \(M=\sum_i p_i>0\). Counts, proportions, and other positive masses are permitted; normalization is not part of the basic definition.

The interval represents agreement with a single proposition. Distances between opinions are meaningful numerical inputs. Applying the measure to an ordinal survey scale therefore requires a choice of numerical embedding; equal spacing is an assumption, not a consequence of ordinality.

A consensus distribution places all mass at one point. The extremal distribution places \(M/2\) at each of 0 and 1. The manuscript's basic definition of a polarization measure requires zero at consensus and a maximum at the balanced extremal distribution among distributions of equal total mass. These two conditions are weaker than satisfying the three Esteban–Ray axioms.

For \(\alpha,\beta\ge1\),

\[
E_{\alpha,\beta}^{\pi}(y)=\sum_i p_i^\alpha|x_i-y|^\beta,
\qquad
\operatorname{MEC}_{\alpha,\beta}(\pi)=\min_{y\in[0,1]}E_{\alpha,\beta}^{\pi}(y).
\]

The parameter \(\alpha\) changes the relative influence of group sizes; \(\beta\) changes the relative penalty for opinion displacement. For two distances \(d_2>d_1>0\), their cost ratio is \((d_2/d_1)^\beta\). Increasing \(\beta\) increases this ratio, although it decreases each fixed \(d^\beta\) when \(0<d<1\). Interpretation across parameter settings must respect this distinction.

Group identity and cognitive dissonance motivate these choices in the manuscript. They do not supply an empirically estimated psychological law equating actual human adjustment costs with \(p_i^\alpha d_i^\beta\).

## Structural results and their dependencies

Existence of a minimizer follows from continuity on the compact interval. Convexity follows from convexity of \(|x-y|^\beta\). For \(\beta>1\), strict convexity gives a unique minimizer. For \(\beta=1\), the minimizers form the weighted-median interval and need not be unique. Convexity here concerns the effort as a function of the consensus position; it should not be silently transferred to every other argument of MEC.

Write \(S_\alpha=\sum_i p_i^\alpha\) and \(\rho_i=p_i^\alpha/S_\alpha\). The central reduction is

\[
\operatorname{MEC}_{\alpha,\beta}(\pi)
=S_\alpha\min_y\sum_i\rho_i|x_i-y|^\beta.
\]

Thus the general measure is a scaled dispersion of the normalized power-transformed distribution. In particular:

| Case | Optimal consensus | Minimum cost |
| --- | --- | --- |
| \(\beta=1\) | Any median of \(\rho\) | \(S_\alpha\) times mean absolute deviation about that median |
| \(\beta=2\) | \(\sum_i\rho_i x_i\) | \(S_\alpha\operatorname{Var}_{\rho}(X)\) |
| Distribution symmetric about \(c\) | \(c\) is a minimizer | Direct evaluation at \(c\) |

For \((\alpha,\beta)=(1,1)\), effort is the equal-mass transportation cost to a Dirac consensus. The one-dimensional quantile representation yields

\[
\operatorname{MEC}_{1,1}(\pi)
=\min_y\operatorname{EMD}(\pi,M\delta_y)
=\frac M2-\operatorname{EMD}(\pi,\pi^*).
\]

The second equality is a complement of distance to the extremal distribution, not that distance itself. For probability masses it has the usual \(W_1\) interpretation. More generally, \(\operatorname{MEC}_{1,\beta}(\pi)/M=\min_y W_\beta(\bar\pi,\delta_y)^\beta\). For \(\alpha>1\), the transportation interpretation applies to transformed masses. The raw generalized objective is not itself a metric between original distributions.

Reflection and admissible translation preserve MEC. Scaling population masses by \(\lambda>0\) multiplies MEC by \(\lambda^\alpha\), yielding Condition H and

\[
\operatorname{MEC}_{\alpha,\beta}(\bar\pi)
=\operatorname{MEC}_{\alpha,\beta}(\pi)/M^\alpha.
\]

The value at the balanced extremal distribution is \(M^\alpha2^{1-\alpha-\beta}\). The maximality argument moves whole groups outward, handles mergers through superadditivity of mass powers, and reduces to balanced endpoint masses. A separate median case is needed for \(\beta=1\). The strict movement statement needs a non-consensus assumption; see the verification record.

A natural normalized index on the fixed unit interval is

\[
P_{\alpha,\beta}(\pi)=2^{\alpha+\beta-1}\operatorname{MEC}_{\alpha,\beta}(\pi)/M^\alpha.
\]

It lies in \([0,1]\) under the maximality result. Taking \(P^{1/\beta}\) preserves rankings for a fixed parameter pair but changes cardinal values. Normalizing the population, power-transforming the masses, normalizing the score, and rescaling the opinion interval are distinct operations.

## Two-group adjustment and partial transfers

For exactly two groups of masses \(A,B>0\) at \(l<r\), and \(\beta>1\), let \(u=A^{\alpha/(\beta-1)}\), \(v=B^{\alpha/(\beta-1)}\). Direct solution of the first-order condition gives

\[
y^*=\frac{ul+vr}{u+v},
\qquad
\operatorname{MEC}_{\alpha,\beta}=\frac{A^\alpha B^\alpha(r-l)^\beta}{(u+v)^{\beta-1}}.
\]

The ratio of the left group's effort contribution to the right group's contribution is \((B/A)^{\alpha/(\beta-1)}\). Consequently, the smaller group contributes the greater total modeled effort. When \(\beta=1\) and the masses differ, the consensus is at the larger group and the minority bears all modeled effort. This is an optimization result for two groups, not an empirical or ethical assertion about what minorities should do.

The distinction between displacement and fragmentation is essential. Moving a whole group outward preserves its mass before any merger. Moving a fraction creates two separately weighted groups, replacing \(n^\alpha\) by contributions involving \((n-k)^\alpha\) and \(k^\alpha\). For \(\alpha>1\), this loss of effective group weight can outweigh increased distance.

For \(\alpha=\beta=2\), an initial two-group distribution at \(l<r\), with mass \(k\) moved from \(r\) to \(r'>r\), has a cost difference of the form

\[
\Delta(k)=kR(k)/Q(k),\qquad Q(k)>0,
\]

where \(R\) is cubic. Appendix F derives the coefficients from the weighted-variance identity and proves a unique simple root in \((0,n)\) using a negative discriminant. For masses 40 and 60 at 0.25 and 0.75, with movement toward 1, the baseline-crossing threshold is \(k^\star\approx17.1559796441\). This value was independently reproduced.

The threshold compares the resulting distribution with the original one: below it the cost is lower; above it the cost is higher. It is not automatically the location at which the derivative of cost first becomes positive. For a general initial finite-support distribution, the manuscript establishes a cubic sign representation for transfer from the rightmost occupied point toward 1. It does not establish the same unique-threshold result for every such distribution.

## Axiomatic comparison

Esteban–Ray sums pairwise identification-weighted distances. To avoid conflating parameter conventions, write its exponent as \(1+\eta\), reserving \(\alpha\) for MEC.

| Requirement | Meaning in this manuscript | MEC parameter conditions |
| --- | --- | --- |
| Basic polarization definition | Consensus zero; balanced endpoints maximal at fixed mass | \(\alpha,\beta\ge1\) |
| Condition H | Common population scaling preserves order | \(\alpha,\beta\ge1\) |
| ER Axiom 1 | Sufficiently small, sufficiently close groups become more polarizing when merged | \(\alpha>1,\beta\ge1\) |
| ER Axiom 2 | Specified asymmetric middle-group movement increases polarization | \(\alpha,\beta\ge1\) |
| ER Axiom 3 | Symmetric transfer of central mass to endpoint groups increases polarization | \(\alpha,\beta\ge1\) |
| Central-split Property 1 | Splitting central mass symmetrically to intermediate positions increases polarization | \(\alpha,\beta\ge1\), with endpoint-merger case treated separately |

The Axiom 1 proof balances the convexity cost of separating opinions against the identification gain from merging masses. Axiom 2 uses the location of the optimizer, with a separate weighted-median case. Axiom 3 and Property 1 follow from symmetry and direct evaluation.

The original ER characterization is conditional on its assumed pairwise form; MEC's compatibility does not contradict that characterization. ER's upper parameter boundary is mathematically characterized and numerically approximated near 1.6, not fitted from the expert sample. [Esteban and Ray, 1994](https://pages.nyu.edu/debraj/Courses/Readings/Esteban%20Ray94.pdf).

## Computational and empirical workflow

The manuscript defines all nonnegative integer histograms of 100 agents over five equally spaced positions. The count is \(\binom{104}{4}=4,598,126\). There are \(\binom{52}{2}=1,326\) reflection-symmetric histograms, yielding 2,299,726 representatives after identifying mirror images. I independently verified these counts.

The inspected pipeline enumerates distributions, converts masses to proportions, selects lexicographically canonical mirror representatives, evaluates candidate measures, and computes pairwise Kendall correlations. It rounds scores before the corpus comparison. Thus numerical precision and artificial ties are part of the method, not merely display choices. Uniform weighting of mirror classes also differs from weighting every original histogram equally; that distinction should be explicit when defining the target correlation.

The candidate family includes MEC settings, ER settings, the EMD-based baseline, Van der Eijk, Tastle–Wierman dissention, and the fitted Koudenburg index. The figure's label “Experts” denotes the fitted index when applied across the full corpus; it cannot denote human ratings for millions of distributions.

The expert benchmark contains fifteen displayed histograms and their aggregate ratings. The original study recruited 60 experts but analyzed 58 after excluding two responses. It used displayed, then renormalized histogram frequencies, including a central bar of 44 rather than the intended 80 in distribution 3. The repository's data follow that displayed-frequency convention. [Koudenburg, Kiers, and Kashima, 2021](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.738258/full).

Using those repository histograms and aggregate scores, I independently computed the following results with a separate bisection calculation and pair-count definition of Kendall's \(\tau_B\). The median case was also calculated with exact rational masses.

| Measure | Independently obtained \(\tau_B\) |
| --- | ---: |
| MEC(2, 1.15) | 0.8857142857 |
| ER(0.8) | 0.8857142857 |
| MEC(2, 1.2) | 0.8666666667 |
| MEC(2, 2) | 0.8095238095 |
| MEC(1, 1), exact weighted-median calculation | 0.7655590024 |

These calculations support the headline concordance result. They do not reproduce the entire corpus analysis, recover original participant-level data, or establish performance on real-world surveys.

The repository's saved parameter search uses a finite coarse grid and a local refined grid. It reports many maximizers with the same Kendall value and uses additional criteria to recommend (2.024, 1.178), which ties the manuscript's simpler (2, 1.15) in Kendall concordance. The coarse search includes parameter values outside the manuscript's theorem domain. “Exhaustive” must therefore be tied to a specified grid and admissible domain, not the whole continuous parameter family.

The reported bootstrap and leave-one-out calculations assess fixed candidates already selected using the full benchmark. They characterize stability on these fifteen cases; they are not independent test-set validation of parameter selection. The rank score is constant whenever parameter changes preserve all pairwise orderings, which explains why many settings can tie.

For \(\beta>1\), theoretical computation can use the monotone derivative

\[
\beta\sum_i p_i^\alpha\operatorname{sgn}(y-x_i)|y-x_i|^{\beta-1}
\]

and bisection, at \(O(n\log(1/\varepsilon))\) arithmetic work for position accuracy on a bounded interval. For \(\beta=1\), use weighted medians; for \(\beta=2\), use weighted moments. The public MEC implementation inspected uses bounded scalar minimization and returns the minimum cost; a separate helper implements derivative bisection. It does not expose the optimizing consensus position in the public return value examined here.

## Work beyond the supplied manuscript

The public repository provides evidence of additional computational directions: generalized alienation functions for MEC and ER, BiPol, classification thresholds using percentiles and clustering, normalization studies, behavior near \(\beta=1\), and timing comparisons with increasing support size. These are candidate components of the wider thesis history, not yet confirmed thesis contributions.

In particular, the saved generalized-measure leaderboard contains settings exceeding 0.8857 on the same benchmark. Thus the manuscript's “best” claim must remain limited to its stated candidate set. Generalized implementations also require their own mathematical assumptions; the existing power-family theorems cannot be automatically applied to any user-defined alienation function.

The timing artifacts distinguish a small fixed scale from growing support. On the stored single-worker five-position corpus runs, ER was faster than MEC. With large support, the stored direct pairwise ER implementation became slower. These observations concern specific implementations and hardware. A sorted one-dimensional ER computation can also exploit prefix sums; the quadratic direct implementation is not a general lower bound.

## Scope and thesis implications

The supplied manuscript establishes a mathematical measurement framework and reports synthetic comparative validation. It does not supply a dynamic theory of persuasion, an empirically measured negotiation cost, a network polarization model, a multidimensional extension, or a demonstrated mediation intervention. Those may be future directions, but they are not current established results.

The absence of a group at some survey category is also different from removing that category from the scale. For \(\alpha>1\), grouping and fragmentation affect effective mass, so binning, treatment of almost coincident opinions, and measurement noise deserve explicit methodological discussion. The opinion axis and its normalization must be fixed consistently across compared populations.

For a thesis covering all completed research, the next materials should establish which repository experiments and other outputs belong to the work, the chronology and exact versions, the student's contribution to each component, and the institutional scope. These facts are deliberately left open pending the user's promised resources and objectives.

## Reading map

The manuscript's progression is: introduction and motivation; definitions of distributions and polarization; definition of MEC; convexity, dispersion, symmetry, and scaling; Minority Principle and tipping points; maximality via outward movement; comparison with ER axioms; computational and expert comparisons; conclusions and open directions.

Appendix A supports the transportation identities. Appendix B supports the structural properties. Appendix C establishes the two-group effort comparison. Appendix D establishes outward movement and maximality. Appendix E proves the axiomatic results. Appendix F supplies the algebraic transfer formulas and tipping-point uniqueness argument.

Use the accompanying verification record to distinguish confirmed numerical discrepancies, missing assumptions, interpretation limits, and evidence that remains to be reproduced.
