# Verification record and questions for the thesis

**Version notice:** This is the initial draft-review record, retained for traceability. The completed-paper reference is [arXiv v1](arxiv-v1-thesis-reference.md). Findings below refer to the earlier mirrored sources unless their applicability is established in the following update; this is not a correction request from the user.

## Update after reading arXiv v1

The supplied archive now provides the authoritative manuscript version, section and appendix directories, and figure assets. The final paragraph's warning about the flattened mirror therefore describes that mirror only, not the newly supplied archive.

The experimental TWD counterexample claim and VdE compliance claim discussed below were removed in v1. Those particular objections must not be presented as objections to the current experimental discussion. Most theorem statements, example formulas, and proof text are unchanged, so the remaining mathematical observations require individual follow-up rather than an assumption that all were resolved.

The ER central-split example now uses positions (0,0.4,0.5,0.6,1), whereas the initial calculation below used equally spaced positions. Under normalization by the balanced endpoint value, K=2^(1+0.8), direct evaluation for the new coordinates gives 0.5371745887 before and 0.4279435492 after the split. The paper reports approximately 0.53 and 0.44. The qualitative decrease remains valid; the new numerical value should be reconciled with the intended normalization and coordinates before reuse. The earlier 0.4481836437 calculation does not verify the revised instance.

14 September 2026. This record accompanies the research-context document. No supplied manuscript files were changed. A confirmed local error does not imply that the main research construction fails; each finding below identifies its actual scope.

## Results independently checked

The stars-and-bars and reflection counts are 4,598,126 total distributions, 1,326 palindromes, and 2,299,726 mirror representatives. The example comparing masses (40,60) with (40,59,1) reproduces the intended reversal: MEC(1,2) increases from 6 to 6.161875, whereas MEC(2,2) decreases from 276.9230769231 to 274.2065254821. The transfer threshold is 17.1559796441. Three exact rational substitutions into the general tipping-point identity agree with its stated cubic expression; these are algebraic spot checks, not a symbolic proof of the identity or its discriminant formula.

On the fifteen benchmark histograms stored in the referenced repository, an independent calculation gives Kendall's tau-B of 0.8857142857 for both MEC(2,1.15) and ER(0.8). The exact weighted-median baseline gives 0.7655590024. The normalized ER central-split example decreases from 0.5371745887 to 0.4481836437. These results support the corresponding manuscript conclusions, subject to precision and provenance distinctions below.

The calculations used separate arithmetic and bisection routines, rather than executing the repository's measure implementations. Full-corpus correlations, all parameter searches, and timing experiments were not rerun.

## Confirmed numerical and statement issues

**Whole-group transfer value.** Section 5.2 gives approximately 461.5 when all 60 units move from 0.75 to 1, while 40 units remain at 0.25. The correct value is

\[
\operatorname{MEC}_{2,2}((40,60),(0.25,1))
=\frac{40^2 60^2(1-0.25)^2}{40^2+60^2}
=\frac{8100}{13}\approx623.0769231.
\]

The qualitative increase remains true. [Source](/Users/ulvenforst/.codex/.chatgpt-projects/g-p-679762270ce0819184d0049533dc560a/sources/05-theoretical-applications.tex).

**Alienation example silently changes mass weighting.** For masses (49,52) at (0,1), the quoted quadratic optimizer 2704/5105 assumes alpha=2. If alpha=1, the mean is 52/101. The example introduces beta before introducing the mass-power parameter, so alpha must be specified. The smaller group is 49/101 of the population, approximately 48.51%, rather than exactly 49%. [Source](/Users/ulvenforst/.codex/.chatgpt-projects/g-p-679762270ce0819184d0049533dc560a/sources/03-mec-measure.tex).

**Strict outward movement omits a non-consensus condition.** Theorem 7 in the PDF, restated as `shiftingAway`, allows a distribution consisting of a single mass at a when beta>1. Moving that mass to b satisfies the written definition of an outward shift and is not a median-shift, yet both values of MEC are zero. Require at least two occupied opinions. In the proof, the function g is strictly increasing only if at least one other positive mass remains. The maximality proof already begins its substantive argument with non-consensus distributions, so this defect in the standalone statement need not invalidate that application. [Statement](/Users/ulvenforst/.codex/.chatgpt-projects/g-p-679762270ce0819184d0049533dc560a/sources/06-shifting-away.tex), [proof](/Users/ulvenforst/.codex/.chatgpt-projects/g-p-679762270ce0819184d0049533dc560a/sources/D-polarization-measure-proof.tex).

**Two-group hypotheses should be explicit in the formal statements.** The Minority Principle and preceding lemmas specify positive masses at l and r but do not explicitly exclude additional support. Their arguments and conclusions require support exactly {l,r}. The surrounding prose supplies the intention; the theorem statements should carry the assumption themselves. [Source](/Users/ulvenforst/.codex/.chatgpt-projects/g-p-679762270ce0819184d0049533dc560a/sources/05-theoretical-applications.tex).

**Endpoint collision in Property 1.** The property permits z=x, but its displayed five-position representation then repeats the two endpoints. Under the distribution-as-function definition, coincident masses must be added before exponentiation. The proof's increment 2 Delta^alpha z^beta applies when 0<z<x. At z=x, the increment is instead

\[
2\big((p+\Delta)^\alpha-p^\alpha\big)x^\beta>0.
\]

The property still holds, but its boundary case needs separate notation and evaluation. [Statement](/Users/ulvenforst/.codex/.chatgpt-projects/g-p-679762270ce0819184d0049533dc560a/sources/07-er-comparison.tex), [proof](/Users/ulvenforst/.codex/.chatgpt-projects/g-p-679762270ce0819184d0049533dc560a/sources/E-er-axioms-proofs.tex).

**The conclusion misstates the extremal-distance identity.** MEC(1,1) equals M/2 minus EMD to the extremal distribution; it does not equal EMD to that distribution. Section 3 states the identity correctly, while the conclusion's wording drops the complement. [Source](/Users/ulvenforst/.codex/.chatgpt-projects/g-p-679762270ce0819184d0049533dc560a/sources/09-conclusion.tex).

## Proof exposition that needs alignment

The section 6 proof sketch claims a shared median before and after movement for beta=1. For alpha>1 with merging, this need not hold for the actual transformed distributions. For example, masses (1,3,2,2) at (0,0.4,0.6,1), with alpha=2, have optimizer interval [0.4,0.6]. Moving the whole mass at 0.6 to 1 gives masses (1,3,4), whose squared-mass optimizer is uniquely 1. The costs increase from 3.6 to 6.4, but no common optimizer remains. Appendix D's intermediate transformed-mass comparison is the appropriate route.

That comparison uses (u+v)^alpha >= u^alpha+v^alpha. Increasing an objective pointwise cannot decrease its minimum. This establishes the required nondecrease without invoking an envelope derivative at every point, which is delicate when beta=1 and minimizers can change.

The section 6 description of the endpoint maximum also differs from Appendix D: the body describes proving monotonicity in endpoint mass, whereas the appendix gives a product bound. Both can support maximality, but the exposition should describe the proof actually supplied.

The power-distribution definition permits alpha=0, although the subsequent research assumes alpha>=1. At zero mass, 0^0 needs a convention, and applying a value of 1 outside the original support would violate finite support. Restrict this ancillary definition to the needed positive-exponent domain or define it on the support explicitly.

The quantile proof's endpoint conventions at q=M/2 should be interpreted almost everywhere. Its integral identity is unaffected by a single boundary value. A distribution is also an intermediate measure in a transport path, rather than itself a transportation plan.

## Computational complexity: what can be claimed

The bisection bound O(n log(1/epsilon)) is justified for beta>1, with epsilon measuring accuracy of the consensus position on a fixed bounded interval. A claim about objective-value accuracy additionally requires a bound connecting position error to objective error. With S_alpha=sum p_i^alpha on [0,1], the elementary Lipschitz bound beta S_alpha |y-z| is one available connection.

The manuscript compares this with direct quadratic ER evaluation. That is the complexity of the inspected implementation, not an inherent bound for one-dimensional ER. To see this, define, on sorted opinions,

\[
L_i=\sum_{j<i}p_j,\quad L_i^x=\sum_{j<i}p_jx_j,
\quad R_i=\sum_{j>i}p_j,\quad R_i^x=\sum_{j>i}p_jx_j.
\]

Then

\[
\sum_j p_j|x_i-x_j|=x_iL_i-L_i^x+R_i^x-x_iR_i.
\]

All these quantities can be obtained by prefix and suffix sums. Substitution into ER gives O(n) arithmetic work on sorted support, or O(n log n) including sorting. This is a direct algebraic observation made in this review. It does not automatically apply to every generalized alienation function.

Stored single-worker corpus timings report about 159.67 seconds for MEC and 6.11 seconds for ER across the 2,299,726 five-position representatives. Stored growing-support runs show the direct pairwise ER implementation eventually becoming slower. These are observations from saved outputs, not fresh timings. The thesis should distinguish corpus size, support size, implementation choice, hardware, and numerical precision.

The public MEC class uses bounded scalar optimization. A separate bisection helper exists for beta>1. The helper and the public class should not be described as a single algorithm. The public class returns the minimum value in the inspected version; it does not return the consensus optimizer despite that object's theoretical importance.

## Empirical evidence and limits

**Analyzed sample.** The primary Koudenburg article states that two of the 60 experts were excluded and that subsequent analyses use 58. The thesis should distinguish recruitment from analysis and verify the aggregate-score provenance. The repository contains aggregate values, not the participant-level responses reviewed here. [Primary study](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.738258/full).

**Version and tie sensitivity.** A saved repository table gives different correlations for EMD and MEC(1,1), although they are mathematically equivalent on normalized, equally spaced five-position histograms. The exact median calculation reproduces 0.7655590024. Approximate minimization can split genuine ties; rounding can introduce ties; applying a monotone normalization before truncation can change the resulting tie pattern. Thus mathematical rank invariance does not justify treating all finite-precision pipelines as interchangeable.

**Finite search versus continuous optimum.** The stored coarse search covers alpha from 0.5 to 3 and beta from 0.8 to 2 in steps of 0.01; a refined search explores a smaller rectangle. Neither constitutes an exhaustive search over the continuous admissible family. Some searched values also fall outside alpha,beta>=1. The largest observed Kendall value can be reported for a specified search; global optimality requires another argument.

**Calibration and validation.** Selecting parameters and reporting agreement on the same fifteen histograms measures in-sample agreement. The stored resampling evaluates fixed candidates chosen using the full data; it does not rerun parameter selection inside training folds. This limits claims about generalization. It does not negate the observed concordance.

**Axioms and finite counterexample searches.** Not finding a violation on the five-category corpus does not prove universal axiomatic compliance. Section 8 moves from “no counterexamples found” for Van der Eijk to language asserting compliance. The evidence supplied here does not bridge that gap. Axiom 1 also has existential neighborhood constants: an arbitrary finite fusion with the opposite inequality does not by itself refute that axiom. The reported TWD comparison requires the distributions, quantifier argument, and search constraints before it can serve as a rigorous disproof.

**Symmetry reduction changes sampling weights.** Counting one item per reflection orbit is appropriate for unique configurations. If the desired correlation is over all original histograms with equal weight, size-two orbits and palindromes must have different weights. The canonical and original universes should not be claimed to yield identical correlation automatically.

**Scope of superiority.** The paper compares a specific candidate set. The repository's separate generalized-measure leaderboard reports about 0.90476 for two generalized ER settings. This is additional exploratory evidence requiring its own provenance and theory, but it rules out casually generalizing the paper's best-in-comparison statement to all implemented or possible measures.

## Implementation conventions to preserve in future work

The inspected public API rescales the supplied opinion range to [0,1] by default and does not normalize weights by default. On a fixed five-category scale spanning both endpoints this matches the intended embedding. If only occupied positions are passed, range rescaling can instead change the measurement domain from one distribution to another.

`MECNormalized` uses a beta-root ratio against a unit-mass endpoint reference. Its description as a population-independent index requires appropriately normalized masses. It should be distinguished from raw MEC and from direct division by the maximum for the original total mass.

The EMD implementation used by the comparison code assumes equally spaced categories and uses category indices rather than arbitrary supplied distances. Its equivalence with MEC(1,1) should be stated for the actual input convention. The public API also requires at least two array positions, although mathematical consensus can be represented by one occupied position on a larger fixed grid.

GeneralizedMEC accepts user-defined nonnegative alienation functions. That alone does not guarantee convexity, zero consensus cost, uniqueness, or global minimization by the bounded solver. The power-family theorem assumptions must be re-established for extensions.

## Conceptual boundaries and unresolved historical questions

The minority result concerns modeled total effort at the optimum. It neither measures observed psychological burdens nor recommends that a minority conform. Likewise, an optimizing consensus is not automatically feasible, fair, stable under opinion dynamics, or accepted by participants.

The tipping point is a crossing of the original cost level. Its equality condition Delta(k)=0 differs from the turning-point condition Delta'(k)=0. General cubic representation should also remain distinct from a proved unique crossing.

For alpha>1, fragmentation is sensitive to how a distribution is grouped. Two equal groups approaching the same non-consensus opinion contribute 2q^alpha d^beta before merging and (2q)^alpha d^beta after merging. This raises continuity and measurement-resolution questions for any extension beyond a fixed discrete scale.

The thesis will need the authoritative manuscript version, experiment-generating commits, raw and processed data provenance, remaining figure assets, the record of other completed studies, individual contribution attribution, and institutional requirements. These are matters to establish from the user's forthcoming resources, not assumptions to fill in.

The mirrored LaTeX files are flattened, while the main file refers to `sections/`, `appendix/`, and `Images/`. Standalone image assets are absent from the supplied mirror. The PDF preserves their appearance, but the mirror is not a complete directly buildable manuscript project. Rebuilding or editing a thesis source tree is separate work.
