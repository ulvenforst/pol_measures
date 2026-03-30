\subsection{Population Scaling}

The following theorem states a simple homogeneity property of the minimal effort to consensus: scaling all population masses by a factor $\lambda$ scales the value of $\MEC$ by $\lambda^{\alpha}$.

\begin{theorem}[Population Scaling] \label{th:multbyscalar}
Let $\pi=((\pi_1,\ldots,\pi_n),(x_1,\ldots,x_n))\in\Dset$ and let $\lambda>0$.
We define
\(
\lambda\pi \defsymbol ((\lambda\pi_1,\ldots,\lambda\pi_n),(x_1,\ldots,x_n)).
\)
Then
\(
\MEC(\lambda\pi)=\lambda^{\alpha}\MEC(\pi).
\)
\end{theorem}

\begin{proof}
By definition,
\[
\MEC(\lambda\pi)
=
\min_{y\in[0,1]}
\sum_{i=1}^{n}(\lambda\pi_i)^{\alpha}|x_i-y|^{\beta}.
\]
Since
\[
(\lambda\pi_i)^{\alpha}=\lambda^{\alpha}\pi_i^{\alpha},
\]
it follows that
\[
\MEC(\lambda\pi)
=
\min_{y\in[0,1]}
\lambda^{\alpha}\sum_{i=1}^{n}\pi_i^{\alpha}|x_i-y|^{\beta}.
\]
Because $\lambda^{\alpha}$ does not depend on $y$, it can be factored out of the minimum:
\[
\MEC(\lambda\pi)
=
\lambda^{\alpha}
\min_{y\in[0,1]}
\sum_{i=1}^{n}\pi_i^{\alpha}|x_i-y|^{\beta}.
\]
Therefore,
\[
\MEC(\lambda\pi)=\lambda^{\alpha}\MEC(\pi).
\]
\end{proof}

As an immediate consequence of the previous theorem we obtain
Condition~H of Esteban and Ray \cite{esteban1994measurement}. 
A polarization measure $\Pol$ satisfies Condition~H if
\[
\Pol(\pi)>\Pol(\pi') \Rightarrow \Pol(\lambda\pi)>\Pol(\lambda\pi')
\]
for every $\lambda>0,\pi,\pi' \in \Dset$. Condition~H expresses invariance to population scaling and is
standard in the theory of inequality measurement (see, e.g., \cite{foster85}). 

Another simple but useful application of the proposition is to obtain the polarization of a normalized distribution from that of the original distribution. This is especially useful if you want to compare populations of different sizes.  

\begin{corollary} 
The following hold: 
\begin{enumerate}
\item $\MEC(\overline{\pi})=\frac{\MEC(\pi)}{(\Mass{\pi})^{\alpha}}.$
\item The $\MEC$ satisfies Condition~H.
\end{enumerate}
\end{corollary}
\begin{proof}
(1) By definition of normalization,
\[
\overline{\pi}=\frac{1}{\Mass{\pi}}\,\pi.
\]
Applying Th.~\ref{th:multbyscalar} with
\[
\lambda=\frac{1}{\Mass{\pi}},
\]
we obtain
\[
\MEC(\overline{\pi})
=
\MEC\!\left(\frac{1}{\Mass{\pi}}\pi\right)
=
\left(\frac{1}{\Mass{\pi}}\right)^\alpha \MEC(\pi).
\]
Therefore,
\[
\MEC(\overline{\pi})
=
\frac{\MEC(\pi)}{(\Mass{\pi})^\alpha},
\]
as claimed.

(2) From Th.~\ref{th:multbyscalar} we have
\[
\MEC(\lambda\pi)=\lambda^{\alpha}\MEC(\pi)
\qquad\text{and}\qquad
\MEC(\lambda\pi')=\lambda^{\alpha}\MEC(\pi').
\]
Since $\lambda^{\alpha}>0$, multiplying the inequality
\[
\MEC(\pi)>\MEC(\pi')
\]
by $\lambda^{\alpha}$ preserves the order, yielding
\[
\MEC(\lambda\pi)>\MEC(\lambda\pi').
\]
Hence MEC satisfies Condition~H.
\end{proof}
 
 


\subsection{A Minority Principle}
 Let us consider two-point opinion distributions 
supported on the extremes $\{0,1\}$ where one mass is smaller than the other, for example 
 $\pi=((10,90),(0,1)).$ 
  One might expect that the share of the total $(\alpha,\beta)$-effort of each mass required to reach an optimal consensus $y^*$ for $\pi$ should depend on the parameters $\alpha$
  and $\beta$. In particular, it would seem natural that, by varying these
parameters, one could shift the burden of adjustment between the masses:
for some values, the larger mass might bear most of the effort, while
for others, the smaller mass might.
  
From the definition of $\EC$ the effort of the smaller mass to $y^*$ is $10^\alpha {y^*}^\beta$ while for the larger mass we obtain $90^\alpha (1-y^*)^\beta$. If $\beta=1$ the larger mass does not move at all, since  from Cor.~\ref{cor:mec-special-cases}, we have $y^* = 1$. Thus, the smaller mass bears the whole effort.  The situation is more subtle when $\beta>1$. In this case, the optimal
consensus point lies in the interior $(0,1)$, and thus both masses contribute to the
effort.  

Nevertheless, and somewhat surprisingly, it is actually the smaller mass that \emph{always} makes the greater effort to reach the optimal consensus, regardless of $\alpha$ and $\beta$. This is established below for any two-point distribution $\pi$ supported on $\{0,1\}$.

First the following lemma confirms the intuition that the optimal consensus point lies closer to the larger mass.

\begin{lemma}\label{lem:yopt-closest-to-largest}
Assume \(\beta>1\). For $A>0$, let \(\pi_A\in\Dset\) satisfy
\(\pi_A(0)=A\) and \(\pi_A(1)=\Mass{\pi_A}-A\),
and let \(y^*\) be such that
\(\MEC(\pi_A)=\EC[\alpha,\beta][\pi_A](y^*)\).   
(1) If 
\(
y^*>\tfrac12 \) then \( \Mass{\pi_A}-A>A,
\) (2) if 
\(
y^*<\tfrac12 \) then \( \Mass{\pi_A}-A<A
\),
and (3) if 
\(
y^*=\tfrac12 \) then \( \Mass{\pi_A}-A=A.
\)
\end{lemma}
\begin{proof}
	We show (1), the other cases are obtained similarly. 
	
	Since $\EC[\alpha,\beta][\pi_A]$ is a convex function on $y$, then $\frac{\partial\EC[\alpha,\beta][\pi_A]}{\partial y}$ is monotonically increasing. Then, $y^* > 0.5$ implies $0=\frac{\partial\EC[\alpha,\beta][\pi_A]}{\partial y}(y^*) > \frac{\partial\EC[\alpha,\beta][\pi_A]}{\partial y}(\tfrac12)$, which in turn means
	
	\[0 > A^\alpha\cdot\beta(\tfrac12)^{\beta-1} - (\Mass{\pi_A}-A)^\alpha\cdot\beta (\tfrac12)^{\beta-1}\]
	\[(\Mass{\pi_A}-A)^\alpha\cdot\beta (\tfrac12)^{\beta-1} > A^\alpha\cdot\beta(\tfrac12)^{\beta-1} \]
	%\[(\Mass{\pi_A}-A)^\alpha > A^\alpha \]
	\[(\Mass{\pi_A}-A) > A\]

    Observe that when the same reasoning is applied to $y^* < \tfrac12$ then we can conclude $(\Mass{\pi_A}-A) < A$ (2). Furthermore, when the same reasoning is applied to $y^* = \tfrac12$ then we can conclude $(\Mass{\pi_A}-A) = A$ (3).
\end{proof}

Having located the side of the optimal consensus point relative to the two masses, we now compare their respective contributions to the total effort.

\begin{lemma}\label{lem:greatest_contribution_from_smallest}
	Under the same assumptions of Lem.~\ref{lem:yopt-closest-to-largest}, we have: (1) if $y^*>\tfrac12$ then $A^\alpha(y^*)^\beta > (\Mass{\pi_A}-A)^\alpha(1-y^*)^\beta$, (2) if $y^*<\tfrac12$ then $A^\alpha(y^*)^\beta < (\Mass{\pi_A}-A)^\alpha(1-y^*)^\beta$, and (3) if $y^* = \tfrac12$ then $A^\alpha(y^*)^\beta =  (\Mass{\pi_A}-A)^\alpha(1-y^*)^\beta$, 
\end{lemma}
\begin{proof}
	We show (1). The case (2) is symmetrical and (3) is immediate from Lem.~\ref{lem:yopt-closest-to-largest}. 
	
	Since $\frac{\partial\EC[\alpha,\beta][\pi_A]}{\partial y}(y^*) = 0$, then
	\[A^\alpha\cdot\beta(y^*)^{\beta-1}-(\Mass{\pi_A}-A)^\alpha\cdot\beta(1-y^*)^{\beta-1}=0\]
	\[A^\alpha\cdot(y^*)^{\beta-1}=(\Mass{\pi_A}-A)^\alpha\cdot(1-y^*)^{\beta-1}\]
	As $y^*>\tfrac12$, we can state
	%\[2y^*>1\]
	\[y^*>1-y^*\]
	\[[A^\alpha\cdot(y^*)^{\beta-1}]y^*>[(\Mass{\pi_A}-A)^\alpha\cdot(1-y^*)^{\beta-1}](1-y^*)\]
	\[A^\alpha(y^*)^\beta>(\Mass{\pi_A}-A)^\alpha(1-y^*)^\beta\]
\end{proof}

As an immediate consequence of the above lemmas and Cor.~\ref{cor:mec-special-cases}, we conclude that indeed it is the minority that always bears the greater effort.

\begin{theorem}[Minority Principle]
\label{thm:smaller-mass-greater-effort} For $A >0$, 
let \(\pi_A\in\Dset\) satisfy
\(\pi_A(0)=A\) and \(\pi_A(1)=\Mass{\pi_A}-A\), with
\(A\neq \Mass{\pi_A}-A\). Let \(y^*\) be such that
\(\MEC(\pi_A)=\EC[\alpha,\beta][\pi_A](y^*)\). Then:
\[
A<\Mass{\pi_A}-A
\quad\text{implies}\quad
A^\alpha (y^*)^\beta
>
(\Mass{\pi_A}-A)^\alpha (1-y^*)^\beta,
\]
and
\[
A>\Mass{\pi_A}-A
\quad\text{implies}\quad
A^\alpha (y^*)^\beta
<
(\Mass{\pi_A}-A)^\alpha (1-y^*)^\beta.
\]
\end{theorem}
\begin{proof}
If \(\beta>1\), the result follows directly from
Lemmas~\ref{lem:yopt-closest-to-largest}
and~\ref{lem:greatest_contribution_from_smallest}. If \(\beta=1\), then \(y^*\) is a median of \(\pi_A\) by
Cor.~\ref{cor:mec-special-cases}. Since \(\pi_A\) has unequal masses
at \(0\) and \(1\), then such a median is the position of the larger mass, so only the smaller
mass contributes to the total effort $\EC[\alpha,\beta][\pi_A](y^*)$.
\end{proof}



\subsection{A Tipping Point Method} 

In Ex.~\ref{ex:group-identity}, we illustrated the role of group identity using a distribution $\pi$ and showing that transferring a small portion of mass at 0.75 toward the extreme right may in fact decrease polarization. Such a shift can reduce the effective weight of the original mass and thereby lower the overall level of polarization.

Nevertheless, if $\pi'$ is obtained from $\pi$ in Ex.~\ref{ex:group-identity} by transferring the entire mass
at $0.75$ to the extreme right, polarization does increase. Indeed, we have 
\(
\MEC[2,2](\pi)<\MEC[2,2](\pi') \approx 461.5 .
\)
This naturally suggests the following question: what is the minimal amount of
mass (\emph{critical mass}) that must move from the moderate right (0.75) to the extreme right (1.0) for
polarization to increase? In other words, at what point does such a shift begin
to raise polarization?


We illustrate a simple method to solve the above tipping-point question for a family of distributions and group identity and alienation parameters $\alpha=\beta=2.$ 



\begin{example}[A Tipping Point Method for Mass Shift]\label{ex:tipping-mass-shift}
Let \(m,n>0\) and consider the opinion distribution
\(
\pi=((m,n),(0.25,0.75)).
\)


For \(k\in(0,n]\), let
\(
\pi_{0.75:k\rightarrow1}=((m,n-k,k),(0.25,0.75,1))
\)
be the distribution obtained by moving mass \(k\le n\) from \(0.75\) to \(1\). 

To study how polarization changes under this transfer, define
\[
\Delta(k)\defsymbol \MEC[2,2](\pi_{0.75:k\rightarrow1})-\MEC[2,2](\pi).
\]

A recurring factor in the calculations below is \(C\defsymbol m^2+n^2\). Using Proposition~\ref{prop:mec-mass-shift-formulas}, we have
\[
\MEC[2,2](\pi)=\frac{m^2n^2}{4C},
\]
and
\[
\MEC[2,2](\pi_{0.75:k\rightarrow1})
=
\frac{
k^4-2nk^3+(12m^2+C)k^2-8m^2nk+4m^2n^2
}{
16(2k^2-2nk+C)
}.
\]
Hence
\[
\Delta(k)=\frac{k\,R_{n,m}(k)}{Q_{n,m}(k)},
\]
where
\[
R_{n,m}(k)
=
Ck^3-2nCk^2+\bigl(C^2+12m^4+4m^2n^2\bigr)k-8m^4n,
\]
and
\[
Q_{n,m}(k)=16C(2k^2-2nk+C).
\]

Since \(Q_{n,m}(k)>0\) for all \(k\in[0,n]\), the sign of \(\Delta(k)\) for
\(k>0\) is determined by \(R_{n,m}(k)\). Moreover,
\[
R_{n,m}(0)=-8m^4n<0,
\qquad
R_{n,m}(n)=5m^2nC>0.
\]
By Proposition~\ref{prop:unique-root-tipping}, the polynomial \(R_{n,m}\) has a
unique root \(k^\star\in(0,n)\). Therefore
\[
\MEC[2,2](\pi_{0.75:k^\star\rightarrow1})=\MEC[2,2](\pi),
\]
and
\[
\MEC[2,2](\pi_{0.75:k\rightarrow1})<\MEC[2,2](\pi)
\quad\text{for }0<k<k^\star,
\]
while
\[
\MEC[2,2](\pi_{0.75:k\rightarrow1})>\MEC[2,2](\pi)
\quad\text{for }k^\star<k\le n.
\]

Thus \(k^\star\) acts as a tipping point: transferring a small amount of mass
from \(0.75\) to \(1\) initially reduces polarization, but once the transferred
mass exceeds \(k^\star\), the polarization becomes strictly larger than that of
the original distribution \(\pi\).\qed
\end{example}

We conclude this section with an application of the tipping point method to the distribution in 
Ex.~\ref{ex:group-identity}.

\begin{example}
Let
\(
\pi=((40,60),(0.25,0.75)),
\)
so that
\[
\pi_{0.75:k\rightarrow1}=((40,60-k,k),(0.25,0.75,1)).
\]
Applying the analysis of Ex.~\ref{ex:tipping-mass-shift} with \(m=40\) and
\(n=60\), the change in polarization is governed by the cubic polynomial
\[
R_{60,40}(k)
=
5200k^3-624000k^2+80800000k-1228800000.
\]

By Proposition~\ref{prop:unique-root-tipping}, this polynomial has a unique
root in \((0,60)\). Solving \(R_{60,40}(k)=0\) yields
\(
k^\star \approx 17.16.
\)
Therefore
\[
\MEC[2,2](\pi_{0.75:k\rightarrow1})<\MEC[2,2](\pi)
\quad\text{for }0<k<k^\star,
\]
while
\[
\MEC[2,2](\pi_{0.75:k\rightarrow1})>\MEC[2,2](\pi)
\quad\text{for }k^\star<k\le 60.
\]

Thus the polarization becomes strictly larger than that of \(\pi\) once the
transferred mass exceeds approximately \(17.16\).
\end{example}
