# Measures

A Python package for computing various polarization measures.

## Installation

```bash
pip install git+https://github.com/Ulvenforst/pol_measures.git
```

---

A distribution is denoted by a pair of $n$-dimensional vectors $(\vec{x},\vec{\pi})=((x_1,\dots,x_n),(\pi_1,\dots,\pi_n))\in\mathbb{R}^n\times\mathbb{R}_{+}^n$ for some $n\in\mathbb{N}$, where $\pi_i$ is the population of individuals with attribute $x_i$ and $\sum_{i=1}^n\pi_i=1$. $x_i\neq x_j$ for distict $i,j\in\{1,\dots,n\}$. For normalization purposes $0\leqslant x_i\leqslant 1$.

$$
\mathscr{D}\equiv\bigcup_{n=1}^{\infty}{\mathbb{R}_{+}^n\times\{\vec{x}\in\mathbb{R}^n: x_i\neq x_j\ \text{for all distict}\ i,j\in\{1,\dots,n\} \}}
$$

A polarization measure is a function $P:\mathscr{D}\to\mathbb{R}_+$.

We are currently studying with polarization measures:

*   $\mathscr{Comete}_{\alpha,\beta}(M)$
*   $\mathrm{EMD}_\text{pol}(M)$
*   $\text{Experts}(M)$
*   $\text{BiPol}(M)$
*   $\mathrm{ER}_{\alpha}(M)$
*   $\text{Shannon}_{pol}(M)$
*   $\text{Eijk}_\text{pol}(M)$

Where $M=(\vec{x},\vec{\pi}_m)\in\mathscr{D}$.
