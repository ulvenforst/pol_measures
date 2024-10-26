# Measures

A Python package for computing various polarization measures.

## Installation

```bash
pip install git+https://github.com/Ulvenforst/pol_measures.git
```

If a new update is generated, you can uninstall the package and then reinstall it:
```bash
pip uninstall measures
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

## Examples of use measures
The measures are divided into two subdirectories, those of literature and proposals:

### Literature `measures.metrics.literature`
**EMD_pol:**
```python
   from measures.metrics.literature import EMDPol

   # Create measure instance
   emd = EMDPol()
   
   # Define test cases
   x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
   
   # Example: Uniform distribution
   w1 = np.ones(5) / 5
   print(f"EMD: {emd(x, w1):.6f}")
```

**EstebanRay:**
```python
   from measures.metrics.literature import EstebanRay

   # Create measure instance
   er = EstebanRay(alpha=1.0) # If alpha is not indicated, the default is 1.6
   
   # Define test cases
   x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
   
   # Example: Uniform distribution
   w1 = np.ones(5) / 5
   print(f"ER: {er(x, w1):.6f}")
```

**Experts:**
```python
   from measures.metrics.literature import Experts

   # Create measure instance
   expert = Experts()

   # Define test cases
   x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
   
   # Example: Uniform distribution
   w1 = np.ones(5) / 5
   print(f"ER: {expert(x, w1):.6f}")
```

**ShannonPol:**
```python
   from measures.metrics.literature import ShannonPol

   # Create measure instance
   shannon = ShannonPol()

   # Define test cases
   x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
   
   # Example: Uniform distribution
   w1 = np.ones(5) / 5
   print(f"ER: {shannon(x, w1):.6f}")
```

**VanDerEijkPol:**
```python
   from measures.metrics.literature import VanDerEijkPol

   # Create measure instance
   eijkpol = VanDerEijkPol()

   # Define test cases
   x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
   
   # Example: Uniform distribution
   w1 = np.ones(5) / 5
   print(f"ER: {eijkpol(x, w1):.6f}")
```

### Proposed `measures.metrics.proposed`
**Comete:**
```python
   from measures.metrics.proposed import Comete

   # Create measure instance
   comete = Comete(alpha=1.0, beta=1.0)

   # Define test cases
   x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
   
   # Example: Uniform distribution
   w1 = np.ones(5) / 5
   print(f"ER: {comete(x, w1):.6f}")
```

**BiPol:**
```python
   from measures.metrics.proposed import BiPol

   # Create measure instance
   bipol = BiPol()

   # Define test cases
   x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
   
   # Example: Uniform distribution
   w1 = np.ones(5) / 5
   print(f"ER: {bipol(x, w1):.6f}")
```
