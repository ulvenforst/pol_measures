# Measures

A Python package for computing various polarization measures in PROMUEVA.

## Installation

```bash
pip install git+https://github.com/Ulvenforst/pol_measures.git
```

If a new update is generated, you can uninstall the package and then reinstall it:
```bash
pip uninstall measures
```
---

A distribution is denoted by a pair of $n$-dimensional vectors $`(\vec{x},\vec{\pi})=((x_1,\dots,x_n),(\pi_1,\dots,\pi_n))\in\mathbb{R}^n\times\mathbb{R}_{+}^n`$ for some $`n\in\mathbb{N}`$, where $`\pi_i`$ is the population of individuals with attribute $`x_i`$ and $`\sum_{i=1}^n\pi_i=1`$. $`x_i\neq x_j`$ for distict $`i,j\in\{1,\dots,n\}`$. For normalization purposes $`0\leqslant x_i\leqslant 1`$.

```math
\mathscr{D}\equiv\bigcup_{n=1}^{\infty}{\mathbb{R}_{+}^n\times \{\vec{x}\in\mathbb{R}^n: x_i\neq x_j\ \text{for all distict}\ i,j\in\{1,\dots,n\} \}}
```

A polarization measure is a function $P:\mathscr{D}\to\mathbb{R}_+$.

This package includes the following polarization measurements:

*   $\mathrm{Comete}_{\alpha,\beta}(M)$ (Implemented by Carlos Pinzón)
*   $\text{BiPol}(M)$ (Proposed and implemented by Carlos Pinzón, Sept 2023)
*   $\mathrm{EMD}_\text{pol}(M)$
*   $\text{Experts}(M)$
*   $\mathrm{ER}_{\alpha}(M)$
*   $\text{Shannon}_{pol}(M)$
*   $\text{Eijk}_\text{pol}(M)$

Where $M=(\vec{x},\vec{\pi}_m)\in\mathscr{D}$.

## Examples of use measures
Let $M=(\vec{x},\vec{\pi})\in\mathscr{D}$.
The measures are divided into two subdirectories, those of literature and proposals:

### Literature `measures.metrics.literature`
**EMDPol:**
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
```math
\mathrm{EMD}_\text{pol}(M)=0.5-\mathrm{EMD}(M,\pi_{\max})
```

Where $Q=(\vec{y},\vec{\pi}_Q)\in\mathscr{D}$, then
```math
\mathrm{EMD}(M,Q)=\inf_{\gamma \in \Pi} \, \sum\limits_{x,y} \gamma (x,y)\Vert x - y \Vert
```

and $`\pi_{\max}=((x_1,\dots,x_n),(0.5,\dots,0.5))\in\mathscr{D}`$

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
```math
\text{ER}(M)=K \sum _{i=1}^n \sum _{j=1}^n \pi _{i}^{1+\alpha}\pi _{j} |x_i-x_j|
```
Where by default $K = 1 / (2(0.5^{2 + \alpha}))$.


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
```math
\mathrm{Experts}(M)=\frac{2.14\pi_2\pi_4 + 2.70(\pi_1\pi_4 + \pi_2\pi_5)+3.96\pi_1\pi_5}{0.0099\left(\sum_{i=1}^n\pi_i\right)^2}
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
```math
\mathrm{Shannon}_{\text{pol}}(M) = -\sum_{i=1}^{n} \pi_i \log_2 \left( 1 - \frac{|x_i - \mu_\vec{x}|}{x_{\max} - x_{\min}} \right)
```

where $\mu_\vec{x}=\sum_{i=1}^{n}{\pi_i x_i}$ is the mean of $\vec{x}$.

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
```math
\mathrm{Eijk}_{\text{pol}}(M) = 1- \sum_{i} w_i \cdot A_i
```
Where $w_i$ is the proportion of cases in the empirical distribution contained in layer $i$, and $A_i$ is the agreement for layer $i$, calculated using
```math
A = U \cdot \left(1 - \frac{S - 1}{|\vec{x}| - 1}\right)
```
Where $S$ is the number of non-empty categories. For any pattern containing both 0's and 1's:

```math
U = \frac{(|\vec{x}| - 2) \cdot TU - (|\vec{x}| - 1) \cdot TDU}{(|\vec{x}| - 2) \cdot (TU + TDU)}
```

Here, $TU$ is the number of triplets of categories conforming to unimodality, $TDU$ is the number of triplets deviating from unimodality.

For patterns consisting only of 1's (where $TU = TDU = 0$):

```math
U = 1
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

Comète is defined as the minimum effort of carrying out a distribution $M$ towards the distribution with a single point of consensus $p=((x_p),(\pi_p))$, where $p\in\mathscr{D}$.

```math
\mathrm{Comete}_{\alpha,\beta}(M) = \min_{x_p} \sum_{i=1}^n \pi_{i}^\alpha |x_i-x_p|^\beta
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
```math
\mathrm{BiPol}(M) := 4 \max_{A \cap B=\emptyset, A \cup B={x_1,...,x_n}} \dfrac{1}{n^2} \sum_{x \in A} \sum_{y \in B} |y-x|
```
