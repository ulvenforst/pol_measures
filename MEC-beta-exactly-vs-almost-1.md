```python
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


Position = float
Weight = float
Dist = list[tuple[Position, Weight]]


@dataclass
class DistContext:
    alpha: float
    beta: float
    pi: Dist
    _y_star: float | None

    @property
    def y_star(self) -> float:
        if self._y_star is None:
            res = minimize_scalar(lambda y: EC(self, y))
            self._y_star = float(res.x)  # type: ignore
        return self._y_star

    def move(self, a: float, b: float):
        pi = {x: w for x, w in self.pi}
        pi[b] = pi.get(b, 0) + pi.pop(a, 0)
        pi = sorted(pi.items())
        assert 0 <= pi[0][0] and pi[-1][0] <= 1
        return DistContext(self.alpha, self.beta, pi, None)


def EC(ctx: DistContext, y: float):
    return sum(w**ctx.alpha * (abs(x - y) ** ctx.beta) for x, w in ctx.pi)


def MEC(ctx: DistContext):
    return EC(ctx, ctx.y_star)


alpha = 1
beta = 1.001

w1 = 5 / 7
w0 = w1**beta
a = 7 / 12
wa = 1 / 20

ctx1: DistContext = DistContext(alpha, 1, [(0, w0), (a, wa), (1, w1)], None)
ctx2: DistContext = DistContext(alpha, beta, [(0, w0), (a, wa), (1, w1)], None)

print(np.isclose(ctx1.y_star, a), np.isclose(ctx2.y_star, a))
print(ctx1)
print(ctx2)
print(ctx1.move(a, 0.5))


fig, ax = plt.subplots(3, 2, sharex=True, sharey=False, figsize=(8, 8))

for row in ax:
    row[1].sharey(row[0])

for i in range(len(ax[0])):
    ctx = [ctx1, ctx2][i]
    color = ["blue", "orange"][i]

    ax[0][i].set_title(f"π for β={ctx.beta} (same)")
    ax[0][i].stem(*zip(*ctx.pi), basefmt=" ", linefmt=color)
    ax[0][i].set_xlabel("x")
    ax[0][i].set_ylabel("π(x)")

    ax[1][i].set_title(f"EC(π, y) for β={ctx.beta}")
    y = np.linspace(0, 1, 100)
    ax[1][i].plot(y, [EC(ctx, yi) for yi in y], color=color)
    ax[1][i].set_xlabel("y")
    ax[1][i].set_ylabel("EC(π, y)")

    ax[2][i].set_title(f"MEC(π_a→b) for β={ctx.beta}")
    b = np.linspace(0, 1, 100)
    ax[2][i].plot(b, [MEC(ctx.move(a, bi)) for bi in b], color=color)
    ax[2][i].set_xlabel("b")
    ax[2][i].set_ylabel("MEC(π_a→b)")

    ax[1][i].axvline(ctx.y_star, color="gray", linestyle="dashed", lw=1)
    ax[2][i].axvline(ctx.y_star, color="gray", linestyle="dashed", lw=1)

    for j in range(len(ax[0])):
        ax[1][j].axhline(MEC(ctx), color=color, linestyle="dashed", lw=1, alpha=0.5)
        ax[2][j].axhline(MEC(ctx), color=color, linestyle="dashed", lw=1, alpha=0.5)
plt.tight_layout()
plt.show()

```

> True True
DistContext(alpha=1, beta=1, pi=[(0, 0.714045417402724), (0.5833333333333334, 0.05), (1, 0.7142857142857143)], _y_star=0.5833333308382262)
DistContext(alpha=1, beta=1.001, pi=[(0, 0.714045417402724), (0.5833333333333334, 0.05), (1, 0.7142857142857143)], _y_star=0.583333336633502)
DistContext(alpha=1, beta=1, pi=[(0, 0.714045417402724), (0.5, 0.05), (1, 0.7142857142857143)], _y_star=None)


```python

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


Position = float
Weight = float
Dist = list[tuple[Position, Weight]]


@dataclass
class DistContext:
    alpha: float
    beta: float
    pi: Dist
    _y_star: float | None

    @property
    def y_star(self) -> float:
        if self._y_star is None:
            res = minimize_scalar(lambda y: EC(self, y))
            self._y_star = float(res.x)  # type: ignore
        return self._y_star

    def move(self, a: float, b: float):
        pi = {x: w for x, w in self.pi}
        pi[b] = pi.get(b, 0) + pi.pop(a, 0)
        pi = sorted(pi.items())
        assert 0 <= pi[0][0] and pi[-1][0] <= 1
        return DistContext(self.alpha, self.beta, pi, None)


from measures.metrics.proposed import MEC as MEC_official

def EC(ctx: DistContext, y: float):
    return sum(w**ctx.alpha * (abs(x - y) ** ctx.beta) for x, w in ctx.pi)


def MEC(ctx: DistContext):
    measure = MEC_official(alpha=ctx.alpha, beta=ctx.beta)
    return measure(x=[x for x,w in ctx.pi], weights=[w for x,w in ctx.pi])


alpha = 1
beta = 1.001

w1 = 5 / 7
w0 = w1**beta
a = 7 / 12
wa = 1 / 20

ctx1: DistContext = DistContext(alpha, 1, [(0, w0), (a, wa), (1, w1)], None)
ctx2: DistContext = DistContext(alpha, beta, [(0, w0), (a, wa), (1, w1)], None)

print(np.isclose(ctx1.y_star, a), np.isclose(ctx2.y_star, a))
print(ctx1)
print(ctx2)
print(ctx1.move(a, 0.5))


fig, ax = plt.subplots(3, 2, sharex=True, sharey=False, figsize=(8, 8))

for row in ax:
    row[1].sharey(row[0])

for i in range(len(ax[0])):
    ctx = [ctx1, ctx2][i]
    color = ["blue", "orange"][i]

    ax[0][i].set_title(f"π for β={ctx.beta} (same)")
    ax[0][i].stem(*zip(*ctx.pi), basefmt=" ", linefmt=color)
    ax[0][i].set_xlabel("x")
    ax[0][i].set_ylabel("π(x)")

    ax[1][i].set_title(f"EC(π, y) for β={ctx.beta}")
    y = np.linspace(0, 1, 100)
    ax[1][i].plot(y, [EC(ctx, yi) for yi in y], color=color)
    ax[1][i].set_xlabel("y")
    ax[1][i].set_ylabel("EC(π, y)")

    ax[2][i].set_title(f"MEC(π_a→b) for β={ctx.beta}")
    b = np.linspace(0, 1, 100)
    ax[2][i].plot(b, [MEC(ctx.move(a, bi)) for bi in b], color=color)
    ax[2][i].set_xlabel("b")
    ax[2][i].set_ylabel("MEC(π_a→b)")

    ax[1][i].axvline(ctx.y_star, color="gray", linestyle="dashed", lw=1)
    ax[2][i].axvline(ctx.y_star, color="gray", linestyle="dashed", lw=1)

    for j in range(len(ax[0])):
        ax[1][j].axhline(MEC(ctx), color=color, linestyle="dashed", lw=1, alpha=0.5)
        ax[2][j].axhline(MEC(ctx), color=color, linestyle="dashed", lw=1, alpha=0.5)
plt.tight_layout()
plt.show()

from measures.metrics.literature import EMDPol

print(EMDPol()(x=[x for x,w in ctx1.pi], weights=[w for x,w in ctx1.pi]))
```

> True True
DistContext(alpha=1, beta=1, pi=[(0, 0.714045417402724), (0.5833333333333334, 0.05), (1, 0.7142857142857143)], _y_star=0.5833333308382262)
DistContext(alpha=1, beta=1.001, pi=[(0, 0.714045417402724), (0.5833333333333334, 0.05), (1, 0.7142857142857143)], _y_star=0.583333336633502)
DistContext(alpha=1, beta=1, pi=[(0, 0.714045417402724), (0.5, 0.05), (1, 0.7142857142857143)], _y_star=None)

0.48308903907648426

```python

ctx1 = DistContext(alpha=1, beta=1, pi=[(0, 0.714), (0.583, 0.05), (1, 0.714)], _y_star=None)
print(ctx1.alpha, ctx1.beta)
print(ctx1.y_star)
print(w0**ctx1.alpha * ctx1.y_star**ctx1.beta + w1**ctx1.alpha * (1-ctx1.y_star)**ctx1.beta)
print(EC(ctx1, ctx1.y_star))
print(MEC(ctx1))


```

1 1
0.5830000000123459
0.714145621202928
0.7140000000006173
0.48308529513767157
