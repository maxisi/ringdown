# The marginalized ringdown likelihood

*A derivation of the analytic marginalization over quadrature amplitudes used by
`ringdown`, and a proof that the one-shot expression implemented in `ringdown/model.py` is
algebraically identical to the sequential per-detector scheme it replaced.*

---

## 0. Purpose, scope, and a notational warning

`ringdown` fits a sum of damped sinusoids to post-merger strain data. The model is *linear*
in the sinusoid amplitudes and *nonlinear* in everything else (the black-hole mass and spin,
frequency and damping-rate deviations, the inclination). Because the noise is modelled as
Gaussian and the amplitude prior is Gaussian, the amplitudes can be integrated out
analytically, leaving a much lower-dimensional and much better-conditioned sampling problem
for the nonlinear parameters. This note derives that marginalization from scratch.

Its second purpose is to establish a specific algebraic identity. The marginalization can be
performed **sequentially** — marginalize detector 1, use the resulting posterior on the
amplitudes as the prior for detector 2, and so on, emitting one likelihood factor per
detector — and that is what `ringdown` did until the closed form below replaced it. Sections
4 and 5 prove that the *sum* of those factors equals a single **one-shot** expression built
from two accumulators,

$$
A^{-1} \;=\; \mathbb{1} + \sum_{i=1}^{n_{\rm det}} M_i^\top C_i^{-1} M_i,
\qquad
v \;=\; \sum_{i=1}^{n_{\rm det}} M_i^\top C_i^{-1} y_i ,
$$

plus a single Cholesky factorization. Every symbol is defined below. This note is concerned
only with showing that the two are the same function of the parameters.

> **On the empirical case.** The per-gradient device-time benchmarks on CPU, an RTX A6000 and
> an H100, the independent adversarial re-verification, and the measurements behind rejecting
> the `vmap`/`scan`, backend-dispatch, concatenated-solve and $\Lambda = S^2$ variants were
> recorded in a set of development reports that are not part of the package. See the pull
> request that introduced this note for that record; the conclusions are summarized inline
> below, and the equivalence claims are pinned by `tests/test_model.py`.

### 0.1 Notational warning: $A$ versus $A^{-1}$

> **Read this before anything else.** Following Isi & Farr, arXiv:2005.14199 — whose notation
> the code deliberately reuses — the symbol $A$ denotes the **covariance** of the marginal
> posterior on the quadrature amplitudes, and $A^{-1}$ its **precision** (inverse covariance).
> Likewise $\Lambda$ is the amplitude **prior covariance** and $\Lambda^{-1}$ the prior
> precision. This note keeps that convention throughout, so that a reader can hold this note
> and `ringdown/model.py` side by side: the code's variable `A_inv` is this note's $A^{-1}$,
> and `Lambda_inv` is $\Lambda^{-1}$.
>
> The consequence, which is admittedly awkward, is that the quantity actually *built* and
> *factorized* in the code is $A^{-1}$, not $A$. $A$ itself is never formed. Expressions like
> "the Cholesky factor $R$ of $A^{-1}$" are therefore normal here, not typos.
>
> Where this note departs from arXiv:2005.14199: that paper treats a single data stream, so it
> has no detector index; we add $i$. It also does not need the whitened variables of §5, which
> are introduced here. Finally, some of the development reports referred to above used the
> *opposite* convention, calling the precision $A$; **this note supersedes them
> notationally**, and where they disagree the convention here — which is the code's and the
> paper's — is the one to use.

---

## 1. Setup and notation

### 1.1 Dimensions

| Symbol | Meaning | Production value |
|---|---|---|
| $n_{\rm det}$ | number of detectors (interferometers) | $2$ (H1, L1); range $1$–$3$ |
| $n_t$ | number of time samples analyzed, per detector | $205$; range $\sim 100$–$400$ |
| $n_{\rm mode}$ | number of quasinormal modes in the template | $2$ (the $\ell m n = 220$ and $221$ modes) |
| $n_{\rm quad}$ | number of *quadratures* per mode (§1.3) | $4$ for the generic model; $2$ for the aligned and single-polarization models |
| $k \equiv n_{\rm quad}\, n_{\rm mode}$ | number of linear amplitude parameters | $8$ |

The detector index is $i \in \{1,\dots,n_{\rm det}\}$, the mode index is
$m \in \{1,\dots,n_{\rm mode}\}$, and the time index is $t \in \{1,\dots,n_t\}$. All detectors
are analyzed on a common number of samples $n_t$ (the code takes the shortest available
segment), though nothing in the derivation requires this: $n_t$ could carry an index $i$
throughout without changing a single step.

### 1.2 The signal model

Each quasinormal mode $m$ is a damped sinusoid with frequency $f_m$ (in Hz) and damping rate
$\gamma_m$ (in Hz). Writing $\omega_m = 2\pi f_m$, the two gravitational-wave
polarizations radiated by the source, evaluated at detector $i$'s local time $t$ measured from
that detector's ringdown start time, are

$$
h_+(t) \;=\; \sum_{m=1}^{n_{\rm mode}} \sigma_m\, e^{-\gamma_m t}
\Big[\, a^{p x}_m \cos\omega_m t \;+\; a^{p y}_m \sin\omega_m t \,\Big],
$$

$$
h_\times(t) \;=\; \sum_{m=1}^{n_{\rm mode}} \sigma_m\, e^{-\gamma_m t}
\Big[\, a^{c x}_m \cos\omega_m t \;+\; a^{c y}_m \sin\omega_m t \,\Big].
$$

Here:

* $a^{px}_m, a^{py}_m, a^{cx}_m, a^{cy}_m$ are dimensionless real numbers — the **quadrature
  amplitudes** (§1.3). There are $n_{\rm quad}=4$ of them per mode in the generic model.
* $\sigma_m > 0$ is the **amplitude scale** of mode $m$, a parameter with the dimensions of
  strain (§1.6). It multiplies all four quadratures of mode $m$.

In the strong-field regime described by general relativity, $f_m$ and $\gamma_m$ are not free:
for a Kerr remnant of detector-frame mass $M$ (in solar masses) and dimensionless spin $\chi$,

$$
f_m \;=\; \frac{\hat f_m(\chi)}{T_\odot M}, \qquad
\gamma_m \;=\; \frac{\hat\gamma_m(\chi)}{T_\odot M},
\qquad T_\odot \equiv \frac{G M_\odot}{c^3} = 4.9255\times10^{-6}\ \text{s},
$$

with $\hat f_m, \hat\gamma_m$ dimensionless functions of spin alone, evaluated in the code by a
polynomial fit in $\log(1-\chi)$ and $\tfrac12\log(1-\chi^2)$. Optional fractional deviations
$\delta f_m,\ \delta\gamma_m$ may multiply these, $f_m \to f_m e^{\delta f_m}$, to test
departures from the Kerr spectrum.

### 1.3 What a "quadrature" is, and why there are four

A damped sinusoid of fixed $f_m$ and $\gamma_m$ but free amplitude and phase,
$\mathcal{A} e^{-\gamma_m t}\cos(\omega_m t - \varphi)$, is *nonlinear* in $\varphi$. Expanding
the cosine,

$$
\mathcal{A}\,e^{-\gamma_m t}\cos(\omega_m t - \varphi)
= \underbrace{(\mathcal{A}\cos\varphi)}_{\text{quadrature }x}\, e^{-\gamma_m t}\cos\omega_m t
+ \underbrace{(\mathcal{A}\sin\varphi)}_{\text{quadrature }y}\, e^{-\gamma_m t}\sin\omega_m t,
$$

makes it *linear* in the pair $(\mathcal{A}\cos\varphi,\ \mathcal{A}\sin\varphi)$. These two
numbers are the **quadratures**: they encode amplitude and phase in a linear parameterization.

Each of $h_+$ and $h_\times$ needs its own amplitude and phase, so a generic (elliptically
polarized) mode requires $n_{\rm quad} = 4$ quadratures:
$(a^{px}_m, a^{py}_m)$ for the plus polarization and $(a^{cx}_m, a^{cy}_m)$ for the cross.
This is the LVK-reviewed production model. Two restricted models reduce this to
$n_{\rm quad} = 2$:

* the **aligned** model, which imposes the general-relativistic relation between the two
  polarizations of a given $(\ell,m)$ harmonic through spin-weighted spherical harmonics and an
  inclination angle, leaving one amplitude and one phase per mode;
* the **single-polarization** model, which sets $h_\times = 0$.

Both are flagged in the source as not reviewed for LVK use. The derivation below is
independent of the value of $n_{\rm quad}$; only $k = n_{\rm quad} n_{\rm mode}$ enters.

### 1.4 Projection onto the detectors: antenna patterns

Detector $i$ measures a single scalar time series formed from the two polarizations weighted by
its **antenna patterns** $F^{(i)}_+$ and $F^{(i)}_\times$ — dimensionless numbers of order unity
fixed by the detector's geometry and the source's sky position $(\alpha,\delta)$ and
polarization angle $\psi$. These are *known constants* in the analysis: the sky position is
taken from a prior full-signal analysis, not sampled. The noise-free strain in detector $i$ is

$$
h_i(t) \;=\; F^{(i)}_+ \, h_+(t) \;+\; F^{(i)}_\times \, h_\times(t).
$$

(The polarizations are evaluated at detector $i$'s own start time, which accounts for the
light-travel-time delay across the network.)

### 1.5 The design matrix

Collect the $k = n_{\rm quad} n_{\rm mode}$ quadratures into a single vector
$a \in \mathbb{R}^{k}$, ordered **quadrature-major, mode-minor**:

$$
a \;=\;
\big(\,\underbrace{a^{px}_1,\dots,a^{px}_{n_{\rm mode}}}_{\text{block }1},\;
\underbrace{a^{py}_1,\dots,a^{py}_{n_{\rm mode}}}_{\text{block }2},\;
\underbrace{a^{cx}_1,\dots,a^{cx}_{n_{\rm mode}}}_{\text{block }3},\;
\underbrace{a^{cy}_1,\dots,a^{cy}_{n_{\rm mode}}}_{\text{block }4}\,\big)^\top .
$$

Then §1.2 and §1.4 combine into a matrix–vector product. Define the **design matrix**
$M_i \in \mathbb{R}^{n_t \times k}$ for detector $i$ by

$$
\big[M_i\big]_{t,\,(q,m)} \;=\;
\sigma_m\, e^{-\gamma_m t}\times
\begin{cases}
F^{(i)}_+ \cos\omega_m t, & q = 1 \quad (\text{plus, cosine}),\\[2pt]
F^{(i)}_+ \sin\omega_m t, & q = 2 \quad (\text{plus, sine}),\\[2pt]
F^{(i)}_\times \cos\omega_m t, & q = 3 \quad (\text{cross, cosine}),\\[2pt]
F^{(i)}_\times \sin\omega_m t, & q = 4 \quad (\text{cross, sine}),
\end{cases}
$$

where the column index is $(q,m) \mapsto (q-1)\,n_{\rm mode} + m$, matching the ordering of $a$.
**Rows of $M_i$ index time samples; columns index (quadrature, mode) pairs.** By construction,

$$
\boxed{\;h_i \;=\; M_i\, a\;}
$$

is the vector of $n_t$ noise-free strain samples in detector $i$.

Two properties of $M_i$ matter for what follows:

1. It depends on the quadrature amplitudes **not at all** — that is the point of the
   parameterization.
2. It depends on *all* the remaining parameters, which we collect into a vector $\theta$
   (§1.7). We write $M_i(\theta)$ when the dependence needs emphasis and $M_i$ otherwise.

### 1.6 Absorbing the amplitude scales: why $\Lambda = \mathbb{1}$

Notice that the scale $\sigma_m$ was placed *inside* $M_i$, not inside $a$. This is deliberate.
The prior on the quadratures is the standard isotropic Gaussian used by Isi & Farr,

$$
a \;\sim\; N(0,\ \Lambda), \qquad \Lambda = \mathbb{1}_k ,
$$

i.e. every quadrature is an independent unit normal. The physical amplitude of mode $m$ is
$\sigma_m$ times a unit-scale quadrature, so the physically meaningful prior scale lives in
$\sigma_m$, and $\sigma_m$ is itself given a prior and sampled:

$$
\sigma_m \;\sim\; \mathrm{Uniform}(0,\ \sigma_{\max}), \qquad m = 1,\dots,n_{\rm mode}.
$$

Had we instead written $M_i$ without the scales and put $\Lambda = \mathrm{diag}(\sigma^2)$, we
would obtain exactly the same model; folding the scales into $M_i$ simply makes $\Lambda$ the
identity and removes it from most of the algebra. **The choice is a change of variables inside
the integral, not a change of prior.**

That equivalence is, however, *analytic only*, and the convention chosen here is not merely
notational. Write $S = \operatorname{diag}(\sigma)$ for the scales tiled over the quadrature
blocks, and let $A^{-1}$ be the posterior precision on the quadratures (defined in §2.1). The
two placements give precisions related by a diagonal congruence,
$A^{-1}_{\Lambda = S^{2}} = S^{-1} A^{-1}_{\Lambda = \mathbb{1}} S^{-1}$, so the
$\Lambda = S^{2}$ placement picks up a factor $\operatorname{cond}(S)^{2}$ in its condition
number. Since $\sigma_m$ is sampled on $(0,\sigma_{\max})$ and may be driven arbitrarily close
to zero, that factor diverges in a region the sampler genuinely visits, and the gradient
eventually becomes non-finite there — whereas the $\Lambda = \mathbb{1}$ placement used here
gives a precision whose eigenvalues are bounded below by $1$ for every $\sigma$ (§5.2). Note
also that, $S$ being diagonal, it commutes with the row operations of the whitening step
introduced in §5, so the scales may equivalently be applied to the small $k\times k$ matrix
after that step instead of to the $(n_t,k)$ design matrix before it, with no change whatsoever
to the value.

It is worth being explicit about what prior this
amounts to on the physical amplitudes: it is *hierarchical* — a zero-mean Gaussian of width
$\sigma_m$, with $\sigma_m$ itself uniform on $(0,\sigma_{\max})$ — and not a flat prior on
amplitude. (A flat amplitude prior is available only in the non-marginalized branch of the
code, via an explicit Jacobian factor; it is not the model discussed here.)

Because $\sigma_m$ is sampled rather than integrated out, **the scales belong to $\theta$**:
$M_i(\theta)$ depends on them.

### 1.7 The split between linear and nonlinear parameters

$$
\theta \;=\; \big(\, M,\ \chi,\ \{\sigma_m\},\ \{\delta f_m\},\ \{\delta\gamma_m\},\ \cos\iota \,\big)
\qquad\text{(whichever are enabled)} ,
$$

with $\cos\iota$ the inclination, present only in the aligned model. In the production
configuration $\theta$ has $2 + n_{\rm mode} = 4$ components: $M$, $\chi$, and two amplitude
scales. Everything in $\theta$ enters the model *nonlinearly* through $M_i(\theta)$.

The parameters $a$ enter *linearly*. That is the entire basis for what follows: for fixed
$\theta$, the map $a \mapsto h_i$ is linear, the noise is Gaussian, and the prior on $a$ is
Gaussian, so the $a$-integral is a Gaussian integral that can be done in closed form. The
sampler then explores only the $4$-dimensional $\theta$ space instead of $4 + k = 12$
dimensions.

$$
\underbrace{\;a \in \mathbb{R}^{k}\;}_{\text{marginalized analytically}}
\qquad\qquad
\underbrace{\;\theta \in \mathbb{R}^{4}\;}_{\text{sampled by NUTS}}
$$

### 1.8 The data and the noise model

Let $y_i \in \mathbb{R}^{n_t}$ be the (conditioned, whitening-free) strain data of detector $i$.
The noise is modelled as a zero-mean, stationary, Gaussian process, so its statistics are
captured entirely by the autocovariance function, assembled into the symmetric positive-definite
**noise covariance matrix**

$$
C_i \in \mathbb{R}^{n_t\times n_t},
\qquad \big[C_i\big]_{tt'} = \big\langle n_i(t)\, n_i(t')\big\rangle
= \rho_i\!\left(|t-t'|\right),
$$

a Toeplitz matrix built from the estimated autocovariance $\rho_i$. The code never inverts
$C_i$; it stores instead its lower-triangular **Cholesky factor**

$$
L_i \in \mathbb{R}^{n_t\times n_t}, \qquad C_i = L_i L_i^\top, \qquad
[L_i]_{tt'} = 0 \ \text{ for } t' > t,\quad [L_i]_{tt} > 0 .
$$

$L_i$ is a *whitening* operator: if $n_i \sim N(0,C_i)$ then $L_i^{-1} n_i \sim N(0,\mathbb{1})$.
Crucially, **$y_i$, $C_i$ and $L_i$ do not depend on $\theta$ or $a$** — they are fixed data.

Noise in different detectors is taken to be independent. This is the assumption that makes the
likelihood factorize over $i$, and it is what makes the sum in §4 legitimate.

### 1.9 The model in one line

$$
\boxed{\;
y_i \;=\; M_i(\theta)\, a \;+\; n_i, \qquad n_i \sim N(0, C_i)\ \text{independently},
\qquad a \sim N(0,\Lambda),\ \ \Lambda = \mathbb{1}_k .
\;}
$$

The likelihood of the full network for fixed $a$ and $\theta$ is therefore

$$
p(y \mid a, \theta) \;=\; \prod_{i=1}^{n_{\rm det}} N\!\big(y_i;\, M_i a,\, C_i\big)
\;=\; \prod_{i=1}^{n_{\rm det}}
\frac{\exp\!\big[-\tfrac12 (y_i - M_i a)^\top C_i^{-1}(y_i - M_i a)\big]}
{(2\pi)^{n_t/2}\,\lvert C_i\rvert^{1/2}} ,
$$

writing $y = (y_1,\dots,y_{n_{\rm det}})$, and the object we want is the **marginal likelihood**
(also called the evidence for $a$, or the profile-free likelihood of $\theta$)

$$
p(y\mid\theta) \;=\; \int_{\mathbb{R}^k} p(y\mid a,\theta)\; p(a)\; \mathrm{d}^k a .
$$

The sampler's target is $p(y\mid\theta)\,p(\theta)$.

---

## 2. The Gaussian marginalization identity

This section derives the integral in complete generality — for one data stream, with a general
Gaussian prior $a \sim N(\mu, \Lambda)$ rather than $N(0,\mathbb{1})$. The generality is needed
in §3, where the sequential scheme feeds a non-trivial $(\mu,\Lambda)$ into the next detector.

**Setup.** Let $y \in \mathbb{R}^n$, $M \in \mathbb{R}^{n\times k}$, $C \in \mathbb{R}^{n\times n}$
symmetric positive definite, $\mu \in \mathbb{R}^k$, $\Lambda \in \mathbb{R}^{k\times k}$
symmetric positive definite, and

$$
y = Ma + n,\qquad n\sim N(0,C),\qquad a \sim N(\mu,\Lambda),\qquad n \perp a .
$$

We want $p(y) = \int N(y; Ma, C)\, N(a;\mu,\Lambda)\, \mathrm{d}^k a$.

### 2.1 Completing the square

Write $-2\log$ of the integrand, dropping nothing:

$$
-2\log\big[\,\cdot\,\big] \;=\;
(y-Ma)^\top C^{-1}(y-Ma) \;+\; (a-\mu)^\top \Lambda^{-1}(a-\mu)
\;+\; n\log 2\pi + \log|C| + k\log2\pi + \log|\Lambda| .
$$

Expand the two quadratic forms and collect powers of $a$:

$$
(y-Ma)^\top C^{-1}(y-Ma) = y^\top C^{-1} y - 2\,a^\top M^\top C^{-1} y + a^\top M^\top C^{-1} M\, a ,
$$
$$
(a-\mu)^\top \Lambda^{-1}(a-\mu) = a^\top \Lambda^{-1} a - 2\,a^\top \Lambda^{-1}\mu + \mu^\top \Lambda^{-1}\mu .
$$

So the $a$-dependence is $a^\top A^{-1} a - 2 a^\top b$, where we **define**

$$
\boxed{\;A^{-1} \;\equiv\; \Lambda^{-1} + M^\top C^{-1} M\;}
\qquad\text{(posterior PRECISION; } k\times k\text{, symmetric positive definite)},
$$
$$
\boxed{\;b \;\equiv\; \Lambda^{-1}\mu + M^\top C^{-1} y\;}
\qquad\text{(information vector; } k\text{-dimensional)} .
$$

$A^{-1}$ is positive definite because $\Lambda^{-1}$ is and $M^\top C^{-1} M$ is positive
semidefinite, so $A \equiv (A^{-1})^{-1}$ exists. Completing the square about
$\hat a \equiv A\, b$:

$$
a^\top A^{-1} a - 2a^\top b \;=\; (a - \hat a)^\top A^{-1} (a-\hat a) \;-\; b^\top A\, b .
$$

The $a$-integral is now a standard Gaussian integral,

$$
\int_{\mathbb{R}^k} \exp\!\Big[-\tfrac12 (a-\hat a)^\top A^{-1}(a-\hat a)\Big] \mathrm{d}^k a
= (2\pi)^{k/2}\, |A^{-1}|^{-1/2} ,
$$

which cancels the prior's $(2\pi)^{-k/2}$ exactly. Assembling:

$$
\boxed{\;
\log p(y) = -\tfrac12\Big[\, y^\top C^{-1} y + \mu^\top\Lambda^{-1}\mu - b^\top A\, b \,\Big]
\;-\;\tfrac12\log|C| \;-\;\tfrac12\log|\Lambda| \;-\;\tfrac12\log|A^{-1}|
\;-\;\tfrac{n}{2}\log 2\pi .
\;}
\tag{2.1}
$$

This is the **information (precision) form** of the marginal likelihood. Note that the
only $k\times k$ matrix that must be formed and factorized is $A^{-1}$; the $n\times n$ matrix
$C$ enters only through $C^{-1}$ applied to $y$ and to the $k$ columns of $M$. Note also the
by-product

$$
a \mid y \;\sim\; N\big(\hat a,\ A\big), \qquad \hat a = A\,b = A\big(\Lambda^{-1}\mu + M^\top C^{-1}y\big),
\tag{2.2}
$$

since the integrand, viewed as a function of $a$, is exactly a Gaussian with that mean and
covariance. We will need this in §3 and §6.

### 2.2 The same integral done a second way: Woodbury and the determinant lemma

There is a second, entirely elementary route to $p(y)$. Since $y = Ma + n$ is a sum of two
independent Gaussian vectors, $y$ is Gaussian, with

$$
\langle y\rangle = M\mu, \qquad
\mathrm{Cov}(y) \;=\; M\,\Lambda\,M^\top + C \;\equiv\; B \in \mathbb{R}^{n\times n},
$$

so, with the **residual** $r \equiv y - M\mu$,

$$
\log p(y) = -\tfrac12\, r^\top B^{-1} r \;-\;\tfrac12\log|B| \;-\;\tfrac{n}{2}\log2\pi .
\tag{2.3}
$$

Equations (2.1) and (2.3) are two expressions for the same function of $y$. Because they agree
for *every* $y$, the quadratic parts and the constants must agree separately, which yields two
classical identities *without needing to cite them*:

**(a) The Woodbury / matrix-inversion identity.** Matching the coefficient of the quadratic in
$y$ gives

$$
\boxed{\;B^{-1} \;=\; \big(C + M\Lambda M^\top\big)^{-1} \;=\; C^{-1} - C^{-1} M\,A\,M^\top C^{-1},
\qquad A = \big(\Lambda^{-1} + M^\top C^{-1}M\big)^{-1} .\;}
\tag{2.4}
$$

This is the practically important one: it converts an $n\times n$ inverse into a $k\times k$
inverse, with $n = 205$ and $k = 8$.

**(b) The determinant lemma.** Matching the $y$-independent constants gives
$\log|C| + \log|\Lambda| + \log|A^{-1}| = \log|B|$, i.e.

$$
\boxed{\;|\Lambda|\,|C| \;=\; |A|\,|B| \;}
\qquad\Longleftrightarrow\qquad
\tfrac12\log|B| = \tfrac12\log|C| + \tfrac12\log|\Lambda| + \tfrac12\log|A^{-1}| .
\tag{2.5}
$$

In terms of Cholesky factors — $C = LL^\top$, $\Lambda^{-1} = \Lambda^{-1/2}_{\rm c}\Lambda^{-\top/2}_{\rm c}$,
$A^{-1} = R R^\top$ with $L$, $\Lambda_{\rm c}^{-1/2}$, $R$ all lower triangular — and using
$\log|X| = 2\sum_j \log X_{jj}$ for a Cholesky factor $X$, so that
$\sum_j \log[\Lambda_{\rm c}^{-1/2}]_{jj} = \tfrac12\log|\Lambda^{-1}| = -\tfrac12\log|\Lambda|$,
(2.5) reads

$$
\tfrac12\log|B| \;=\; \sum_{t}\log L_{tt} \;-\; \sum_j \log \big[\Lambda_{\rm c}^{-1/2}\big]_{jj}
\;+\; \sum_j \log R_{jj} .
\tag{2.6}
$$

This is precisely the three-term expression the code assembles as `log_sqrt_det_B`.

---

## 3. The sequential scheme

The scheme this code used before the closed form of §5 exploits the fact that Bayesian
updating is sequential: the posterior on $a$ after seeing detector $1$ is the correct prior
for detector $2$, and so on. Concretely, it initializes

$$
\mu^{(0)} = 0 \in \mathbb{R}^k, \qquad
\big[\Lambda^{(0)}\big]^{-1} = \mathbb{1}_k, \qquad
\big[\Lambda^{(0)}\big]^{-1/2}_{\rm c} = \mathbb{1}_k ,
$$

and then, for $i = 1,\dots,n_{\rm det}$, applies §2 with $(\mu,\Lambda) = (\mu^{(i-1)}, \Lambda^{(i-1)})$:

$$
\text{(precision update)}\qquad
\big[A^{(i)}\big]^{-1} = \big[\Lambda^{(i-1)}\big]^{-1} + M_i^\top C_i^{-1} M_i ,
\tag{3.1}
$$
$$
\text{(mean update)}\qquad
\mu^{(i)} \;=\; A^{(i)}\Big(\big[\Lambda^{(i-1)}\big]^{-1}\mu^{(i-1)} + M_i^\top C_i^{-1} y_i\Big),
\tag{3.2}
$$
$$
\text{(carry forward)}\qquad
\big[\Lambda^{(i)}\big]^{-1} \;=\; \big[A^{(i)}\big]^{-1} ,
\tag{3.3}
$$

with $R^{(i)}$ the lower Cholesky factor of $[A^{(i)}]^{-1}$. The likelihood contribution
emitted for detector $i$ is (2.3) with (2.4) and (2.6) substituted, and with the
parameter-independent constant $-\tfrac{n_t}{2}\log2\pi$ **deliberately dropped**:

$$
\ell_i \;\equiv\;
-\tfrac12\, r_i^\top\Big[C_i^{-1} - C_i^{-1}M_i A^{(i)} M_i^\top C_i^{-1}\Big] r_i
\;-\;\Big[\textstyle\sum_t \log [L_i]_{tt} - \sum_j \log R^{(i-1)}_{jj} + \sum_j \log R^{(i)}_{jj}\Big],
\tag{3.4}
$$

$$
r_i \;=\; y_i - M_i\,\mu^{(i-1)} \in \mathbb{R}^{n_t}, \qquad R^{(0)} \equiv \mathbb{1}_k .
$$

The quantity added to the sampler's log-posterior is $\sum_i \ell_i$, and by construction

$$
\ell_i \;=\; \log p\big(y_i \mid y_{<i},\, \theta\big) \;+\; \tfrac{n_t}{2}\log 2\pi ,
\tag{3.5}
$$

because $a \mid y_{<i} \sim N(\mu^{(i-1)}, \Lambda^{(i-1)})$ by (2.2) and (3.1)–(3.3), and
$y_i = M_i a + n_i$ with $n_i$ independent of everything before it.

Each detector therefore requires: two triangular solves against $L_i$ for $M_i$, further
triangular solves against $L_i$ for the three vectors $y_i$, $r_i$ and $M_i A^{(i)}M_i^\top C_i^{-1}r_i$,
one $k\times k$ Cholesky, and a strictly serial dependency through $\mu^{(i)}$ and $\Lambda^{(i)}$.

---

## 4. The telescoping proof

We now show that $\sum_i \ell_i$ collapses to a single expression involving only two
accumulators and one Cholesky, and in particular that the intermediate means $\mu^{(i)}$ and
residuals $r_i$ cancel identically.

### 4.1 The probabilistic argument

The fast argument is the chain rule of probability. Summing (3.5) over $i$ and telescoping the
conditionals,

$$
\sum_{i=1}^{n_{\rm det}} \ell_i
= \sum_{i=1}^{n_{\rm det}} \log p(y_i\mid y_{<i},\theta) + \tfrac{n_{\rm det} n_t}{2}\log2\pi
= \log p(y_1,\dots,y_{n_{\rm det}}\mid\theta) + \tfrac{n_{\rm det}n_t}{2}\log 2\pi .
\tag{4.1}
$$

So the sum is the *joint* marginal likelihood, up to the same parameter-independent constant
the code already drops. Evaluating that joint quantity directly is a one-line application of
§2.1 to the stacked system. Define the stacked objects

$$
\mathbf{y} = \begin{pmatrix} y_1\\ \vdots\\ y_{n_{\rm det}}\end{pmatrix}\in\mathbb{R}^{n_{\rm det}n_t},
\quad
\mathbf{M} = \begin{pmatrix} M_1\\ \vdots\\ M_{n_{\rm det}}\end{pmatrix}\in\mathbb{R}^{n_{\rm det}n_t\times k},
\quad
\mathbf{C} = \mathrm{blockdiag}\big(C_1,\dots,C_{n_{\rm det}}\big),
$$

where $\mathbf{C}$ is block diagonal **precisely because the detector noises are independent**
(§1.8). Then $\mathbf{y} = \mathbf{M}a + \mathbf{n}$ with $\mathbf{n}\sim N(0,\mathbf{C})$ and
$a \sim N(0,\mathbb{1})$, and block-diagonality gives

$$
\mathbf{M}^\top \mathbf{C}^{-1}\mathbf{M} = \sum_i M_i^\top C_i^{-1} M_i, \quad
\mathbf{M}^\top \mathbf{C}^{-1}\mathbf{y} = \sum_i M_i^\top C_i^{-1} y_i, \quad
\mathbf{y}^\top \mathbf{C}^{-1}\mathbf{y} = \sum_i y_i^\top C_i^{-1} y_i,
$$
$$
\tfrac12\log|\mathbf{C}| = \sum_i \tfrac12\log|C_i| = \sum_i\sum_t \log[L_i]_{tt}.
$$

Applying (2.1) with $\mu = 0$ and $\Lambda = \mathbb{1}$ (so $\mu^\top\Lambda^{-1}\mu = 0$ and
$\log|\Lambda| = 0$) and dropping $-\tfrac{n_{\rm det}n_t}{2}\log2\pi$ as before:

$$
\boxed{\;
\sum_{i=1}^{n_{\rm det}} \ell_i
\;=\; -\tfrac12\Big[\,Q \;-\; v^\top A\, v\,\Big]
\;-\; \sum_{i}\sum_{t}\log [L_i]_{tt}
\;-\; \sum_{j=1}^{k}\log R_{jj}
\;}
\tag{4.2}
$$

with the **three accumulators** and the single Cholesky

$$
A^{-1} \;=\; \mathbb{1}_k + \sum_{i} M_i^\top C_i^{-1} M_i \quad (k\times k), \qquad
v \;=\; \sum_i M_i^\top C_i^{-1} y_i \quad (k), \qquad
Q \;=\; \sum_i y_i^\top C_i^{-1} y_i \quad (\text{scalar}),
$$
$$
A^{-1} = R R^\top, \qquad R \ \text{lower triangular},\qquad
v^\top A\, v = v^\top (RR^\top)^{-1} v = \big\lVert R^{-1} v\big\rVert^2 .
$$

Note $\mu^{(i)}$, $r_i$, $b$ and $B$ have all disappeared, and the per-detector work is now a
plain *sum* — order-independent and with no serial dependency.

### 4.2 The same result by explicit algebra

The probabilistic argument is complete and rigorous, but it is worth seeing the cancellation
happen term by term, because it makes clear *why* the intermediate means drop out. Define the
running accumulators after $i$ detectors,

$$
J_i \equiv \sum_{j\le i} M_j^\top C_j^{-1} M_j \ \ (k\times k), \qquad
v_i \equiv \sum_{j\le i} M_j^\top C_j^{-1} y_j\ \ (k), \qquad
Q_i \equiv \sum_{j\le i} y_j^\top C_j^{-1} y_j \ \ (\text{scalar}),
$$

with $J_0 = 0$, $v_0 = 0$, $Q_0 = 0$. For brevity write, for the step $i$,

$$
N_i \equiv M_i^\top C_i^{-1} M_i, \qquad
w_i \equiv M_i^\top C_i^{-1} y_i, \qquad
q_i \equiv y_i^\top C_i^{-1} y_i,
$$
$$
P \equiv \big[\Lambda^{(i-1)}\big]^{-1}, \qquad S \equiv \big[A^{(i)}\big]^{-1} = P + N_i,
\qquad \mu \equiv \mu^{(i-1)} .
$$

**Lemma 1 (the running state has a closed form).** For all $i \ge 0$,

$$
\big[\Lambda^{(i)}\big]^{-1} = \mathbb{1} + J_i,
\qquad
\mu^{(i)} = \big(\mathbb{1}+J_i\big)^{-1} v_i .
$$

*Proof (induction).* For $i=0$: $[\Lambda^{(0)}]^{-1} = \mathbb{1} = \mathbb{1}+J_0$ and
$\mu^{(0)} = 0 = (\mathbb{1})^{-1}v_0$. Assume it holds at $i-1$. Then by (3.1) and (3.3),

$$
\big[\Lambda^{(i)}\big]^{-1} = \big[A^{(i)}\big]^{-1} = \big(\mathbb{1}+J_{i-1}\big) + N_i = \mathbb{1}+J_i ,
$$

and by (3.2),

$$
\mu^{(i)} = \big(\mathbb{1}+J_i\big)^{-1}\Big[\underbrace{\big(\mathbb{1}+J_{i-1}\big)\big(\mathbb{1}+J_{i-1}\big)^{-1}v_{i-1}}_{=\,v_{i-1}} + w_i\Big]
= \big(\mathbb{1}+J_i\big)^{-1} v_i . \qquad\blacksquare
$$

This is the crux. The mean update (3.2) always multiplies the previous mean by the previous
*precision*, and those two undo each other, leaving only the raw accumulated information vector
$v_{i-1}$. The means never really carry independent information; they are a re-encoding of
$(J_i, v_i)$.

**Lemma 2 (the determinant terms telescope).** Summing the bracketed term in (3.4),

$$
\sum_{i=1}^{n_{\rm det}}\Big[\sum_t \log [L_i]_{tt} - \sum_j\log R^{(i-1)}_{jj} + \sum_j\log R^{(i)}_{jj}\Big]
= \sum_i\sum_t\log[L_i]_{tt} + \sum_j \log R^{(n_{\rm det})}_{jj} - \underbrace{\sum_j\log R^{(0)}_{jj}}_{=\,0},
$$

because (3.3) makes the Cholesky factor carried into step $i$ equal to the one produced at step
$i-1$, so consecutive terms cancel in pairs, and $R^{(0)} = \mathbb{1}$. By Lemma 1,
$[A^{(n_{\rm det})}]^{-1} = \mathbb{1} + J_{n_{\rm det}} = A^{-1}$, so
$R^{(n_{\rm det})} = R$, and the total is $\sum_i\sum_t\log[L_i]_{tt} + \sum_j \log R_{jj}$ —
exactly the last two terms of (4.2). $\blacksquare$

**Lemma 3 (the quadratic terms telescope).** Define the running "evidence quadratic"

$$
E_i \;\equiv\; -\tfrac12\Big[\,Q_i - v_i^\top \big(\mathbb{1}+J_i\big)^{-1} v_i\,\Big], \qquad E_0 = 0 .
$$

Then the quadratic part of $\ell_i$ in (3.4) equals $E_i - E_{i-1}$.

*Proof.* Write the quadratic part as $-\tfrac12 r_i^\top B_i^{-1} r_i$ with
$r_i = y_i - M_i\mu$ and, by (2.4), $B_i^{-1} = C_i^{-1} - C_i^{-1}M_i S^{-1} M_i^\top C_i^{-1}$.
Expanding,

$$
r_i^\top C_i^{-1} r_i = q_i - 2\mu^\top w_i + \mu^\top N_i \mu ,
\qquad
M_i^\top C_i^{-1} r_i = w_i - N_i\mu ,
$$
$$
\Longrightarrow\quad
r_i^\top B_i^{-1} r_i = \big(q_i - 2\mu^\top w_i + \mu^\top N_i\mu\big) - \big(w_i - N_i\mu\big)^\top S^{-1}\big(w_i-N_i\mu\big) .
\tag{4.3}
$$

The claim $E_i - E_{i-1} = -\tfrac12 r_i^\top B_i^{-1} r_i$ is, after multiplying by $-2$ and
using $Q_i = Q_{i-1} + q_i$,

$$
r_i^\top B_i^{-1} r_i \;\overset{?}{=}\; q_i \;-\; v_i^\top S^{-1} v_i \;+\; v_{i-1}^\top P^{-1} v_{i-1} .
\tag{4.4}
$$

Now use Lemma 1 twice: $\mu = P^{-1}v_{i-1}$, hence $v_{i-1} = P\mu$ and
$v_{i-1}^\top P^{-1} v_{i-1} = \mu^\top P \mu$. Also $v_i = v_{i-1} + w_i = P\mu + w_i$, so

$$
w_i - N_i\mu \;=\; \big(v_i - P\mu\big) - N_i\mu \;=\; v_i - (P+N_i)\mu \;=\; v_i - S\mu ,
$$

and therefore

$$
\big(w_i - N_i\mu\big)^\top S^{-1}\big(w_i - N_i\mu\big)
= \big(v_i - S\mu\big)^\top S^{-1}\big(v_i - S\mu\big)
= v_i^\top S^{-1} v_i - 2\mu^\top v_i + \mu^\top S\mu .
$$

Substituting this and (4.3) into (4.4), and cancelling $q_i$ from both sides, the claim becomes

$$
-2\mu^\top w_i + \mu^\top N_i \mu - \Big[v_i^\top S^{-1}v_i - 2\mu^\top v_i + \mu^\top S\mu\Big]
\;\overset{?}{=}\; - v_i^\top S^{-1} v_i + \mu^\top P\mu .
$$

The $v_i^\top S^{-1} v_i$ terms cancel, leaving

$$
-2\mu^\top w_i + \mu^\top N_i\mu + 2\mu^\top v_i - \mu^\top S \mu - \mu^\top P\mu \;\overset{?}{=}\; 0 .
$$

Finally use $v_i = P\mu + w_i$, so $2\mu^\top v_i = 2\mu^\top P\mu + 2\mu^\top w_i$, which
cancels the leading $-2\mu^\top w_i$:

$$
\mu^\top N_i\mu + 2\mu^\top P\mu - \mu^\top S\mu - \mu^\top P\mu
= \mu^\top\big(N_i + P - S\big)\mu = 0 ,
$$

since $S = P + N_i$ by definition. $\blacksquare$

Summing Lemma 3 over $i$ gives $\sum_i(\text{quadratic part of }\ell_i) = E_{n_{\rm det}} - E_0
= -\tfrac12[\,Q - v^\top A v\,]$, which together with Lemma 2 reproduces (4.2). The two routes
agree, as they must.

**Where the residuals went.** The individual $r_i = y_i - M_i\mu^{(i-1)}$ are *not* zero and
*not* negligible; they are exactly the innovations of the sequential filter. What Lemma 3 shows
is that the $\mu$-dependence enters only through the combination $\mu^\top(N_i + P - S)\mu$,
which vanishes identically by the precision update (3.1). The subtraction of the predicted mean
in the residual is exactly compensated by the widening of the predictive covariance $B_i$
relative to $C_i$. This is the same cancellation that makes a Kalman filter's accumulated
innovation likelihood equal the batch likelihood.

---

## 5. Whitening

Equation (4.2) is stated in terms of $C_i^{-1}$, which is never formed. The Cholesky factor
$L_i$ turns every occurrence into a single triangular solve. Define, for each detector, the
**whitened design matrix** and **whitened data**

$$
\boxed{\;W_i \;\equiv\; L_i^{-1} M_i \in \mathbb{R}^{n_t\times k}, \qquad
z_i \;\equiv\; L_i^{-1} y_i \in \mathbb{R}^{n_t} .\;}
$$

Both are obtained by forward substitution against a lower-triangular matrix, never by inversion.
Then, using $C_i^{-1} = (L_iL_i^\top)^{-1} = L_i^{-\top}L_i^{-1}$:

$$
M_i^\top C_i^{-1} M_i = \big(L_i^{-1}M_i\big)^\top\big(L_i^{-1}M_i\big) = W_i^\top W_i ,
$$
$$
M_i^\top C_i^{-1} y_i = \big(L_i^{-1}M_i\big)^\top\big(L_i^{-1}y_i\big) = W_i^\top z_i ,
$$
$$
y_i^\top C_i^{-1} y_i = \big(L_i^{-1}y_i\big)^\top\big(L_i^{-1}y_i\big) = z_i^\top z_i = \lVert z_i\rVert^2 .
$$

These are exact identities. Substituting into (4.2) gives the final form:

$$
\boxed{
\begin{aligned}
A^{-1} &= \mathbb{1}_k + \sum_{i=1}^{n_{\rm det}} W_i^\top W_i, \qquad
v = \sum_{i=1}^{n_{\rm det}} W_i^\top z_i, \qquad
Q = \sum_{i=1}^{n_{\rm det}} \lVert z_i\rVert^2 ,\\[4pt]
A^{-1} &= R R^\top \ \ (\text{Cholesky, } R \text{ lower triangular}), \qquad
u \equiv R^{-1} v ,\\[4pt]
\sum_{i}\ell_i &= -\tfrac12 Q \;+\; \tfrac12 \lVert u\rVert^2
\;-\; \sum_{i}\sum_t \log [L_i]_{tt} \;-\; \sum_{j=1}^{k}\log R_{jj} .
\end{aligned}}
\tag{5.1}
$$

### 5.1 Which terms actually depend on $\theta$

This is worth stating explicitly, because it shows how little of (5.1) is real per-sample work:

| Quantity | Depends on $\theta$? | Why |
|---|---|---|
| $z_i = L_i^{-1}y_i$ | **No** | $L_i$ and $y_i$ are fixed data (§1.8) |
| $Q = \sum_i\lVert z_i\rVert^2$ | **No** | function of $z_i$ alone |
| $\sum_i\sum_t\log[L_i]_{tt}$ | **No** | function of $L_i$ alone |
| $W_i = L_i^{-1}M_i(\theta)$ | **Yes** | $M_i$ depends on $f_m,\gamma_m,\sigma_m$ |
| $A^{-1}$, $R$, $v$, $u$ | **Yes** | built from $W_i$ |

So $Q$ and $\sum_i\sum_t \log[L_i]_{tt}$ are *additive constants in $\log$-space*. They shift
the reported log-likelihood but are irrelevant to the sampler, exactly like the
$-\tfrac{n_{\rm det}n_t}{2}\log2\pi$ that the code already drops. They should nevertheless be
retained if one wants $\sum_i\ell_i$ to keep the same numerical value it has today, which is
the recommendation.

The genuinely $\theta$-dependent work per sample is: one triangular solve $L_i^{-1}M_i$ per
detector ($\mathcal{O}(n_t^2 k)$), one symmetric product $W_i^\top W_i$ per detector
($\mathcal{O}(n_t k^2)$), one $k\times k$ Cholesky ($\mathcal{O}(k^3)$), and one $k$-dimensional
triangular solve. Compare with §3, which needs *two* $\mathcal{O}(n_t^2 k)$ solves plus six
$\mathcal{O}(n_t^2)$ vector solves per detector and $n_{\rm det}$ Choleskys.

### 5.2 Conditioning

$A^{-1} = \mathbb{1} + \sum_i W_i^\top W_i$ is a sum of the identity and positive semidefinite
matrices, so its eigenvalues are bounded below by $1$: it is never singular, however degenerate
the design matrices become. This is a direct consequence of the unit prior $\Lambda=\mathbb{1}$
regularizing the problem, and it holds equally for the sequential and one-shot forms — indeed
the final $[A^{(n_{\rm det})}]^{-1}$ of the recursion *is* the one-shot $A^{-1}$, by Lemma 1, so
the two schemes factorize the same matrix and inherit the same conditioning. Its condition
number grows as the amplitude scales $\sigma_m$ grow or as two modes become degenerate
(nearly parallel columns of $M_i$); numerical experiments confirm that, **in double
precision**, the two forms lose accuracy at identical rates. `tests/test_model.py` exercises
exactly that regime: with nearly coincident modes on top of an aLIGO-like covariance
($\operatorname{cond}(C)\sim10^{9}$) the two forms' gradients separate by $\sim10^{-8}$
relative, which is $\operatorname{cond}(A^{-1})$ times machine epsilon and is shared roundoff
in both, while with a white covariance the same near-degenerate points agree to
$\sim3\times10^{-15}$.

That last equivalence is specific to float64. In single precision the two forms do *not* fail
together: the sequential recursion returns NaN in regimes where the one-shot form still returns
a finite value. The controlling variable is dynamic range rather than conditioning. The
sequential form builds $C_i^{-1}M_i$ and $C_i^{-1}r_i$ explicitly, and for strain data at its
natural scale ($L_i\sim10^{-22}$, so $C_i^{-1}\sim10^{44}$) those intermediates reach
$10^{26}$–$10^{44}$ and overflow the float32 ceiling of $3.4\times10^{38}$ — even where
$\operatorname{cond}(A^{-1})$ is of order ten and the final contractions are $O(1)$. Whitening
once keeps $W_i$ and $z_i$ of order unity, so no such intermediate is ever formed. This
attribution is measured, not inferred; an earlier account that located the failure at
$\operatorname{cond}(A)\gtrsim5\times10^{6}$ and ascribed it to accumulated rounding across
the repeated `cho_solve` chain was superseded by the later and more direct measurement. Two
caveats keep this in proportion: remaining finite is not the same as
remaining accurate — at high conditioning the one-shot form's float32 gradient is already
$O(1)$ in relative error, and at $\operatorname{cond}(A)\sim10^{13}$ it too returns NaN — and
none of this is an argument for running in single precision, which changes the numerics and is
out of scope here. It bears only on which formulation would be the prerequisite if that were
ever attempted.

Forming $A^{-1}$ as $W_i^\top W_i$ rather than $M_i^\top(C_i^{-1}M_i)$ is also the numerically
preferable choice: it is a Gram matrix of the whitened design, so its condition number is the
square of that of $W_i$, whereas the unsymmetrized product can pick up additional rounding
asymmetry. It is exactly symmetric in floating point, since $[W^\top W]_{jl}$ and
$[W^\top W]_{lj}$ are the same sum of the same products in the same order.

---

## 6. The predictive draw

When posterior samples of the quadratures themselves are wanted — to reconstruct waveforms, or
to report amplitudes, phases and ellipticities — they are drawn from the conditional posterior
$p(a\mid y,\theta)$ at each sampled $\theta$. By (2.2) applied to the stacked system (or
equivalently by Lemma 1 at $i = n_{\rm det}$),

$$
a \mid y,\theta \;\sim\; N\big(\hat a,\ A\big), \qquad
\hat a = A\,v, \qquad A = \big(A^{-1}\big)^{-1} = \big(RR^\top\big)^{-1} = R^{-\top}R^{-1} .
$$

Neither $A$ nor $A^{-1}$ need be inverted. With $u = R^{-1}v$ as in (5.1),

$$
\hat a = R^{-\top}R^{-1} v = R^{-\top} u ,
$$

and if $\xi \sim N(0,\mathbb{1}_k)$ is a vector of $k$ independent standard normals, then

$$
\boxed{\; a \;=\; \hat a \;+\; R^{-\top}\xi \;=\; R^{-\top}\big(u + \xi\big) \;}
$$

has the required distribution, because

$$
\mathrm{Cov}\big(R^{-\top}\xi\big) = R^{-\top}\,\mathbb{1}\,R^{-1} = \big(RR^\top\big)^{-1} = A .
$$

Both operations are single triangular solves against $R^\top$ (back substitution). The current
code performs the equivalent computation using the final state of the recursion,
$a = \mu^{(n_{\rm det})} + [R^{(n_{\rm det})}]^{-\top}\xi$, which by Lemma 1 is the same thing;
in the one-shot form one simply substitutes $\mu^{(n_{\rm det})} = R^{-\top}u$.

### 6.1 Derived physical quantities

The drawn $a$ is a vector of *unit-scale* quadratures; the physical quadratures of mode $m$ are
$\sigma_m$ times these. From the four physical quadratures of a mode the code reconstructs the
amplitude, ellipticity, and the two polarization phase angles used by Isi & Farr. Writing
$(a^{px}, a^{py}, a^{cx}, a^{cy})$ for one mode's unit quadratures, define

$$
A_R \equiv \tfrac12\sqrt{\big(a^{cy}+a^{px}\big)^2 + \big(a^{cx}-a^{py}\big)^2}, \qquad
A_L \equiv \tfrac12\sqrt{\big(a^{cy}-a^{px}\big)^2 + \big(a^{cx}+a^{py}\big)^2},
$$
$$
\varphi_R \equiv \operatorname{atan2}\big(a^{py}-a^{cx},\ a^{cy}+a^{px}\big), \qquad
\varphi_L \equiv \operatorname{atan2}\big(-a^{cx}-a^{py},\ a^{px}-a^{cy}\big),
$$

and then

$$
\mathcal{A} = \sigma_m\,(A_R + A_L), \qquad
\epsilon = \frac{A_R - A_L}{A_R + A_L}, \qquad
\vartheta = -\tfrac12(\varphi_R + \varphi_L), \qquad
\varphi = \tfrac12(\varphi_R - \varphi_L) .
$$

$A_R$ and $A_L$ are the right- and left-circularly-polarized amplitudes, $\mathcal{A}$ is the
total amplitude, $\epsilon\in[-1,1]$ the ellipticity, $\vartheta$ the orientation of the
polarization ellipse in the $(h_+,h_\times)$ plane, and $\varphi$ the phase. This is exactly the
inverse of Eq. (8) of Isi & Farr (2021),

$$
h_+ - i\,h_\times = \tfrac12\mathcal{A}\,e^{-\gamma t}
\Big[(1+\epsilon)\,e^{-i(\omega t - \varphi_p)} + (1-\epsilon)\,e^{i(\omega t + \varphi_m)}\Big],
\qquad \varphi_p = \varphi - \vartheta,\ \ \varphi_m = -(\varphi+\vartheta),
$$

with $\varphi_R = \varphi_p$ and $\varphi_L = \varphi_m$. (I verified this inverse relation both
symbolically and numerically against the implementation of Eq. (8) in
`ringdown/waveforms/ringdown.py:32`; agreement to $5\times10^{-15}$.) For the two-quadrature
models the reduction is simpler: $\mathcal{A} = \sigma_m\sqrt{(a^x)^2+(a^y)^2}$ and
$\varphi = \operatorname{atan2}(a^y, a^x)$, with $\vartheta = 0$ and $\epsilon$ set by the
inclination rather than fitted.

**None of this affects the likelihood.** These are deterministic post-processing functions of a
draw, recorded for interpretation only.

---

## 7. Translation to the code

All references are to `ringdown/model.py` unless stated otherwise.

### 7.1 Symbol dictionary

Locations are given as identifiers rather than line numbers, which rot.

| Note symbol | Meaning | Shape | Code name | Where |
|---|---|---|---|---|
| $n_{\rm det}$ | number of detectors | scalar | `n_det` | `make_model.model` |
| $n_t$ | analysis samples per detector | scalar | `n_analyze` | `fit.py`, `Fit.n_analyze` |
| $n_{\rm mode}$ | number of modes | scalar | `n_modes` | `make_model` |
| $k = n_{\rm quad}n_{\rm mode}$ | number of quadratures | scalar | `n_quad_n_modes` | marginalized branch |
| $y_i$ | strain data, detector $i$ | $n_t$ | `strains[i]` / `y` | model argument; built in `Fit.run_input` |
| $C_i$ | noise covariance | $n_t\times n_t$ | *(never formed)* | — |
| $L_i$ | Cholesky factor, $C_i = L_iL_i^\top$ | $n_t\times n_t$ | `ls[i]` / `L` | model argument; built in `Fit.run_input` |
| $F^{(i)}_+,F^{(i)}_\times$ | antenna patterns | scalars | `fps[i]`, `fcs[i]` | model arguments |
| $f_m$ | mode frequency (Hz) | $n_{\rm mode}$ | `f` | site or deterministic `"f"` |
| $\gamma_m$ | mode damping rate (1/s) | $n_{\rm mode}$ | `g` | site or deterministic `"g"` |
| $\sigma_m$ | amplitude scale | $n_{\rm mode}$ | `a_scale` | site `"a_scale"` |
| $M$, $\chi$ | remnant mass, spin | scalars | `m`, `chi` | sites `"m"`, `"chi"` |
| $T_\odot$ | solar mass in seconds | scalar | `qnms.T_MSUN` | `config.py` |
| $\hat f_m,\hat\gamma_m$ | dimensionless spin fits | $n_{\rm mode}$ | `chi_factors(chi, fcoeffs/gcoeffs)` | `chi_factors` |
| $\theta$ | nonlinear parameters | $2+n_{\rm mode}$ | *(the sampled sites)* | — |
| $a$ | quadrature amplitudes | $k$ | `quads` | predictive block |
| $M_i$ | design matrix | $n_t\times k$ | `dms[i]` / `M` | built by `rd_design_matrix` |
| $\Lambda$ | quadrature prior covariance $=\mathbb{1}$ | $k\times k$ | *(implicit)* | initial value of `A_inv` |
| $W_i$ | whitened design $L_i^{-1}M_i$ | $n_t\times k$ | `W` | detector loop |
| $z_i$ | whitened data $L_i^{-1}y_i$ | $n_t$ | `z` | detector loop |
| $A^{-1}$ | posterior **precision** | $k\times k$ | **`A_inv`** | detector loop (accumulated) |
| $A$ | posterior **covariance** | $k\times k$ | *(never formed)* | — |
| $v$ | $\sum_i W_i^\top z_i$ | $k$ | `v` | detector loop (accumulated) |
| $Q$ | $\sum_i \lVert z_i\rVert^2$ | scalar | `Q` | detector loop (accumulated) |
| $\sum_i\sum_t\log[L_i]_{tt}$ | noise log-determinant | scalar | `logdetL` | detector loop (accumulated) |
| $R$ | lower Cholesky of $A^{-1}$ | $k\times k$ | `A_inv_chol` | after the loop |
| $u = R^{-1}v$ | whitened information vector | $k$ | `u` | after the loop |
| $\sum_i\ell_i$ | total marginal log-likelihood | scalar | site `"logl_total"` | `numpyro.factor` |

> **The one trap.** `A_inv` is the note's $A^{-1}$, the *precision*. It is what is built and
> Cholesky-factorized. The note's $A$ — the covariance — corresponds to no variable in the
> code; wherever $A$ appears in a formula, the code realizes it as a solve against
> `A_inv_chol`. `A_inv` is initialized to the identity because the prior precision
> $\Lambda^{-1} = \mathbb{1}$.

### 7.2 Column ordering of $M_i$

Built in `rd_design_matrix`:

```python
decay = jnp.exp(-gamma * ts)                       # e^{-gamma_m t}
ct = Ascales * decay * jnp.cos(2 * np.pi * f * ts) # sigma_m e^{-gamma_m t} cos(omega_m t)
st = Ascales * decay * jnp.sin(2 * np.pi * f * ts) # sigma_m e^{-gamma_m t} sin(omega_m t)
dm = jnp.concatenate((Fp * ct, Fp * st, Fc * ct, Fc * st), axis=2)
```

so the columns are, in order,

| Column block | Index range | Content |
|---|---|---|
| $q=1$ | `[0 : n_mode]` | $F_+\,\sigma_m e^{-\gamma_m t}\cos\omega_m t$ |
| $q=2$ | `[n_mode : 2*n_mode]` | $F_+\,\sigma_m e^{-\gamma_m t}\sin\omega_m t$ |
| $q=3$ | `[2*n_mode : 3*n_mode]` | $F_\times\,\sigma_m e^{-\gamma_m t}\cos\omega_m t$ |
| $q=4$ | `[3*n_mode : 4*n_mode]` | $F_\times\,\sigma_m e^{-\gamma_m t}\sin\omega_m t$ |

matching the ordering of $a$ in §1.5, and matching the slicing of `quads` in
`get_quad_derived_quantities`. Note that `Ascales` ($\sigma_m$) is applied inside
`rd_design_matrix`, i.e. *inside* $M_i$ — this is the absorption of §1.6. The aligned model collapses the four blocks to
two using the spin-weighted harmonic factors $Y^+_m, Y^\times_m$ from
`ringdown.utils.swsh`.

### 7.3 Derived quantities

Implemented in `get_quad_derived_quantities`:

| Note symbol | Code | Where |
|---|---|---|
| $2A_R$ | `term1` | `Aellip_from_quadratures` |
| $2A_L$ | `term2` | `Aellip_from_quadratures` |
| $\mathcal{A} = \sigma_m(A_R+A_L)$ | `a` (site `"a"`) | `Aellip_from_quadratures` |
| $\epsilon = (A_R-A_L)/(A_R+A_L)$ | `e` (site `"ellip"`) | `Aellip_from_quadratures` |
| $\varphi_R$ | site `"phi_r"` | `phiR_from_quadratures` |
| $\varphi_L$ | site `"phi_l"` | `phiL_from_quadratures` |
| $\vartheta = -\tfrac12(\varphi_R+\varphi_L)$ | site `"theta"` | `get_quad_derived_quantities` |
| $\varphi = \tfrac12(\varphi_R-\varphi_L)$ | site `"phi"` | `get_quad_derived_quantities` |
| $h_i = M_i a$ | `h_det` | `get_quad_derived_quantities` |
| per-mode $h_i$ | `h_det_mode` | `get_quad_derived_quantities` |

### 7.4 Before and after

**Before**, condensed and annotated:

```python
mu = jnp.zeros(k)                    # mu^(0) = 0
Lambda_inv = jnp.eye(k)              # [Lambda^(0)]^-1 = 1
Lambda_inv_chol = jnp.eye(k)         # R^(0) = 1
for i in range(n_det):
    M, L, y = dms[i], ls[i], strains[i]                         # M_i, L_i, y_i
    A_inv = Lambda_inv + M.T @ cho_solve((L, True), M)          # (3.1)
    A_inv_chol = cholesky(A_inv, lower=True)                    # R^(i)
    a = cho_solve((A_inv_chol, True),
                  Lambda_inv @ mu + M.T @ cho_solve((L,True), y))  # (3.2): mu^(i)
    b = M @ mu                                                  # M_i mu^(i-1)
    r = y - b                                                   # r_i
    Cinv_r = cho_solve((L, True), r)                            # C_i^-1 r_i
    M_A_Mt_Cinv_r  = M @ cho_solve((A_inv_chol,True), M.T @ Cinv_r)
    Cinv_M_A_Mt_Cinv_r = cho_solve((L, True), M_A_Mt_Cinv_r)    # Woodbury (2.4)
    log_sqrt_det_B = (sum(log(diag(L)))                         # (2.6)
                      - sum(log(diag(Lambda_inv_chol)))
                      + sum(log(diag(A_inv_chol))))
    logl = -0.5 * r @ (Cinv_r - Cinv_M_A_Mt_Cinv_r) - log_sqrt_det_B   # (3.4) = l_i
    numpyro.factor(f"logl_{i}", logl)
    mu, Lambda_inv, Lambda_inv_chol = a, A_inv, A_inv_chol      # (3.3)
```

**After** — the same number, computed via (5.1), and what `ringdown.model.make_model` now
does:

```python
st = lambda X, Y: solve_triangular(X, Y, lower=True)
A_inv = jnp.eye(k); v = jnp.zeros(k); Q = 0.0; logdetL = 0.0
for i in range(n_det):
    L = ls[i]
    W = st(L, dms[i])                       # W_i = L_i^-1 M_i
    z = st(L, strains[i])                   # z_i = L_i^-1 y_i
    A_inv   = A_inv   + W.T @ W             # 1 + sum_i W_i^T W_i
    v       = v       + W.T @ z             # sum_i W_i^T z_i
    Q       = Q       + jnp.dot(z, z)       # sum_i ||z_i||^2
    logdetL = logdetL + jnp.sum(jnp.log(jnp.diagonal(L)))
R = cholesky(A_inv, lower=True)             # A^-1 = R R^T
u = st(R, v)                                # u = R^-1 v
numpyro.factor("logl_total",                # (5.1)
               -0.5*Q + 0.5*jnp.dot(u, u) - logdetL - jnp.sum(jnp.log(jnp.diag(R))))
```

and, for the predictive draw,

```python
quads = solve(R.T, u + unit_quads)          # a = R^-T (u + xi),  Section 6
```

replacing `mu + solve(Lambda_inv_chol.T, unit_quads)`, which by Lemma 1 is the same vector.

Two implementation remarks, both settled empirically rather than mathematically. First, $W_i$ and
$z_i$ may equivalently be obtained from a *single* solve against the concatenated
$[\,M_i \mid y_i\,]$, an $n_t\times(k+1)$ right-hand side; that is algebraically identical and was
the form originally proposed here, but it has been retired. On CPU it saves exactly one LAPACK
dispatch and no arithmetic, while on GPU in float64 cuBLAS switches to a markedly slower kernel
once the right-hand side has $\gtrsim 17$ columns (exact powers of two being fast outliers), so
the concatenation can cost up to $\sim 2\times$ for $n_{\rm mode}\gtrsim 4$ while gaining at most
$\sim10\%$ elsewhere. Two separate solves are therefore used, as a single backend- and
dtype-independent code path. For the same reason the detector loop is left unrolled rather
than expressed with `vmap` or `scan`: unrolling won or tied on CPU, an RTX A6000 and an H100,
so there is no configuration in which a dispatch would pay for itself.

Second, $z_i$, $Q$ and $\sum_i\sum_t\log[L_i]_{tt}$ are independent of $\theta$ (§5.1) and so
carry no gradient, but they are *not* removed by the compiler: on XLA:CPU the triangular solve
lowers to an opaque `lapack_dtrsm_ffi` custom call, which the constant folder cannot evaluate,
so $z_i$ is genuinely recomputed on every call even though $L_i$ and $y_i$ are compile-time
constants. Precomputing them and passing them in as model arguments
was measured and found to be worth nothing end-to-end, so they are simply computed in the model
body; they are $O(n_t^2)$ against the $O(n_t^2 k)$ of the $W_i$ solve, and gradient-free.

Sections 4 and 5 prove these compute the same real number. Numerically, the two forms agree to
$\sim10^{-15}$ in both the log-likelihood and every gradient component for well-conditioned test
covariances; under a realistically conditioned aLIGO-like noise covariance the agreement
degrades, as shared roundoff amplified by $\operatorname{cond}(C)$ rather than as a bias in
either form — a central finite-difference check cannot tell which is more accurate — and stays
far below anything HMC can resolve. `tests/test_model.py` pins this against a frozen copy of
the sequential scheme: at $\operatorname{cond}(C)$ of a few times $10^{9}$ the potential energy
and every gradient component agree to $\le 4\times10^{-13}$ relative over
$(n_{\rm det}, n_t, n_{\rm mode}) \in \{(1,205,1),(2,205,2),(3,205,3),(2,410,2)\}$. The
predictive draw of §6 is additionally *pointwise* identical given the same $\xi$, not merely
equal in distribution, with all twelve derived quantities agreeing to the same tolerance.

### 7.5 One behavioural difference

The sequential scheme emits $n_{\rm det}$ separate factor sites `logl_0`, `logl_1`, …; the
one-shot scheme emits one. The individual $\ell_i = \log p(y_i\mid y_{<i},\theta)$ are
*order-dependent* quantities — they depend on the arbitrary order in which detectors are
processed — so no information of physical interest is lost by merging them; only their sum,
which is the quantity that enters the posterior, is order-independent. The merged site is
named `logl_total`, so downstream code that looks for site names beginning with `logl_` — in
particular `Result.draw_sample(map=True)` — continues to work unchanged. `Result.loo` and the
whitened pointwise log-likelihood never read these sites at all; they are computed from the
whitened residuals.

---

## References

* M. Isi and W. M. Farr, *Analyzing black-hole ringdowns*, arXiv:2107.05609 — the
  $(\mathcal{A},\epsilon,\vartheta,\varphi)$ parameterization of §6.1, Eq. (8).
* M. Isi, W. M. Farr, M. Giesler, M. A. Scheel and S. A. Teukolsky,
  *Testing the black-hole area law with GW150914*, arXiv:2005.14199 — the source of the
  $\Lambda$, $A$, $A^{-1}$, $B$, $b$, $\mu$ notation reused by the code and by this note.
* The Woodbury identity (2.4) and determinant lemma (2.5) are derived in §2.2 rather than cited.
