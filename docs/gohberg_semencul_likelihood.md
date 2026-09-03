# Applying the inverse noise covariance through the Gohberg–Semencul formula

*The mathematics behind PR #141, which replaces the Cholesky-based application of
$C^{-1}$ in the `ringdown` likelihood by an autoregressive (Yule–Walker) representation of
the noise, the Gohberg–Semencul inversion formula for symmetric Toeplitz matrices, and
FFT-based convolutions. Includes a careful comparison of the implemented formulas with the
exact ones.*

---

## 0. Purpose, scope, and two notational warnings

`ringdown` models the post-merger noise in each detector as a zero-mean, stationary Gaussian
process. Its covariance over the $N$ analyzed samples is therefore a symmetric
positive-definite Toeplitz matrix $C$ built from the autocovariance function (ACF). The
Gaussian log-likelihood needs two things from $C$: the action of $C^{-1}$ on a handful of
vectors (the data, the residual, and the columns of the design matrix), and the scalar
$\log|C|$.

The code on `main` obtains both from a dense Cholesky factorization $C = LL^\top$, computed
once per detector outside the sampler and passed into the model as `ls`. Inside the model,
every $C^{-1}v$ becomes a pair of triangular solves (or, on the current one-shot code path, a
single whitening solve $L^{-1}v$), and $\log|C| = 2\sum_t \log L_{tt}$.

PR #141 ("Optimizations using the Gohberg Semencul theorem") replaces this with a different
representation of the same matrix. Outside the sampler it solves the Yule–Walker equations
for the coefficients of an order-$(N-1)$ autoregressive model of the noise, plus one scalar,
the innovation variance $\sigma^2$. Inside the sampler it applies $C^{-1}$ through the
Gohberg–Semencul formula, which writes the $N\times N$ inverse of a symmetric Toeplitz matrix
as a difference of two products of lower-triangular Toeplitz matrices, each of which is a
causal convolution and hence a product in Fourier space. It also replaces $\log|C|$ by
$2N\log\sigma$.

This note derives all of that from first principles, states each identity precisely, proves
the Gohberg–Semencul formula, and then compares the implemented formulas with the exact ones,
item by item, in §9. The comparison is neutral: its purpose is to record, for anyone
evaluating or building on the PR, exactly which quantities are computed exactly and which are
not, and what the consequences are. It does not benchmark the code; whether the FFT route is
faster for $N\approx 200$ is an empirical question that this note does not settle (§7).

### 0.1 Warning: the letters $A$ and $B$

The PR's code and comments call the two lower-triangular Toeplitz matrices of the
Gohberg–Semencul formula "A" and "B" (`fft_as`, `fft_bs`, `apply_A`, `apply_B`). In
`docs/marginalized_likelihood.md`, which this note is meant to sit beside, $A$ is the
posterior *covariance* of the quadrature amplitudes, $A^{-1}$ its precision (the code's
`A_inv`), and $B = C + M\Lambda M^\top$ is the marginal data covariance. To avoid the clash,
this note writes the two Toeplitz factors as $L(a)$ and $L(\tilde a)$ (defined in §4.1) and
never uses bare $A$ or $B$ for them. Where the code's names are needed for side-by-side
reading, they appear in backticks.

### 0.2 Warning: indexing from zero

Because shift matrices, first columns, and lag-$k$ autocovariances are central here, vectors
and matrices are indexed from $0$ to $N-1$ in this note, matching the code. The reference
note indexes time from $1$ to $n_t$; nothing depends on the choice.

---

## 1. Setup and notation

### 1.1 Notation table

| Symbol | Meaning | Code name (PR #141) |
|---|---|---|
| $N$ | number of analyzed time samples per detector ($n_t$ in the reference note; $205$ in production) | `n_analyze`, `n_time`, `N` |
| $i$ | detector index, $i = 1,\dots,n_{\rm det}$ | loop index `i` |
| $k$ | number of quadrature amplitudes ($n_{\rm quad}\,n_{\rm mode}$; $8$ in production) | `n_quad_n_modes` |
| $n_t$ | noise sample at time index $t$ (a random variable) | |
| $\gamma_j$ | autocovariance at lag $j$: $\gamma_j = \langle n_t n_{t+j}\rangle$; $\gamma_{-j} = \gamma_j$ | `acf_vals[j]` |
| $\gamma$ | the vector $(\gamma_0,\dots,\gamma_{N-1})^\top$ | `acf_vals` |
| $C$ | noise covariance, $C_{tt'} = \gamma_{\lvert t - t'\rvert}$; symmetric positive-definite Toeplitz, $N\times N$ | never formed in the PR |
| $C_i, M_i, y_i$ | per-detector covariance, design matrix ($N\times k$), data ($N$) | `M`, `y` |
| $L$ | lower Cholesky factor of $C$, $C = LL^\top$ (used by `main`, not by the PR) | `ls[i]`, `L` |
| $e_j$ | $j$-th standard basis vector of $\mathbb{R}^N$, $j = 0,\dots,N-1$ | |
| $Z$ | lower shift matrix, $Z e_j = e_{j+1}$, $Z e_{N-1} = 0$; $Z_{tt'} = \delta_{t,\,t'+1}$ | |
| $J$ | exchange (reversal) matrix, $J e_j = e_{N-1-j}$; $Jv = \mathrm{flip}(v)$ | `jnp.flip` |
| $L(x)$ | lower-triangular Toeplitz matrix with first column $x$: $[L(x)]_{tt'} = x_{t-t'}$ for $t\ge t'$, $0$ otherwise | |
| $\nabla(X)$ | displacement of a matrix, $\nabla(X) = X - Z X Z^\top$ | |
| $a = (a_0, a_1, \dots, a_{N-1})^\top$ | autoregressive (prediction-error) filter, $a_0 = 1$ | `full_ar_coeffs`, `ar_coeffs[i]` |
| $a' = (a_1,\dots,a_{N-1})^\top$ | the nontrivial part of $a$ | `a_coeffs` |
| $\tilde a$ | reversed-and-shifted filter, $\tilde a = ZJa = (0, a_{N-1}, a_{N-2}, \dots, a_1)^\top$ | `rev_coeffs` |
| $x$ | first column of $C^{-1}$, $x = C^{-1} e_0$ | |
| $\bar x$ | last column of $C^{-1}$, $\bar x = C^{-1} e_{N-1} = Jx$ | |
| $\Delta$, $\Delta_m$ | Schur complement of the leading block of $C$ (resp. of the leading $(m-1)\times(m-1)$ block of $C_m$); $\Delta = \sigma^2$, $\Delta_{m+1} = \sigma_m^2$ | |
| $\sigma^2$ | innovation variance of the order-$(N-1)$ predictor; $\sigma^2 = \gamma_0 + \sum_{j\ge1} a_j\gamma_j$ | `sigma_sq`, `sigmas[i]**2` |
| $\sigma_m^2$ | innovation variance of the order-$m$ predictor, $m = 0,\dots,N-1$; $\sigma_0^2 = \gamma_0$, $\sigma_{N-1}^2 = \sigma^2$ | not computed |
| $\kappa_m$ | $m$-th reflection (partial autocorrelation) coefficient of the Levinson recursion | not computed |
| $C_m$ | leading $m\times m$ block of $C$ (so $C_N = C$) | |
| $n_{\rm fft}$ | FFT length used for the convolutions | `n_fft` |
| $\hat x$ | discrete Fourier transform of a zero-padded vector $x$, length $n_{\rm fft}$ | `fft_as[i]`, `fft_bs[i]`, `X_f` |
| $\psi$ | a generic vector in $\mathbb{R}^N$ to which $C^{-1}$ is applied | `vector` |
| $A^{-1}, v, Q, R, u$ | accumulators of the one-shot marginal likelihood (reference note §5) | `A_inv`, `v`, `Q`, `A_inv_chol`, `u` |
| $\Lambda, \mu, b, r, B$ | prior covariance, prior mean, marginal mean, residual, marginal covariance of the sequential scheme (reference note §§2–3) | `Lambda_inv`, `mu`, `b`, `r` |
| $\theta$ | the sampled nonlinear parameters (mass, spin, amplitude scales, ...) | |
| $s$ | strain rescaling factor applied before the likelihood sees the data | `scale` |

Throughout, "the reference note" means `docs/marginalized_likelihood.md`. Its $\rho_i(|t-t'|)$
is this note's $\gamma_{|t-t'|}$ for detector $i$.

### 1.2 The one-line model

For detector $i$, with $\theta$ the nonlinear parameters and $\alpha\in\mathbb{R}^k$ the
quadrature amplitudes (the reference note's $a$, renamed here because $a$ is taken by the AR
filter),

$$
y_i = M_i(\theta)\,\alpha + n_i, \qquad n_i \sim N(0, C_i),
$$

with independent noise across detectors. The Gaussian log-density of one detector's data,
for a given mean $h = M_i\alpha$, is

$$
\log N(y_i; h, C_i) = -\tfrac12 (y_i - h)^\top C_i^{-1} (y_i - h) - \tfrac12\log|C_i| - \tfrac{N}{2}\log 2\pi .
\tag{1.1}
$$

Every use of $C_i$ in the code, in either the marginalized or the non-marginalized branch,
reduces to the two ingredients visible here: $C_i^{-1}$ applied to vectors, and $\log|C_i|$.
This note is about how those two ingredients are computed. For the rest of the note the
detector index is dropped whenever only one detector is in play.

---

## 2. Stationary noise gives a symmetric positive-definite Toeplitz covariance

**Stationarity.** A zero-mean process $n_t$ is (wide-sense) stationary if
$\langle n_t n_{t'}\rangle$ depends only on the lag $t - t'$. Writing
$\gamma_j \equiv \langle n_t n_{t+j}\rangle$, symmetry of the expectation gives
$\gamma_{-j} = \gamma_j$, so the covariance of the $N$ consecutive samples
$n = (n_0,\dots,n_{N-1})^\top$ is

$$
C_{tt'} = \gamma_{\lvert t - t'\rvert}, \qquad
C = \begin{pmatrix}
\gamma_0 & \gamma_1 & \gamma_2 & \cdots & \gamma_{N-1}\\
\gamma_1 & \gamma_0 & \gamma_1 & \cdots & \gamma_{N-2}\\
\gamma_2 & \gamma_1 & \gamma_0 & \cdots & \gamma_{N-3}\\
\vdots & & & \ddots & \vdots\\
\gamma_{N-1} & \gamma_{N-2} & \gamma_{N-3} & \cdots & \gamma_0
\end{pmatrix}.
\tag{2.1}
$$

A matrix whose entries depend only on $t - t'$ is **Toeplitz**: it is constant along every
diagonal and is fully specified by its first column (here also its first row, by symmetry).
Instead of $N^2$ numbers it carries $N$.

**Positive definiteness.** For any deterministic $w\in\mathbb{R}^N$,
$w^\top C w = \langle (w^\top n)^2\rangle \ge 0$, so $C$ is positive semidefinite. It is
positive *definite* unless some nontrivial linear combination of the samples is almost surely
zero, which never happens for a process with a strictly positive power spectral density on a
set of positive measure. `ringdown` estimates $\gamma$ from a PSD that is positive everywhere
(the ACF is the inverse Fourier transform of the PSD), so $C$ is positive definite and all
inverses, Cholesky factors and logarithms below exist.

**Persymmetry.** Besides being symmetric, a symmetric Toeplitz matrix is symmetric about its
anti-diagonal: with $J$ the exchange matrix, $(JCJ)_{tt'} = C_{N-1-t,\,N-1-t'} = \gamma_{|t-t'|}$,
so

$$
J C J = C, \qquad\text{hence}\qquad J C^{-1} J = C^{-1}.
\tag{2.2}
$$

This small fact is used twice below: to relate the last column of $C^{-1}$ to its first
(§4.3), and to relate the transpose of a lower-triangular Toeplitz matrix to a reversal
(§5.3).

**What the likelihood needs.** Equation (1.1) needs $C^{-1}$ applied to the residual (and, in
the marginalized branch, to the columns of $M_i$ and to $y_i$), and $\log|C|$. Neither
$C^{-1}$ nor $\log|C|$ depends on $\theta$; only the *vectors* they are applied to do. The
$\theta$-independent structure of $C$ can therefore be prepared once, outside the sampler.
The question is what to prepare. `main` prepares $L$; the PR prepares $(a, \sigma)$.

---

## 3. The autoregressive view: Yule–Walker, innovations, and the first column of $C^{-1}$

### 3.1 One-step linear prediction

Consider predicting the sample $n_t$ from the $p$ preceding samples $n_{t-1},\dots,n_{t-p}$
by a linear combination. Write the predictor in the form

$$
\hat n_t = -\sum_{j=1}^{p} a_j\, n_{t-j},
$$

so that the **prediction error** (also called the innovation or the prediction-error
filter output) is

$$
\varepsilon_t = n_t - \hat n_t = \sum_{j=0}^{p} a_j\, n_{t-j}, \qquad a_0 \equiv 1 .
\tag{3.1}
$$

The sign convention is chosen so that the filter $a = (1, a_1, \dots, a_p)$ acts on the
signal as a plain convolution. (The other common convention, $n_t = \sum_j \phi_j n_{t-j} +
\varepsilon_t$, has $\phi_j = -a_j$. The PR's `a_coeffs` follow the $a_j$ convention.)

The mean-square error $\langle\varepsilon_t^2\rangle$ is minimized when the error is
orthogonal to every regressor, $\langle\varepsilon_t n_{t-m}\rangle = 0$ for $m = 1,\dots,p$.
Expanding with (3.1) and stationarity,

$$
\sum_{j=0}^{p} a_j\,\gamma_{|m-j|} = 0, \qquad m = 1,\dots,p .
\tag{3.2}
$$

These are the **Yule–Walker equations**. Separating the $j=0$ term and using $a_0 = 1$,

$$
\sum_{j=1}^{p} \gamma_{|m-j|}\, a_j = -\gamma_m, \qquad m = 1,\dots,p,
\qquad\Longleftrightarrow\qquad
C_p\, a' = -\gamma' ,
\tag{3.3}
$$

where $C_p$ is the leading $p\times p$ block of $C$ (Toeplitz with first column
$(\gamma_0,\dots,\gamma_{p-1})$), $a' = (a_1,\dots,a_p)^\top$ and $\gamma' = (\gamma_1,\dots,\gamma_p)^\top$.

**This is exactly what the PR solves.** In `Fit.run_input` it sets `R = acf_vals[:-1]`
(the first column of $C_p$ with $p = N-1$), `r = acf_vals[1:]` ($=\gamma'$), and calls
`scipy.linalg.solve_toeplitz((R, R), -r)`, which returns $a'$. The order is the maximal
one, $p = N-1$: the last sample is predicted from all $N-1$ preceding ones.

### 3.2 The innovation variance

With $a$ the optimal filter, the mean-square prediction error is

$$
\sigma^2 \equiv \langle \varepsilon_t^2\rangle
= \sum_{j,l=0}^{p} a_j a_l\,\gamma_{|j-l|}
= \sum_{j=0}^{p} a_j\,\Big(\sum_{l=0}^{p} a_l\,\gamma_{|j-l|}\Big).
$$

By (3.2) the inner sum vanishes for $j = 1,\dots,p$, leaving only $j = 0$:

$$
\boxed{\;\sigma^2 = \sum_{l=0}^{p} a_l\,\gamma_l = \gamma_0 + \sum_{l=1}^{p} a_l\,\gamma_l\;}
\tag{3.4}
$$

which is the PR's `sigma_sq = acf_vals[0] + np.dot(a_coeffs, r)`. Because it is a
mean-square error, $\sigma^2 > 0$; because the predictor is optimal, $\sigma^2 \le \gamma_0$,
with equality only if the past carries no information about the present (white noise).

### 3.3 The first column of $C^{-1}$

Take $p = N-1$, so the filter $a\in\mathbb{R}^N$ has the same length as a data segment.
Equations (3.2) for $m = 1,\dots,N-1$ together with (3.4) for $m = 0$ say precisely that

$$
(Ca)_m = \sum_{j=0}^{N-1}\gamma_{|m-j|}a_j = \begin{cases}\sigma^2, & m = 0,\\ 0, & m = 1,\dots,N-1,\end{cases}
\qquad\text{i.e.}\qquad
\boxed{\;C a = \sigma^2 e_0 .\;}
\tag{3.5}
$$

Multiplying by $C^{-1}/\sigma^2$,

$$
\boxed{\; x \equiv C^{-1} e_0 = \frac{a}{\sigma^2}, \qquad x_0 = \frac{1}{\sigma^2} .\;}
\tag{3.6}
$$

**The prediction-error filter, normalized by the innovation variance, is the first column
of the inverse covariance.** This is the bridge between the statistical (autoregressive)
picture and the linear-algebraic one: the Gohberg–Semencul formula of §4 reconstructs all of
$C^{-1}$ from this one column, so the Yule–Walker solve is the *entire* preprocessing the PR
needs.

There is a second way to see (3.6) that will also give the determinant. Partition $C$ by its
first row and column,

$$
C = \begin{pmatrix}\gamma_0 & \gamma'^\top\\ \gamma' & C_{N-1}\end{pmatrix},
$$

and recall the block-inverse (bordering) formula: with the **Schur complement**
$\Delta \equiv \gamma_0 - \gamma'^\top C_{N-1}^{-1}\gamma'$,

$$
C^{-1} = \begin{pmatrix}0 & 0\\ 0 & C_{N-1}^{-1}\end{pmatrix}
+ \frac{1}{\Delta}\begin{pmatrix}1\\ -C_{N-1}^{-1}\gamma'\end{pmatrix}
\begin{pmatrix}1\\ -C_{N-1}^{-1}\gamma'\end{pmatrix}^{\!\top}.
\tag{3.7}
$$

(Direct multiplication by $C$ verifies it.) Comparing with (3.3), $-C_{N-1}^{-1}\gamma' = a'$,
so the bordering vector is exactly $a$; and reading off the $(0,0)$ entry, $x_0 = 1/\Delta$, so
$\Delta = \sigma^2$ by (3.6). Thus

$$
\boxed{\;C^{-1} = \begin{pmatrix}0 & 0\\ 0 & C_{N-1}^{-1}\end{pmatrix} + \frac{1}{x_0}\,x\,x^\top,
\qquad \sigma^2 = \gamma_0 - \gamma'^\top C_{N-1}^{-1}\gamma' .\;}
\tag{3.8}
$$

The second equality identifies the innovation variance with the Schur complement of the
leading block: the variance of $n_0$ left after conditioning on the other $N-1$ samples (by
stationarity, the same as the variance of the last sample given the preceding ones).

### 3.4 The Levinson–Durbin recursion and the sequence of innovation variances

Solving (3.3) by Gaussian elimination costs $O(N^3)$. The Toeplitz structure allows $O(N^2)$
via the **Levinson–Durbin recursion**, which is what `scipy.linalg.solve_toeplitz`
implements. It builds the order-$m$ predictor from the order-$(m-1)$ predictor for
$m = 1,\dots,N-1$. Write $a^{(m)} = (1, a^{(m)}_1, \dots, a^{(m)}_m)^\top$ for the order-$m$
filter and $\sigma_m^2$ for its innovation variance, with $a^{(0)} = (1)$ and
$\sigma_0^2 = \gamma_0$ (no past: the best prediction is zero and the error variance is the
process variance). The recursion is

$$
\kappa_m = -\frac{1}{\sigma_{m-1}^2}\Big(\gamma_m + \sum_{j=1}^{m-1} a^{(m-1)}_j\,\gamma_{m-j}\Big),
\qquad
a^{(m)} = \begin{pmatrix}a^{(m-1)}\\ 0\end{pmatrix} + \kappa_m\begin{pmatrix}0\\ J a^{(m-1)}\end{pmatrix},
\qquad
\sigma_m^2 = \sigma_{m-1}^2\,(1 - \kappa_m^2).
\tag{3.9}
$$

The scalar $\kappa_m$ is the **reflection coefficient** (in statistics, the partial
autocorrelation at lag $m$), and $|\kappa_m| < 1$ for a positive-definite $C$. Two facts
about this sequence matter below.

1. **Monotonicity.** $\sigma_m^2 = \sigma_{m-1}^2(1-\kappa_m^2) \le \sigma_{m-1}^2$: adding a
   regressor cannot worsen the optimal linear predictor. Hence
   $\gamma_0 = \sigma_0^2 \ge \sigma_1^2 \ge \dots \ge \sigma_{N-1}^2 = \sigma^2$, with
   equality at step $m$ if and only if $\kappa_m = 0$.

2. **Determinant.** The Schur determinant formula $|C_{m+1}| = |C_m|\cdot \Delta_{m+1}$, with
   $\Delta_{m+1}$ the Schur complement of the leading $m\times m$ block of $C_{m+1}$, together with the
   identification (3.8) of that Schur complement as the order-$m$ innovation variance, gives
   by induction

$$
\boxed{\;|C| = |C_N| = \prod_{m=0}^{N-1}\sigma_m^2,
\qquad
\log|C| = \sum_{m=0}^{N-1}\log\sigma_m^2 .\;}
\tag{3.10}
$$

   This is the exact log-determinant, and it is a by-product of the same recursion that
   produces $a$. It is also what the Cholesky route computes: the squared diagonal entries of
   $L$ are the successive Schur complements, $L_{mm}^2 = \sigma_m^2$, so
   $2\sum_m\log L_{mm} = \sum_m\log\sigma_m^2$. The whitened data $L^{-1}y$ of the reference
   note are, in this language, the standardized innovations of $y$.

---

## 4. The Gohberg–Semencul formula

### 4.1 Definitions

For $x\in\mathbb{R}^N$ let $L(x)$ be the **lower-triangular Toeplitz matrix with first
column $x$**,

$$
[L(x)]_{tt'} = \begin{cases} x_{t-t'}, & t\ge t',\\ 0, & t < t',\end{cases}
\qquad
L(x) = \begin{pmatrix} x_0 & & & \\ x_1 & x_0 & & \\ \vdots & & \ddots & \\ x_{N-1} & \cdots & x_1 & x_0\end{pmatrix}
= \sum_{j=0}^{N-1} x_j\, Z^j ,
\tag{4.1}
$$

where $Z$ is the lower shift matrix ($Z e_j = e_{j+1}$, $Z e_{N-1} = 0$, so $Z^j$ has ones
on the $j$-th subdiagonal and $Z^N = 0$). The polynomial expression shows that all $L(x)$
commute with one another and with $Z$, and that $L(x)e_0 = x$.

For $x\in\mathbb{R}^N$ define the **reversed-and-shifted vector**

$$
\tilde x \equiv Z J x = (0,\ x_{N-1},\ x_{N-2},\ \dots,\ x_1)^\top .
\tag{4.2}
$$

Note the leading zero and the absence of $x_0$: $\tilde x$ is $x$ with its first entry
removed, the remainder reversed, and a zero prepended. In the PR this is built as
`jnp.pad(ac[1:][::-1], (1, 0))` (drop `ac[0]`, reverse, pad one zero on the left), which is
exactly (4.2).

### 4.2 Statement

**Theorem (Gohberg–Semencul, symmetric case).** Let $C$ be a symmetric positive-definite
Toeplitz matrix, and let $x = C^{-1}e_0$ be the first column of its inverse (so $x_0 > 0$).
Then

$$
\boxed{\;C^{-1} = \frac{1}{x_0}\Big[\,L(x)\,L(x)^\top - L(\tilde x)\,L(\tilde x)^\top\,\Big].\;}
\tag{4.3}
$$

Substituting $x = a/\sigma^2$ from (3.6), and noting $L(a/\sigma^2) = L(a)/\sigma^2$ and
$\widetilde{a/\sigma^2} = \tilde a/\sigma^2$, the prefactor $1/x_0 = \sigma^2$ combines with
the two factors of $1/\sigma^4$ to give the form used in the PR:

$$
\boxed{\;C^{-1} = \frac{1}{\sigma^2}\Big[\,L(a)\,L(a)^\top - L(\tilde a)\,L(\tilde a)^\top\,\Big],
\qquad \tilde a = (0, a_{N-1}, \dots, a_1)^\top .\;}
\tag{4.4}
$$

This is the identity implemented by `apply_cinv_gs_fast`: `t1` is $L(a)L(a)^\top v$, `t2`
is $L(\tilde a)L(\tilde a)^\top v$, and the return value is `(t1 - t2) / sigma**2`.

Two remarks before the proof. First, (4.3) is an *exact* algebraic identity, not an
approximation, and it holds for any symmetric positive-definite Toeplitz matrix, not only for
covariances of finite-order AR processes. Second, the right-hand side is manifestly
symmetric, but it is a *difference* of two positive-semidefinite matrices; positive
definiteness of the result is guaranteed by the theorem, not visible term by term.

### 4.3 Proof via displacement structure

The proof has three short steps: a displacement operator that is injective; the displacement
of the right-hand side; the displacement of $C^{-1}$.

**Step 1: the displacement operator is injective.** Define, for any $N\times N$ matrix $X$,

$$
\nabla(X) \equiv X - Z X Z^\top .
\tag{4.5}
$$

$Z X Z^\top$ is $X$ moved one step down and one step to the right, with the first row and
first column filled with zeros. If $\nabla(X) = 0$ then $X = ZXZ^\top = Z^2X(Z^\top)^2 =
\dots = Z^N X (Z^\top)^N = 0$ because $Z^N = 0$. So $\nabla$ is injective (indeed bijective,
being a linear map of a finite-dimensional space to itself), and an $N\times N$ matrix is
uniquely determined by its displacement. Explicitly, iterating $X = \nabla(X) + ZXZ^\top$,

$$
X = \sum_{j=0}^{N-1} Z^j\,\nabla(X)\,(Z^\top)^j .
\tag{4.6}
$$

**Step 2: the displacement of $L(x)L(x)^\top$.** Since $L(x)$ commutes with $Z$, and since
$ZZ^\top = \mathbb{1} - e_0 e_0^\top$ (applying $Z^\top$ then $Z$, i.e. shifting up and then back down, kills only the first component),

$$
\nabla\big(L(x)L(x)^\top\big) = L(x)L(x)^\top - L(x)\,ZZ^\top L(x)^\top
= L(x)\big(\mathbb{1} - ZZ^\top\big)L(x)^\top
= L(x)\,e_0e_0^\top\,L(x)^\top = x\,x^\top ,
\tag{4.7}
$$

using $L(x)e_0 = x$. The same holds with $\tilde x$ in place of $x$. Hence the right-hand
side of (4.3) has displacement

$$
\nabla\Big(\tfrac{1}{x_0}\big[L(x)L(x)^\top - L(\tilde x)L(\tilde x)^\top\big]\Big)
= \frac{1}{x_0}\big(x x^\top - \tilde x\tilde x^\top\big).
\tag{4.8}
$$

**Step 3: the displacement of $C^{-1}$.** Bordering $C$ by its first row and column gave
(3.8):

$$
C^{-1} = \begin{pmatrix}0 & 0\\ 0 & C_{N-1}^{-1}\end{pmatrix} + \frac{1}{x_0}\,x x^\top .
\tag{4.9a}
$$

Bordering instead by its *last* row and column, the same block-inverse formula gives, with
$\bar x \equiv C^{-1}e_{N-1}$ the last column of $C^{-1}$ and $\bar x_{N-1}$ its last entry,

$$
C^{-1} = \begin{pmatrix}C_{N-1}^{-1} & 0\\ 0 & 0\end{pmatrix} + \frac{1}{\bar x_{N-1}}\,\bar x \bar x^\top ,
\tag{4.9b}
$$

where the leading block is again $C_{N-1}^{-1}$ because, for a Toeplitz matrix, deleting the
last row and column gives the same matrix as deleting the first. By persymmetry (2.2),
$\bar x = C^{-1}Je_0 = JC^{-1}e_0 = Jx$, so $\bar x_{N-1} = x_0$ and $\bar x\bar x^\top = (Jx)(Jx)^\top$.

Now observe that $Z\begin{pmatrix}C_{N-1}^{-1} & 0\\ 0 & 0\end{pmatrix}Z^\top =
\begin{pmatrix}0 & 0\\ 0 & C_{N-1}^{-1}\end{pmatrix}$: conjugating by $Z$ moves the block
from the top-left corner to the bottom-right. Applying $Z(\cdot)Z^\top$ to (4.9b) and
subtracting from (4.9a) therefore eliminates $C_{N-1}^{-1}$:

$$
C^{-1} - Z C^{-1} Z^\top = \frac{1}{x_0}\,xx^\top - \frac{1}{x_0}\,(ZJx)(ZJx)^\top
= \frac{1}{x_0}\big(xx^\top - \tilde x\tilde x^\top\big),
\tag{4.10}
$$

using $\tilde x = ZJx$ from (4.2). This is the same as (4.8). By Step 1, two matrices with
the same displacement are equal, which proves (4.3). $\blacksquare$

### 4.4 Displacement rank, and why $O(N)$ numbers suffice

Equation (4.10) says that $\nabla(C^{-1})$ has rank at most $2$: one says $C^{-1}$ has
**displacement rank 2** (with respect to $\nabla$). $C$ itself also has displacement rank
at most 2, since $\nabla(C) = C - ZCZ^\top$ is nonzero only in its first row and column
(the shifted copy reproduces every entry $C_{tt'}$ with $t,t'\ge1$). The inverse of a Toeplitz
matrix is generally *not* Toeplitz, but it inherits this low displacement rank, and that is
the structural content of the theorem: while $C^{-1}$ has $N^2$ distinct entries, it is
determined through (4.6) by the $2N$ numbers in $x$ and $\tilde x$ (in fact by the $N$
numbers in $x$ alone, since $\tilde x$ is a rearrangement of $x$). The Cholesky factor $L$,
by contrast, is dense lower-triangular and carries $N(N+1)/2$ independent numbers.

The formula (4.3) is one convenient way to turn the displacement generators into an operator
that can be *applied*: each term is a product of two lower-triangular Toeplitz matrices,
and §5 shows that such matrices act in $O(N\log N)$ time.

---

## 5. Applying lower-triangular Toeplitz matrices with FFTs

### 5.1 Lower-triangular Toeplitz products are causal convolutions

From (4.1), for $\psi\in\mathbb{R}^N$,

$$
\big[L(x)\psi\big]_t = \sum_{t'=0}^{t} x_{t-t'}\,\psi_{t'} = (x * \psi)_t, \qquad t = 0,\dots,N-1,
\tag{5.1}
$$

where $x * \psi$ denotes the linear (aperiodic) convolution of the two length-$N$ sequences,
which has length $2N-1$ with entries $(x*\psi)_t = \sum_{t'} x_{t-t'}\psi_{t'}$ for
$t = 0,\dots,2N-2$ (indices outside $[0,N-1]$ read as zero). The matrix product is the first
$N$ entries of the full convolution. It is *causal* in the sense that output $t$ depends only
on inputs $t'\le t$, which is just the lower-triangularity of $L(x)$.

### 5.2 Linear convolution by zero-padded FFT, and the minimal length

The discrete Fourier transform of length $n_{\rm fft}$ diagonalizes *circular* convolution:
if $\hat x$ and $\hat\psi$ are the DFTs of $x$ and $\psi$ zero-padded to length $n_{\rm fft}$, the
inverse DFT of the pointwise product $\hat x\hat\psi$ is the circular convolution

$$
(x \circledast \psi)_t = \sum_{t'} x_{(t - t')\bmod n_{\rm fft}}\, \psi_{t'} .
$$

The circular convolution equals the linear one wherever no index wraps around, i.e. the
linear convolution's entry at $t + n_{\rm fft}$ must vanish for every $t$ that is read off.
The linear convolution of two length-$N$ sequences occupies indices $0,\dots,2N-2$. We read
off indices $0,\dots,N-1$. Index $t$ is contaminated by index $t + n_{\rm fft}$ if
$t + n_{\rm fft}\le 2N-2$, i.e. if $t \le 2N-2-n_{\rm fft}$; the set of contaminated $t$ is
empty precisely when

$$
\boxed{\;n_{\rm fft} \ge N + P - 1 = 2N - 1,\;}
\tag{5.2}
$$

where $P$ is the filter length, here $P = N$ because the AR filter $a$ has the same length
as the data. (In general, for a filter of length $P < N$, $N + P - 1$ suffices.) Any
$n_{\rm fft}$ at or above this bound gives the exact linear convolution in the first $N$
entries; a shorter one does not (§10 confirms this numerically: with $n_{\rm fft} = N$ the
errors are of order unity).

The PR sets `P = ar_coeffs[0].shape[0]` ($= N$) and `n_fft = next_fast_len(n_time + P - 1)`,
where its `next_fast_len` returns the smallest power of two that is at least its argument.
For $N = 205$: $2N - 1 = 409$ and $n_{\rm fft} = 512$. Since a power of two at or above
$2N-1$ satisfies (5.2), the padding is sufficient; it is not minimal (SciPy's function of the
same name, `scipy.fft.next_fast_len`, would return $2^a3^b5^c7^d11^e$ lengths, $420$ here, or $432$ with `real=True`), but any length satisfying (5.2)
gives the same answer up to roundoff, and powers of two are the most reliably fast lengths
on every FFT backend.

`apply_matrix_fft_precomputed(x, filter_fft, N, n_fft)` is then exactly: real FFT of the
zero-padded input, pointwise product with the precomputed filter transform, inverse real
FFT, truncate to the first $N$ entries. The filter transforms $\hat a$ (`fft_as[i]`) and
$\hat{\tilde a}$ (`fft_bs[i]`) are computed once per detector at the top of the model
function.

### 5.3 The transpose via reversal

$L(x)^\top$ is *upper*-triangular Toeplitz, and applying it is an anti-causal convolution.
Rather than a second filter transform, the PR uses

$$
\boxed{\;L(x)^\top \psi = J\,L(x)\,J\,\psi = \mathrm{flip}\big(L(x)\,\mathrm{flip}(\psi)\big).\;}
\tag{5.3}
$$

*Proof.* $(JL(x)J)_{tt'} = [L(x)]_{N-1-t,\,N-1-t'}$, which by (4.1) is $x_{(N-1-t)-(N-1-t')} =
x_{t'-t}$ when $N-1-t \ge N-1-t'$, i.e. when $t'\ge t$, and $0$ otherwise. That is exactly
$[L(x)^\top]_{tt'}$. $\blacksquare$

This is an exact identity, valid for any Toeplitz matrix (transpose equals reversal
conjugate), and costs two array reversals. It is implemented as `apply_At` and `apply_Bt`.

### 5.4 The full application

Putting §§4–5 together, `apply_cinv_gs_fast(vector, fft_a, fft_b, n_fft, sigma)`, with
`vector` $= \psi$, computes

$$
C^{-1}\psi = \frac{1}{\sigma^2}\Big[\, L(a)\,\big(J L(a) J \psi\big) - L(\tilde a)\,\big(J L(\tilde a) J \psi\big)\Big]
\tag{5.4}
$$

with each of the four $L(\cdot)$ applications a padded FFT convolution: four forward and four
inverse real FFTs of length $n_{\rm fft}$, plus $O(N)$ pointwise work. For a matrix
right-hand side (the $k$ columns of $M_i$) the PR maps this over columns with `jax.vmap`,
which the FFT backend executes as a batched transform.

---

## 6. Where $C^{-1}$ enters the marginalized likelihood

### 6.1 The quantities

The reference note shows that the marginal (amplitude-integrated) likelihood of the network
is a function of the following per-detector contractions with $C_i^{-1}$:

$$
M_i^\top C_i^{-1} M_i \ \ (k\times k),\qquad
M_i^\top C_i^{-1} y_i\ \ (k),\qquad
y_i^\top C_i^{-1} y_i\ \ (\text{scalar}),\qquad
\log|C_i|\ \ (\text{scalar}),
\tag{6.1}
$$

which on `main` are accumulated as

$$
A^{-1} = \mathbb{1}_k + \sum_i M_i^\top C_i^{-1}M_i, \qquad
v = \sum_i M_i^\top C_i^{-1}y_i, \qquad
Q = \sum_i y_i^\top C_i^{-1}y_i ,
$$

followed by one $k\times k$ Cholesky $A^{-1} = RR^\top$, $u = R^{-1}v$, and (reference
note eq. 5.1)

$$
\log p(y\mid\theta) + \text{const} = -\tfrac12 Q + \tfrac12\lVert u\rVert^2 - \tfrac12\sum_i\log|C_i| - \sum_{j=1}^k\log R_{jj} .
\tag{6.2}
$$

The PR was written against the older *sequential* scheme (reference note §3), in which
detector $i$ is processed with a prior $N(\mu, \Lambda)$ inherited from the previous
detectors, and the per-detector factor is (reference note eq. 3.4)

$$
\ell_i = -\tfrac12\, r_i^\top\Big[C_i^{-1} - C_i^{-1}M_i A^{(i)} M_i^\top C_i^{-1}\Big] r_i
- \Big[\tfrac12\log|C_i| - \textstyle\sum_j\log R^{(i-1)}_{jj} + \sum_j\log R^{(i)}_{jj}\Big],
\qquad r_i = y_i - M_i\mu ,
\tag{6.3}
$$

where $A^{(i)} = (\Lambda^{-1} + M_i^\top C_i^{-1}M_i)^{-1}$, $R^{(i)}$ is the lower Cholesky
factor of $[A^{(i)}]^{-1}$ (so $R^{(i-1)}$ is that of the prior precision $\Lambda^{-1}$ carried
into step $i$, the code's `Lambda_inv_chol`, with $R^{(0)} = \mathbb{1}$), and the bracketed determinant
combination is the code's `log_sqrt_det_B`. The matrix in square brackets is the Woodbury
form of $B_i^{-1}$, $B_i = C_i + M_i\Lambda M_i^\top$.

### 6.2 How the PR computes each of them

Reading the diff of the marginalized branch:

| Quantity | `main` (Cholesky) | PR #141 (Gohberg–Semencul) |
|---|---|---|
| $C_i^{-1}M_i$ | `cho_solve((L, True), M)` (sequential) or $W_i = L^{-1}M_i$ (one-shot) | `Cinv_M = vmap(apply_cinv_gs_fast)(M)` over the $k$ columns |
| $M_i^\top C_i^{-1}M_i$ | `M.T @ cho_solve(...)` or $W_i^\top W_i$ | `M.T @ Cinv_M` |
| $C_i^{-1}y_i$ | `cho_solve((L, True), y)` or $z_i = L^{-1}y_i$ | `Cinv_y = apply_cinv_gs_fast(y, ...)` |
| $C_i^{-1}r_i$ | `cho_solve((L, True), r)` | `Cinv_r = apply_cinv_gs_fast(r, ...)` |
| Woodbury correction | $r^\top C^{-1}M A M^\top C^{-1} r$ via a second `cho_solve` against $L$ | $(M^\top C^{-1}r)^\top A\,(M^\top C^{-1}r)$ with one `cho_solve` against `A_inv_chol` |
| $\tfrac12\log\lvert C_i\rvert$ | $\sum_t\log L_{tt}$ | $N\log\sigma_i$ (see §8) |

The Woodbury rearrangement deserves a word. `main`'s sequential code computed
$C_i^{-1}\big(M_i A^{(i)} M_i^\top C_i^{-1} r_i\big)$ and then contracted with $r_i$. The PR
instead forms $w \equiv M_i^\top C_i^{-1}r_i\in\mathbb{R}^k$ once and computes
$w^\top A^{(i)} w$. These are equal because $C_i^{-1}$ is symmetric:

$$
r_i^\top C_i^{-1}M_i A^{(i)} M_i^\top C_i^{-1} r_i
= \big(M_i^\top C_i^{-1} r_i\big)^\top A^{(i)}\big(M_i^\top C_i^{-1} r_i\big) = w^\top A^{(i)} w .
$$

This is an exact identity and saves one application of $C_i^{-1}$ per detector. It is the
same observation that, in the one-shot form, makes the $\tfrac12\lVert u\rVert^2$ term of
(6.2) a $k$-dimensional quantity.

The non-marginalized branch (explicit amplitudes, `MultivariateNormal` on `main`) is
changed analogously: the residual $r = y_i - h_i$ is formed, $C_i^{-1}r$ is obtained from
`apply_cinv_gs_fast`, and $-\tfrac12 r^\top C_i^{-1}r - \tfrac12\log|C_i|$ is added as a
`numpyro.factor`, with $\log|C_i|$ again replaced by $2N\log\sigma_i$.

### 6.3 What does and does not depend on $\theta$

Exactly as in the reference note §5.1: $C_i^{-1}y_i$, $y_i^\top C_i^{-1}y_i$ and $\log|C_i|$
are $\theta$-independent constants (the PR nonetheless recomputes `Cinv_y` in the model
body every call, as `main` does for $z_i$). The genuinely per-sample work is $C_i^{-1}M_i$,
$k$ right-hand sides per detector, plus in the sequential scheme $C_i^{-1}r_i$, whose
right-hand side depends on $\theta$ through $\mu$.

---

## 7. Operation counts, stated honestly

Let $N$ be the number of samples and $n_{\rm rhs}$ the number of vectors to which $C^{-1}$ is
applied per detector per likelihood evaluation ($n_{\rm rhs} = k + 1$ in the one-shot
scheme, $k + 2$ in the sequential one).

**Cholesky route (`main`).** Once, outside the sampler: a dense Cholesky factorization,
$\tfrac13 N^3$ flops, about $3\times10^6$ for $N = 205$; this is negligible and is not
differentiated. Per evaluation: one triangular solve per right-hand side, $N^2$ flops each
(or $2N^2$ for a full `cho_solve`), so about $N^2 n_{\rm rhs}\approx 4\times10^5$ flops for
$k = 8$. Memory: $N^2/2$ numbers for $L$.

**Gohberg–Semencul route (PR).** Once, outside the sampler: the Levinson recursion,
$O(N^2)$. Per evaluation and per right-hand side: eight real FFTs of length $n_{\rm fft}$
(four forward, four inverse), each roughly $\tfrac52 n_{\rm fft}\log_2 n_{\rm fft}$ flops,
plus $O(n_{\rm fft})$ pointwise work. For $n_{\rm fft} = 512$ that is of order
$8\times 2.5\times512\times9\approx 9\times10^4$ flops per right-hand side, about
$8\times10^5$ for $n_{\rm rhs} = 9$. Memory: $O(N)$ for $a$, $\sigma$, and the two filter
transforms.

**Asymptotically** $O(N\log N)$ beats $O(N^2)$ per right-hand side, and the memory footprint
drops from $O(N^2)$ to $O(N)$. **For $N\approx200$**, however, the estimates above are within
a factor of a few of each other, and which one wins is decided by constants that the flop
count does not see: the efficiency of the batched FFT kernels versus the triangular-solve
kernels on the backend in use (CPU or GPU, float32 or float64), the cost of the eight
padding/truncation/reversal passes, and what XLA does with the eight FFTs of a
$\theta$-independent right-hand side. FFTs parallelize well on GPUs and triangular solves do
not, so the balance may differ between backends. `docs/dev/model_optimization_study.md`
records that on `main` the per-gradient cost is *dominated* by the triangular solves on
$n_t\times n_t$ matrices, which are block-sequential by construction (its §6 attributes the
poor GPU utilization at $n_t = 205$ to exactly this), and that the one-shot marginalization's
speedup came largely from cutting the number of `dtrsm` calls per gradient from 42 to 8. That
makes the $C^{-1}$ application a natural next target, and is presumably the PR's motivation;
it does not by itself say that FFTs at $n_{\rm fft} = 512$ beat eight triangular solves at
$N = 205$. **None of this is settled by this note**; it must be measured, per backend and precision, in the way
`docs/dev/model_optimization_study.md` measured the marginalization change.

Two structural points are independent of timing. First, the GS route never forms the
whitened design $W_i = L^{-1}M_i$; the reference note §5.2 explains why `main` prefers the
Gram form $W_i^\top W_i$, which is exactly symmetric in floating point and has condition
number $\mathrm{cond}(W_i)^2$, over the product $M_i^\top(C_i^{-1}M_i)$, which is not exactly
symmetric. A GS-based implementation would have to symmetrize explicitly or accept the
asymmetry. Second, and independently of speed, post-processing needs its own access to the
noise model. `Result.cholesky_factors` reads `constant_data.L` or
`constant_data.cholesky_factor` from the stored result, and `Result.whiten`,
`whitened_data`, `whitened_templates`, `whitened_residuals`, `whitened_injection`,
`compute_posterior_snrs` and `loo` all go through it. The PR's `get_arviz` no longer stores
`cholesky_factor` (and stores `sigma` but not `ar_coeffs`), so a `Result` produced by the PR
carries no representation of $C_i$ at all and those methods fail; see (D3) in §9.1. This is
a consequence of changing the model's inputs rather than of the mathematics, and any
GS-based implementation would have to either keep storing $L_i$ (which `Fit.cholesky_factors`
still computes) or teach `Result` to whiten from $(a, \sigma)$.

---

## 8. The log-determinant: exact expression versus what the PR implements

### 8.1 Exact

From (3.10),

$$
\log|C| = \sum_{m=0}^{N-1}\log\sigma_m^2 ,
\tag{8.1}
$$

the sum of the logarithms of the innovation variances of the predictors of order
$0, 1, \dots, N-1$. Equivalently $\log|C| = 2\sum_m \log L_{mm}$, which is what `main`
computes.

### 8.2 Implemented

The PR sets `log_det_C = 2.0 * len(y) * jnp.log(sigma)`, i.e.

$$
\log|C|_{\rm PR} = 2N\log\sigma = N\log\sigma_{N-1}^2 .
\tag{8.2}
$$

Its code comment reads "log|C| approx 2 * N * log(sigma) for AR process".

### 8.3 Comparison

Subtracting,

$$
\log|C| - \log|C|_{\rm PR} = \sum_{m=0}^{N-1}\big(\log\sigma_m^2 - \log\sigma_{N-1}^2\big)
= \sum_{m=0}^{N-1}\log\frac{\sigma_m^2}{\sigma_{N-1}^2}
= -\sum_{m=0}^{N-1}\ \sum_{j=m+1}^{N-1}\log\big(1-\kappa_j^2\big) \ \ge\ 0 .
\tag{8.3}
$$

The inequality follows from the monotonicity $\sigma_m^2 \ge \sigma_{N-1}^2$ of §3.4. Hence:

* **They are not equal in general.** $2N\log\sigma$ is a lower bound on $\log|C|$.
* **They agree if and only if every reflection coefficient vanishes**, $\kappa_m = 0$ for
  all $m$, i.e. $\sigma_m^2 = \gamma_0$ for all $m$, i.e. $\gamma_j = 0$ for all $j\ge1$:
  **white noise**. For an exact AR($p$) process with $p < N$ the innovation variances are
  constant only *from order $p$ on*, $\sigma_p^2 = \sigma_{p+1}^2 = \dots = \sigma_{N-1}^2$,
  so even then the discrepancy is $\sum_{m<p}\log(\sigma_m^2/\sigma^2) > 0$; the
  "approximately, for an AR process" of the code comment is accurate only in the sense that
  the *per-sample* discrepancy $\tfrac1N\sum_{m<p}\log(\sigma_m^2/\sigma^2)$ tends to zero
  as $N\to\infty$ at fixed $p$. For the colored spectra `ringdown` deals with, the
  effective $p$ is not small compared with $N$ (§10 gives numbers).
* **The discrepancy does not depend on $\theta$.** Both $\sigma_m^2$ and $\sigma^2$ are
  functions of the ACF alone. In the marginalized branch it enters `log_sqrt_det_B` as
  $\tfrac12\log|C_i|$, and in the non-marginalized branch as $-\tfrac12\log|C_i|$; in both
  cases it is an additive constant in the log-likelihood, of the same kind as the
  $-\tfrac N2\log2\pi$ that the code already drops. **It therefore cannot bias the posterior
  on $\theta$ or affect the sampler in any way.** It does shift the *absolute value* of the
  log-likelihood reported at each sample (the `logl_i` sites), by a constant that differs
  between detectors and between noise estimates. Anything that consumes that absolute value
  is affected: evidence or Bayes-factor estimates built from the likelihood values, and any
  comparison of likelihoods across runs with different ACFs or different $N$. The
  `Result.draw_sample(map=True)` selection uses `sample_stats.lp` when present and otherwise
  the argmax of the summed `logl_` sites; either way it is unaffected, since a constant does
  not move an argmax.
* **The exact quantity costs nothing extra.** The Levinson recursion that `solve_toeplitz`
  runs produces every $\sigma_m^2$ along the way (SciPy does not expose them, but a
  twenty-line Python implementation does, and so does the diagonal of the dense Cholesky
  factor that `Fit.cholesky_factors` already computes). Since $\log|C_i|$ is
  $\theta$-independent, it can be evaluated once in `run_input` and passed in as a scalar, at
  no per-sample cost.

---

## 9. Where the implementation and the exact mathematics part ways

This section lists every place in the diff where the formula implemented differs from the
exact expression it stands in for, and every place where one might suspect a difference but
there is none. Numerical results referred to are those of §10.

### 9.1 Differences

**(D1) The log-determinant.** Implemented: $2N\log\sigma_{N-1}$. Exact:
$\sum_{m=0}^{N-1}\log\sigma_m^2$. As shown in §8.3 the implemented value is a lower bound
that equals the exact value only for white noise. The difference is a $\theta$-independent
constant per detector, so the sampled posterior is unaffected; the reported absolute
log-likelihood and any evidence estimate are shifted. In the test cases of §10 with
$N = 205$ the shift ranges from $3.6$ nats (an AR(2) process) to $22$–$38$ nats (colored
spectra of the kind `ringdown` encounters).

**(D2) The $-\tfrac N2\log2\pi$ constant in the non-marginalized branch.** On `main` this
branch uses `dist.MultivariateNormal(...).log_prob`, whose value includes $-\tfrac N2\log2\pi$.
The PR's replacement `numpyro.factor` omits it. This is again a $\theta$-independent constant,
harmless for sampling, and it brings the non-marginalized branch into line with the
marginalized one, which has always dropped it; it is listed because it changes the
absolute value of the `logl_i` sites in that branch. A side effect that is not about the
value: replacing `numpyro.sample(..., obs=strain)` by `numpyro.factor` changes the site
type, so the strain is no longer an observed sample site of the model, which matters to
anything that enumerates observed sites (for instance `numpyro.infer.Predictive`).

**(D3) The stored result loses the noise model.** Not a formula, but a difference between
what the code computed before and what it can compute after. On `main`, `get_arviz` stores the
Cholesky factors as `constant_data.cholesky_factor` (dimensions `["ifo", "time_index",
"time_index_1"]`), and `Result.cholesky_factors` (`ringdown/result.py`) returns
`constant_data.L` or `constant_data.cholesky_factor`. Every whitening-based method of
`Result` (`whiten`, `whitened_data`, `whitened_templates`, `whitened_residuals`,
`whitened_injection`, `compute_posterior_snrs`, `loo`) depends on it. The PR's `get_arviz`
removes `cholesky_factor` from `in_dims` and `in_data`, stores `sigma` with dimension
`["ifo"]`, leaves the line that would store `ar_coeffs` commented out, and yet still declares
`"ar_coeffs": ["ifo", "ar_lag"]` in `in_dims` with no corresponding data and no `ar_lag`
coordinate. A `Result` produced by the PR therefore has no representation of $C_i$, neither
$L_i$ nor $(a, \sigma)$, and the methods above cannot run on it. The likelihood itself is
unaffected; the loss is in what can be done with the output afterwards. The remedy is
mechanical (store $L_i$, which `Fit.cholesky_factors` still provides, or store
$(a_i, \sigma_i)$ and implement whitening from them via §3.4's identification of $L_i^{-1}y_i$
with the standardized innovations), but it is not in the PR.

### 9.2 Suspected differences that are in fact exact

**(E1) The first column of $C^{-1}$.** $x = a/\sigma^2$ with $a$ from Yule–Walker and
$\sigma^2 = \gamma_0 + a'^\top\gamma'$ is exact (§3.3); confirmed numerically to relative
$10^{-15}$ for well-conditioned $C$.

**(E2) The Gohberg–Semencul formula in the form $(1/\sigma^2)[L(a)L(a)^\top -
L(\tilde a)L(\tilde a)^\top]$.** Exact (§4.2–4.3).

**(E3) The construction of $\tilde a$.** `jnp.pad(ac[1:][::-1], (1, 0))` produces
$(0, a_{N-1}, \dots, a_1)$, which is exactly the $\tilde x = ZJx$ of (4.2). The leading zero
and the omission of $a_0$ are both essential: the variant $(a_{N-1},\dots,a_1,0)$, i.e.
without the leading zero, gives errors of order unity (§10, row "b lacking lead zero").

**(E4) Padding.** $n_{\rm fft}$ is the smallest power of two at or above $2N - 1$, which
satisfies the minimal requirement (5.2). Exact; the wrap-around that occurs with
$n_{\rm fft} = N$ is demonstrated in §10 to produce order-unity errors, confirming that (5.2)
is the correct bound and that the PR's choice respects it.

**(E5) The transpose via `flip`.** $L(x)^\top \psi = \mathrm{flip}(L(x)\,\mathrm{flip}(\psi))$ is an
exact identity (§5.3); confirmed to $10^{-16}$.

**(E6) The Woodbury rearrangement.** $w^\top A w$ with $w = M^\top C^{-1}r$ equals
$r^\top C^{-1}MAM^\top C^{-1}r$ exactly (§6.2).

**(E7) Strain rescaling.** The PR divides the ACF by $s^2$ before the Yule–Walker solve.
Scaling $\gamma\to\gamma/s^2$ leaves $a$ unchanged (the Yule–Walker system is homogeneous
in $\gamma$) and scales $\sigma^2\to\sigma^2/s^2$, which is exactly the covariance of the
rescaled data $y/s$. Consistent with `main`, which computes `.cholesky` of the rescaled ACF.

### 9.3 Floating-point remarks (not discrepancies in the mathematics)

**(F1) Stability of the Levinson solve.** `solve_toeplitz` uses the Levinson–Durbin
recursion, which is weakly stable: its backward error grows with $\mathrm{cond}(C)$ faster
than that of Cholesky. In the tests of §10, at $\mathrm{cond}(C)\approx1.7\times10^9$, the
first column of $C^{-1}$ was recovered to relative $5\times10^{-9}$ and the GS-applied
$C^{-1}\psi$ agreed with the exact solve to $3\times10^{-8}$, essentially the same as the
Cholesky solve's $4\times10^{-8}$. No accuracy gap was observed in these cases; this is an
observation about the test cases, not a theorem.

**(F2) Cancellation in the GS difference.** $C^{-1}\psi$ is computed as the difference of
two terms each of which is a positive-semidefinite operator applied to $\psi$. If the two terms
were much larger than their difference, digits would be lost. In the tests of §10 the ratio
$\lVert L(a)L(a)^\top \psi\rVert / \lVert\sigma^2 C^{-1}\psi\rVert$ was between $1.0$ and $1.1$ for
every case tried, i.e. no significant cancellation occurred. Whether this holds for every
ACF `ringdown` produces has not been established here.

**(F3) Symmetry of $M^\top C^{-1}M$.** Computed as `M.T @ Cinv_M`, this is symmetric only
up to roundoff, whereas `main`'s $W^\top W$ is exactly symmetric (reference note §5.2). The
$k\times k$ Cholesky that follows is defined for the symmetric part; in practice
`jsp.linalg.cholesky` reads one triangle, so the asymmetry is silently discarded rather than
propagated.

**(F4) A docstring.** `apply_cinv_gs_fast` is documented as computing "$x^\top C^{-1}y$";
it returns the vector $C^{-1}v$. The code is correct; the docstring is not.

---

## 10. Numerical checks

The identities above were checked with a standalone NumPy/SciPy script that reproduces the
PR's arithmetic (its Yule–Walker call, its construction of $\tilde a$, its `next_fast_len`,
and a line-by-line transcription of `apply_cinv_gs_fast`) and compares against dense linear
algebra, all in float64 with $N = 205$. Five ACFs were used:

1. an exact AR(2) process with $\phi = (1.6, -0.8)$ ($\mathrm{cond}(C) = 1.4\times10^3$);
2. white noise;
3. a smooth colored PSD, unit floor plus $(f/20\,\mathrm{Hz})^{-4}$ wall, at 2048 Hz
   ($\mathrm{cond}(C) = 1.7\times10^9$);
4. the same with a narrow spectral line added at 60 Hz ($\mathrm{cond}(C) = 1.7\times10^9$);
5. a random positive PSD ($\mathrm{cond}(C) = 70$).

Norms: for vector results (rows 4, 5, 6) "rel. err." is the Euclidean norm of the error
divided by the Euclidean norm of the exact vector, for one fixed standard-normal right-hand
side; for matrix and column results (rows 1, 2, 3, 7) it is the largest absolute entry of
the error divided by the largest absolute entry of the exact result.

| Check | AR(2) | white | colored | colored + line | random PSD |
|---|---|---|---|---|---|
| $x = a/\sigma^2$ vs first column of `inv(C)`, rel. err. | $3\times10^{-15}$ | $2\times10^{-16}$ | $5\times10^{-9}$ | $7\times10^{-9}$ | $5\times10^{-16}$ |
| dense GS (4.4) vs `inv(C)`, rel. err. | $6\times10^{-14}$ | $2\times10^{-16}$ | $1\times10^{-8}$ | $1\times10^{-8}$ | $9\times10^{-16}$ |
| dense GS with $\tilde a$ lacking the leading zero | $0.61$ | $2\times10^{-16}$ | $7\times10^{-3}$ | $8\times10^{-2}$ | $1\times10^{-2}$ |
| FFT route as coded vs exact solve, rel. err. | $3\times10^{-14}$ | $4\times10^{-16}$ | $3\times10^{-8}$ | $3\times10^{-8}$ | $2\times10^{-15}$ |
| Cholesky `cho_solve` vs exact solve, rel. err. | $4\times10^{-14}$ | $2\times10^{-16}$ | $4\times10^{-8}$ | $4\times10^{-8}$ | $7\times10^{-16}$ |
| FFT route with $n_{\rm fft} = N$ (wrap-around) | $0.65$ | $5\times10^{-16}$ | $0.29$ | $0.31$ | $0.45$ |
| $L^\top v$ vs $\mathrm{flip}(L\,\mathrm{flip}\,v)$ | $2\times10^{-16}$ | $0$ | $6\times10^{-16}$ | $6\times10^{-16}$ | $5\times10^{-16}$ |
| $\sigma^2_{N-1}$ (Levinson) vs PR's $\sigma^2$, rel. diff. | $2\times10^{-15}$ | $2\times10^{-16}$ | $5\times10^{-9}$ | $1\times10^{-8}$ | $1\times10^{-15}$ |
| $\sum_m\log\sigma_m^2$ minus `slogdet`, absolute | $8\times10^{-15}$ | $3\times10^{-13}$ | $4\times10^{-9}$ | $5\times10^{-7}$ | $2\times10^{-13}$ |
| $2N\log\sigma$ minus `slogdet`, absolute (nats) | $-3.60$ | $0.00$ | $-22.44$ | $-27.18$ | $-37.95$ |
| `slogdet` value, for scale | $3.60$ | $142.10$ | $1461.16$ | $1517.06$ | $386.54$ |

Reading the table:

* Rows 1, 2, 4, 5 confirm (3.6), (4.4) and the FFT implementation, with the GS route and the
  Cholesky route losing digits at the *same* rate as $\mathrm{cond}(C)$ grows.
* Rows 3 and 6 show that the two structural choices the PR gets right, the leading zero in
  $\tilde a$ and the padding to at least $2N-1$, are load-bearing: getting either wrong is
  not a small error.
* Row 9 confirms (3.10) to roundoff, and row 8 confirms that the PR's $\sigma^2$ is exactly
  the last innovation variance.
* Row 10 quantifies (D1). For the AR(2) case the discrepancy is exactly
  $\log\sigma_0^2 + \log\sigma_1^2 - 2\log\sigma^2 = \log(13.235) + \log(2.778) - 0 = 3.6045$,
  as (8.3) predicts for $p = 2$. For the colored spectra it is tens of nats: the effective
  AR order of a steep low-frequency wall is large, so many $\sigma_m^2$ sit well above
  $\sigma_{N-1}^2$ (here $\sigma^2/\gamma_0\approx10^{-7}$). For white noise it is zero, as
  it must be. The sign is negative in every case, as (8.3) requires.

The script is not part of the repository; the numbers are reproducible from the formulas in
this note.

---

## 11. Relation to the current `main` branch

PR #141 was written against the sequential per-detector marginalization (reference note
§3); `main` has since moved to the one-shot form (reference note §§4–5), so the diff does not
apply cleanly. The two changes are orthogonal, and it is worth saying precisely how.

**The marginalization scheme decides which contractions with $C_i^{-1}$ are needed.** The
one-shot form needs, per detector, $C_i^{-1}M_i$ (for $A^{-1}$ and, via $M_i^\top C_i^{-1}y_i$,
for $v$), $C_i^{-1}y_i$ (for $v$ and $Q$), and $\log|C_i|$. It does *not* need $C_i^{-1}r_i$
or the Woodbury correction, because the $\mu$-dependent residuals cancel identically
(reference note §4). So in the one-shot form the GS route would be called on $k+1$
right-hand sides per detector, not $k+2$, and the Woodbury rearrangement of §6.2 becomes
moot.

**The method of applying $C_i^{-1}$ decides how each contraction is evaluated.** Whether one
whitens with $L_i^{-1}$ or applies the GS operator changes nothing in the algebra of the
reference note: $M_i^\top C_i^{-1}M_i$ is the same $k\times k$ matrix either way, $A^{-1}$,
$R$, $u$ and the predictive draw $\alpha = R^{-\top}(u+\xi)$ are unchanged. Concretely, in
`main`'s loop

```python
W = solve_triangular(L, M, lower=True);  z = solve_triangular(L, y, lower=True)
A_inv += W.T @ W;  v += W.T @ z;  Q += z @ z;  logdetL += sum(log(diag(L)))
```

a GS version would read

```python
CiM = vmap(apply_cinv_gs_fast, in_axes=1, out_axes=1)(M, ...);  Ciy = apply_cinv_gs_fast(y, ...)
A_inv += M.T @ CiM;  v += M.T @ Ciy;  Q += y @ Ciy;  half_logdet += 0.5 * logdetC_i
```

with `logdetC_i` a precomputed scalar (§8.3 explains why it should be the exact
$\sum_m\log\sigma_m^2$ and why that costs nothing). The three points where the two differ in
character are the ones already noted: the GS product $M^\top(C^{-1}M)$ is not exactly
symmetric (F3); $Q$ becomes $y^\top(C^{-1}y)$ instead of $\lVert z\rVert^2$, both
$\theta$-independent; and the intermediate $C^{-1}M$ has the dynamic range of $C^{-1}$ rather
than that of $L^{-1}$, which is the float32 concern discussed in the reference note §5.2
(mitigated by `strain_scale="auto"`).

**Model signature and stored output.** The PR changes the model arguments from
`(times, strains, ls, fps, fcs)` to `(times, strains, ar_coeffs, sigmas, fps, fcs)` and
adjusts `get_arviz` accordingly. This is plumbing rather than mathematics, but it has
consequences. `Fit.cholesky_factors` and `Fit.run_input` now carry two different
representations of the same $C_i$; they are consistent by construction, since both derive
from the same ACF slice. More importantly, the stored `Result` no longer carries either
representation (§9.1, D3): `get_arviz` drops `cholesky_factor`, does not store `ar_coeffs`,
and declares an `ar_coeffs` dimension with no data, so every whitening-based method of
`Result` fails on the PR's output. `get_arviz` also now infers the number of detectors from
`sampler._args[1]` (the strains) instead of the last argument. Two further details of the
model body: `n_fft` is derived from `ar_coeffs[0]` alone, which assumes a common $N$ across
detectors (as `main` also does, via a common `n_analyze`); and the filter transforms
`fft_as`, `fft_bs` are computed inside the traced model body from constant inputs. XLA does
not constant-fold FFTs of such literals, so these transforms remain in the compiled gradient
and are re-evaluated on every call (on the CPU backend, loop-invariant code motion can hoist
them out of the sampler's while loop; the GPU backend does not). This was measured in the
benchmark kit under `benchmarks/gs/`.

**Non-mathematical changes in the diff, for completeness.** `pyproject.toml` adds a runtime
dependency `pip>=26.0.1` and renames the `[tool.uv] dev-dependencies` table to
`[dependency-groups]`; `fit.py` adds a second `import jax` (the module already imports it);
`model.py` imports `apply_matrix_fft_precomputed` from `ringdown.utils.matrix` without using
it directly (it is called only inside `apply_cinv_gs_fast`). None of these affect the
likelihood.

---

## 12. Symbol dictionary for the PR's code

| Note symbol | Code name | Where |
|---|---|---|
| $\gamma$ (rescaled by $1/s^2$) | `acf_vals` | `Fit.run_input` |
| $a'$ | `a_coeffs` | `Fit.run_input`, from `scipy.linalg.solve_toeplitz((R, R), -r)` |
| $a = (1, a')$ | `full_ar_coeffs`, `ar_coeffs[i]`, `ac` | `Fit.run_input`, `make_model.model` |
| $\sigma^2$, $\sigma$ | `sigma_sq`, `sigmas[i]`, `sigma` | `Fit.run_input`, `make_model.model` |
| $\tilde a$ | `rev_coeffs` | `make_model.model` |
| $n_{\rm fft}$ | `n_fft = next_fast_len(n_time + P - 1)` | `make_model.model`; `ringdown/utils/matrix.py` |
| $\hat a$, $\hat{\tilde a}$ (padded rFFTs) | `fft_as[i]`, `fft_bs[i]` | `make_model.model` |
| $L(x)\psi$ via FFT | `apply_matrix_fft_precomputed` | `ringdown/utils/matrix.py` |
| $L(a)$, $L(a)^\top$, $L(\tilde a)$, $L(\tilde a)^\top$ applied | `apply_A`, `apply_At`, `apply_B`, `apply_Bt` | inside `apply_cinv_gs_fast` |
| $C^{-1}\psi$ by (5.4) | `apply_cinv_gs_fast(vector, fft_a, fft_b, n_fft, sigma)` | `ringdown/utils/matrix.py` |
| $C_i^{-1}y_i$, $C_i^{-1}M_i$, $C_i^{-1}r_i$ | `Cinv_y`, `Cinv_M`, `Cinv_r` | marginalized branch |
| $w^\top A^{(i)} w$ | `woodbury_corr` | marginalized branch |
| $2N\log\sigma$ (stands in for $\log\lvert C_i\rvert$) | `log_det_C` | both branches |

---

## References

* I. Gohberg and A. Semencul, *On the inversion of finite Toeplitz matrices and their
  continuous analogs*, Mat. Issled. 7 (1972) 201–223. The original statement of (4.3).
* T. Kailath, S.-Y. Kung and M. Morf, *Displacement ranks of matrices and linear equations*,
  J. Math. Anal. Appl. 68 (1979) 395–407. The displacement-rank viewpoint of §4.3–4.4.
* G. Heinig and K. Rost, *Algebraic Methods for Toeplitz-like Matrices and Operators*,
  Birkhäuser (1984). Systematic treatment of inversion formulas of Gohberg–Semencul type.
* N. Levinson, *The Wiener RMS error criterion in filter design and prediction*, J. Math.
  Phys. 25 (1947) 261–278; J. Durbin, *The fitting of time-series models*, Rev. Inst. Int.
  Stat. 28 (1960) 233–244. The recursion of §3.4.
* G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., §4.7 (Toeplitz systems),
  including the Levinson–Durbin algorithm and its stability.
* `docs/marginalized_likelihood.md` in this repository, for the marginalization into which
  the quantities of §6 enter and for the notation $A^{-1}$, $\Lambda$, $B$, $\mu$, $b$, $r$.
