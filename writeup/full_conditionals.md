# Appendix A — Full Conditional Derivations

Notation. For game $i$ in which team $j[i]$ is the favorite and which is played in
season $s[i]$, let
$$
X_i = \theta_{j[i]} + \beta_1 R_i + \beta_2 H_i + \gamma_{s[i]} + \varepsilon_i,
\qquad \varepsilon_i \stackrel{\text{iid}}{\sim} N(0,\sigma^2).
$$
We use $\mathbf{X}\in\mathbb{R}^N$ for the stacked outcome vector, $\boldsymbol{\theta}\in\mathbb{R}^J$
for team intercepts ($J=30$), and $\boldsymbol{\gamma}\in\mathbb{R}^S$ for season intercepts
($S=10$). Write $n_j = \#\{i: j[i]=j\}$ and $n_s = \#\{i: s[i]=s\}$. All
conditionals below are derived by completing the square in $\log p(X\mid \cdot)\,p(\cdot)$.

---

### A.1 Team intercept $\theta_j$

Define the *partial residual* with the covariate part removed:
$$
r_i \;=\; X_i - \beta_1 R_i - \beta_2 H_i - \gamma_{s[i]}
\;\sim\; N(\theta_{j[i]},\sigma^2).
$$
With prior $\theta_j\sim N(\mu,\tau^2)$, the standard normal–normal update gives
$$
\boxed{\;\theta_j \mid \text{rest}\;\sim\;N(m_j, v_j),\quad
v_j = \!\left(\tfrac{1}{\tau^2} + \tfrac{n_j}{\sigma^2}\right)^{\!-1},\;
m_j = v_j\!\left(\tfrac{\mu}{\tau^2} + \tfrac{1}{\sigma^2}\sum_{i:j[i]=j} r_i\right).\;}
$$
The 30 $\theta_j$ are conditionally independent given the rest.

### A.2 League mean $\mu$

Conditional on $(\theta_1,\ldots,\theta_J)\sim N(\mu,\tau^2)$ and prior
$\mu\sim N(\mu_0,\gamma_0^2)$,
$$
\boxed{\;\mu\mid\text{rest}\;\sim\;N(m_\mu, v_\mu),\quad
v_\mu = \!\left(\tfrac{1}{\gamma_0^2} + \tfrac{J}{\tau^2}\right)^{\!-1},\;
m_\mu = v_\mu\!\left(\tfrac{\mu_0}{\gamma_0^2} + \tfrac{J\,\bar{\theta}}{\tau^2}\right),\;}
$$
where $\bar{\theta} = J^{-1}\sum_j \theta_j$.

### A.3 Between-team variance $\tau^2$

Prior $\tau^2\sim\mathrm{IG}(\eta_0/2,\eta_0\tau_0^2/2)$ is conjugate to the
$\theta_j\mid\mu,\tau^2$ likelihood:
$$
\boxed{\;\tau^2\mid\text{rest}\;\sim\;\mathrm{IG}\!\left(\tfrac{\eta_0+J}{2},\;
\tfrac{\eta_0\tau_0^2 + \sum_{j}(\theta_j-\mu)^2}{2}\right).\;}
$$

### A.4 Within-team variance $\sigma^2$

Let
$\mathrm{SSR} \;=\; \sum_{i=1}^{N}\!\bigl(X_i - \theta_{j[i]} - \beta_1 R_i - \beta_2 H_i - \gamma_{s[i]}\bigr)^{2}.$
With prior $\sigma^2\sim\mathrm{IG}(\nu_0/2,\nu_0\sigma_0^2/2)$,
$$
\boxed{\;\sigma^2\mid\text{rest}\;\sim\;\mathrm{IG}\!\left(\tfrac{\nu_0+N}{2},\;
\tfrac{\nu_0\sigma_0^2 + \mathrm{SSR}}{2}\right).\;}
$$

### A.5 Season intercept $\gamma_s$

Define the partial residual with team and covariate parts removed:
$$
q_i \;=\; X_i - \theta_{j[i]} - \beta_1 R_i - \beta_2 H_i \;\sim\; N(\gamma_{s[i]},\sigma^2).
$$
With prior $\gamma_s\sim N(0,\tau_s^2)$,
$$
\boxed{\;\gamma_s\mid\text{rest}\;\sim\;N(m_s, v_s),\quad
v_s = \!\left(\tfrac{1}{\tau_s^2} + \tfrac{n_s}{\sigma^2}\right)^{\!-1},\;
m_s = v_s\cdot\tfrac{1}{\sigma^2}\!\sum_{i:s[i]=s} q_i.\;}
$$

### A.6 Between-season variance $\tau_s^2$

$$
\boxed{\;\tau_s^2\mid\text{rest}\;\sim\;\mathrm{IG}\!\left(\tfrac{\eta_{s0}+S}{2},\;
\tfrac{\eta_{s0}\tau_{s0}^2 + \sum_s \gamma_s^2}{2}\right).\;}
$$

### A.7 Regression coefficients $(\beta_1,\beta_2)$ jointly

Stack the design $\mathbf{Z}\in\mathbb{R}^{N\times 2}$ with columns $(R_i,H_i)$
and the partial residual $\mathbf{y} = \bigl(X_i - \theta_{j[i]} - \gamma_{s[i]}\bigr)_{i=1}^{N}$,
so $\mathbf{y} = \mathbf{Z}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with
$\boldsymbol{\varepsilon}\sim N(\mathbf{0},\sigma^2 I_N)$. Conjugate prior
$\boldsymbol{\beta}\sim N\!\bigl(\mathbf{0},\sigma_\beta^2 I_2\bigr)$ gives
$$
\boxed{\;\boldsymbol{\beta}\mid\text{rest}\;\sim\;N(\boldsymbol{m}_\beta,\boldsymbol{V}_\beta),
\quad
\boldsymbol{V}_\beta = \!\left(\tfrac{\mathbf{Z}^\top \mathbf{Z}}{\sigma^2} + \tfrac{1}{\sigma_\beta^2} I_2\right)^{\!-1},\;
\boldsymbol{m}_\beta = \boldsymbol{V}_\beta\,\tfrac{\mathbf{Z}^\top\mathbf{y}}{\sigma^2}.\;}
$$
This is the standard Bayesian linear-regression update conditional on all
other parameters; the joint update preserves the (small) posterior correlation
between the two coefficients.

---

### Gibbs sampler skeleton

Initialize $(\boldsymbol{\theta}, \boldsymbol{\gamma}, \mu, \tau^2, \tau_s^2,
\sigma^2, \beta_1, \beta_2)$ at dispersed values. For $t=1,\ldots,T$:

1. Sample $\mu \mid \text{rest}$ — A.2.
2. Sample $\tau^2 \mid \text{rest}$ — A.3.
3. For $j=1,\ldots,J$: sample $\theta_j \mid \text{rest}$ — A.1.
4. Sample $\boldsymbol{\beta} \mid \text{rest}$ jointly — A.7.
5. For $s=1,\ldots,S$: sample $\gamma_s \mid \text{rest}$ — A.5.
6. Sample $\tau_s^2 \mid \text{rest}$ — A.6.
7. Sample $\sigma^2 \mid \text{rest}$ — A.4.

Each block update is conjugate, so no Metropolis steps are required.
