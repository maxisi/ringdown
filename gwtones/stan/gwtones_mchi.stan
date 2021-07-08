functions {
  real get_snr(vector h, vector d, matrix L, int nsamp) {
    vector[nsamp] wh = mdivide_left_tri_low(L, h);
    return dot_product(wh, mdivide_left_tri_low(L, d)) / sqrt(dot_self(wh));
  }

  vector f_factors(real chi, int nmode, matrix f_coeffs) {
    /* return f's for first `nmode` modes */
    real log1mc = log1p(-chi);

    vector[nmode] f;
    for (i in 1:nmode) {
      row_vector[6] c = row(f_coeffs, i);
      f[i] = c[1]*log1mc + c[2] + chi*(c[3] + chi*(c[4] + chi*(c[5] + chi*c[6])));
    }
    return f;
  }

  vector g_factors(real chi, int nmode, matrix g_coeffs) {
    /* return g's for first `nmode` modes */
    real log1mc = log1p(-chi);

    vector[nmode] g;
    for (i in 1:nmode) {
      row_vector[6] c = row(g_coeffs, i);
      g[i] = c[1]*log1mc + c[2] + chi*(c[3] + chi*(c[4] + chi*(c[5] + chi*c[6])));
    }
    return g;
  }

  vector rd(vector t, real f, real gamma, real Ap, real Ac, real phip, real phic, real Fp, real Fc) {
    int n = rows(t);
    vector[n] p = Ap*exp(-gamma*t).*cos(2*pi()*f*t - phip);
    vector[n] c = Ac*exp(-gamma*t).*cos(2*pi()*f*t - phic);
    return Fp*p + Fc*c;
  }
}

data {
  int nobs;
  int nsamp;
  int nmode;

  real t0[nobs];
  vector[nsamp] times[nobs];
  vector[nsamp] strain[nobs];
  matrix[nsamp,nsamp] L[nobs];

  vector[2] FpFc[nobs];

  matrix[nmode,6] f_coeffs;
  matrix[nmode,6] g_coeffs;

  /* Priors on m and chi are flat */
  real M_min;
  real M_max;
  real chi_min;
  real chi_max;

  real A_max;

  real dt_min;
  real dt_max;

  real df_max;
  real dtau_max;

  /* boolean arrays indicating whether to perturb n-th mode */
  vector[nmode] perturb_f;
  vector[nmode] perturb_tau;

  int only_prior;
}

parameters {
  real<lower=M_min, upper=M_max> M;
  real<lower=chi_min, upper=chi_max> chi;

  vector[nmode] Ap_x;
  vector[nmode] Ap_y;
  vector[nmode] Ac_x;
  vector[nmode] Ac_y;

  vector<lower=dt_min, upper=dt_max>[nobs-1] dts;

  vector<lower=-df_max,upper=df_max>[nmode] df;
  vector<lower=-dtau_max,upper=dtau_max>[nmode] dtau;
}

transformed parameters {
  vector[nmode] gamma;
  vector[nmode] f;
  vector[nsamp] h_det_mode[nobs,nmode];
  vector[nsamp] h_det[nobs];

  vector[nmode] A;
  vector[nmode] ellip;

  vector[nmode] Ap;
  vector[nmode] Ac;
  vector[nmode] phip;
  vector[nmode] phic;

  for (i in 1:nmode) {
    Ap[i] = A_max*sqrt(Ap_x[i]^2 + Ap_y[i]^2);
    Ac[i] = A_max*sqrt(Ac_x[i]^2 + Ac_y[i]^2);
    phip[i] = atan2(Ap_y[i], Ap_x[i]);
    phic[i] = atan2(Ac_y[i], Ac_x[i]);

    A[i] = 0.5*A_max*(sqrt((Ac_y[i] + Ap_x[i])^2 + (Ac_x[i] - Ap_y[i])^2) + sqrt((Ac_y[i] - Ap_x[i])^2 + (Ac_x[i] + Ap_y[i])^2));
    ellip[i] = (sqrt((Ac_y[i] + Ap_x[i])^2 + (Ac_x[i] - Ap_y[i])^2) -  sqrt((Ac_y[i] - Ap_x[i])^2 + (Ac_x[i] + Ap_y[i])^2))/( sqrt((Ac_y[i] + Ap_x[i])^2 + (Ac_x[i] - Ap_y[i])^2) +  sqrt((Ac_y[i] - Ap_x[i])^2 + (Ac_x[i] + Ap_y[i])^2));

    if (only_prior) {
      // impose constraint on total amplitude; otherwise we rely on the likelihood to cut it off.
      if (A[i] > 10*A_max) reject("A", i, " > A_max");
    }
  }

  {
    real fref = 2985.668287014743;
    real mref = 68.0;

    real f0 = fref*mref/M;

    f = f0*f_factors(chi, nmode, f_coeffs) .* (1 + df .* perturb_f);
    gamma = f0*g_factors(chi, nmode, g_coeffs) ./ (1 + dtau .* perturb_tau);
  }

  for (i in 1:nmode-1) {
      if (gamma[i+1] <= gamma[i]) reject("gamma[", i, "] > gamma[", i+1, "]");
  }

  for (i in 1:nobs) {
    real torigin;
    h_det[i] = rep_vector(0.0, nsamp);

    if (i > 1) {
      torigin = t0[i] + dts[i-1];
    } else {
      torigin = t0[i];
    }

    for (j in 1:nmode) {
      h_det_mode[i, j] = rd(times[i] - torigin, f[j], gamma[j], Ap[j], Ac[j], phip[j], phic[j], FpFc[i][1], FpFc[i][2]);
      h_det[i] = h_det[i] + h_det_mode[i,j];
    }
  }
}

model {
  // Ap_x ~ std_normal();
  // Ap_y ~ std_normal();
  // Ac_x ~ std_normal();
  // Ac_y ~ std_normal();

  for (i in 1:nmode) {
    //target += -log(A[i]); // + 0.5*Ap[i]^2 / (0.25*A_max^2);
    // target += 0.5*Ap_x[i]^2;
    // target += 0.5*Ap_y[i]^2;
    // target += 0.5*Ac_x[i]^2;
    // target += 0.5*Ac_y[i]^2;
    target += -3*log(A[i]); // - log1m(ellip[i]^2);
    /* go to flat in cosi */
    // target += -2*log(abs(ellip[i])) + log(-1 + 1/sqrt(1-ellip[i]^2));
    // target += log((-1 + 1/sqrt(1-ellip[i]^2))/ellip[i]^2);
  }
  /* Flat prior on M, chi */

  /* Flat prior on the delta-fs. */

  /* Likelihood */
  if ( only_prior == 0 ) {
      for (i in 1:nobs) {
        strain[i] ~ multi_normal_cholesky(h_det[i], L[i]);
      }
  }
}

generated quantities {
  vector[nmode] tau = 1.0 ./ gamma;
  vector[nmode] Q = pi()* f .* tau;
  vector[nmode] phiR;
  vector[nmode] phiL;
  vector[nmode] phi;
  vector[nmode] theta;
  //real net_snr;
  //vector[nobs] snr;

  for (i in 1:nmode) {
    phiR[i] = atan2(-Ac_x[i] + Ap_y[i], Ac_y[i] + Ap_x[i]);
    phiL[i] = atan2(-Ac_x[i] - Ap_y[i], -Ac_y[i] + Ap_x[i]);
    theta[i] = -0.5*(phiR[i] + phiL[i]);
    phi[i] = 0.5*(phiR[i] - phiL[i]);

  }

  // for (i in 1:nobs) {
  //   snr[i] = get_snr(h_det[i], strain[i], L[i], nsamp);
  // }
  // net_snr = sqrt(dot_self(snr));
}
