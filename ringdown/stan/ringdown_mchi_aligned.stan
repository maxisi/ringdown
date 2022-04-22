functions {
  real get_snr(vector h, vector d, matrix L, int nsamp) {
    vector[nsamp] wh = mdivide_left_tri_low(L, h);
    return dot_product(wh, mdivide_left_tri_low(L, d)) / sqrt(dot_self(wh));
  }

  vector chi_factors(real chi, int nmode, matrix coeffs) {
    /* return f's for first `nmode` modes */
    real log1mc = log1p(-chi);

    vector[nmode] f;
    for (i in 1:nmode) {
      row_vector[6] c = row(coeffs, i);
      f[i] = c[1]*chi + c[2] + log1mc*(c[3] + log1mc*(c[4] + log1mc*(c[5] + log1mc*c[6])));
    }
    return f;
  }

  vector rd(vector t, real f, real gamma, real A, real cosi, real phi, real Fp, real Fc) {
    int n = rows(t);
    vector[n] phase = 2*pi()*f*t - phi;

    /* could easily add polarization angle here */
    vector[n] p = (1 + cosi^2)*A*exp(-gamma*t).*cos(phase);
    vector[n] c = 2*cosi*A*exp(-gamma*t).*sin(phase);

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

  real A_scale;
  
  real cosi_min;
  real cosi_max;

  real dt_min;
  real dt_max;

  real df_max;
  real dtau_max;

  /* boolean arrays indicating whether to perturb n-th mode */
  vector[nmode] perturb_f;
  vector[nmode] perturb_tau;

  int flat_A;
  int only_prior;
}

parameters {
  real<lower=M_min, upper=M_max> M;
  real<lower=chi_min, upper=chi_max> chi;
  real<lower=cosi_min, upper=cosi_max> cosi;
  unit_vector[2] iota_unit;

  vector[nmode] Ax_unit;
  vector[nmode] Ay_unit;

  vector<lower=dt_min, upper=dt_max>[nobs-1] dts;

  vector<lower=-df_max,upper=df_max>[nmode] df;
  vector<lower=-dtau_max,upper=dtau_max>[nmode] dtau;
}

transformed parameters {
  vector[nmode] gamma;
  vector[nmode] f;
  vector[nsamp] h_det_mode[nobs,nmode];
  vector[nsamp] h_det[nobs];
  // real cosi = cos(atan2(iota_unit[2], iota_unit[1]));

  vector[nmode] A;
  real phi[nmode];

  for (i in 1:nmode) {
    A[i] = A_scale*sqrt(Ax_unit[i]^2 + Ay_unit[i]^2);
    phi[i] = atan2(Ay_unit[i], Ax_unit[i]);
  }

  {
    real fref = 2985.668287014743;
    real mref = 68.0;

    real f0 = fref*mref/M;

    f = f0*chi_factors(chi, nmode, f_coeffs) .* exp(df .* perturb_f);
    gamma = f0*chi_factors(chi, nmode, g_coeffs) .* exp(-dtau .* perturb_tau);
  }

  if ((flat_A) && (only_prior)) {
      for (i in 1:nmode) {
          if (A[i] > A_scale) reject("A", i, " > A_scale");
      }
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
      h_det_mode[i, j] = rd(times[i] - torigin, f[j], gamma[j], A[j], cosi, phi[j], FpFc[i][1], FpFc[i][2]);
      h_det[i] = h_det[i] + h_det_mode[i,j];
    }
  }
}

model {
  /* Amplitude prior */
  if (flat_A) {
      for (i in 1:nmode) {
        target += -log(A[i]);
      }
  } else {
    Ax_unit ~ std_normal();
    Ay_unit ~ std_normal();
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
  vector[nmode] Q = pi() * f .* tau;
  vector[nmode] Ap = (1 + cosi^2)*A;
  vector[nmode] Ac = 2*cosi*A;
  real ellip = 2*cosi/(1 + cosi^2);
}
