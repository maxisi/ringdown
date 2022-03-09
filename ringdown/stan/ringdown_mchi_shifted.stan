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

  vector rd(vector t, real f, real gamma, real Apx, real Apy, real Acx, real Acy, real Fp, real Fc) {
    int n = rows(t);
    vector[n] ct = cos(2*pi()*f*t);
    vector[n] st = sin(2*pi()*f*t);
    vector[n] p = exp(-gamma*t).*(Apx*ct + Apy*st);
    vector[n] c = exp(-gamma*t).*(Acx*ct + Acy*st);
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
  matrix[nmode,6] df_coeffs;
  matrix[nmode,6] dg_coeffs;

  /* Priors on m and chi are flat */
  real M_min;
  real M_max;

  real drift_scale;

  real dt_min;
  real dt_max;

  real r2_qchi_min;
  real r2_qchi_max;

  real theta_qchi_min;
  real theta_qchi_max;

  real A_scale;

  int flat_A_ellip;
  int only_prior;
}

parameters {
  real log_drift_unit[nobs];
  real<lower=M_min, upper=M_max> M;
  real<lower=r2_qchi_min, upper=r2_qchi_max> r2_qchi;
  real<lower=theta_qchi_min, upper=theta_qchi_max> theta_qchi;

  vector[nmode] Apx_unit;
  vector[nmode] Apy_unit;
  vector[nmode] Acx_unit;
  vector[nmode] Acy_unit;

  vector<lower=dt_min, upper=dt_max>[nobs-1] dts;
}

transformed parameters {
  real q;
  real chi;
  real drift[nobs];
  vector[nmode] gamma;
  vector[nmode] f;
  vector[nsamp] h_det_mode[nobs,nmode];
  vector[nsamp] h_det[nobs];

  vector[nmode] Apx;
  vector[nmode] Apy;
  vector[nmode] Acx;
  vector[nmode] Acy;

  vector[nmode] A;
  vector[nmode] ellip;

  for (i in 1:nobs) {
    drift[i] = exp(log_drift_unit[i]*drift_scale);
  }

  for (i in 1:nmode) {
    Apx[i] = A_scale*Apx_unit[i];
    Apy[i] = A_scale*Apy_unit[i];
    Acx[i] = A_scale*Acx_unit[i];
    Acy[i] = A_scale*Acy_unit[i];

    A[i] = 0.5*(sqrt((Acy[i] + Apx[i])^2 + (Acx[i] - Apy[i])^2) + sqrt((Acy[i] - Apx[i])^2 + (Acx[i] + Apy[i])^2));

    ellip[i] = (sqrt((Acy[i] + Apx[i])^2 + (Acx[i] - Apy[i])^2) -  sqrt((Acy[i] - Apx[i])^2 + (Acx[i] + Apy[i])^2))/( sqrt((Acy[i] + Apx[i])^2 + (Acx[i] - Apy[i])^2) +  sqrt((Acy[i] - Apx[i])^2 + (Acx[i] + Apy[i])^2));
  }

  { 
    q = r2_qchi*(sin(theta_qchi)^2);
    chi = sqrt(r2_qchi)*(cos(theta_qchi));
  }

  {
    real fref = 2985.668287014743;
    real mref = 68.0;

    real f0 = fref*mref/M;

    f = f0*(chi_factors(chi, nmode, f_coeffs) + q*chi_factors(chi, nmode, df_coeffs));
    gamma = f0*(chi_factors(chi, nmode, g_coeffs) + q*chi_factors(chi, nmode, dg_coeffs));
  }

  if ((flat_A_ellip) && (only_prior)) {
      for (i in 1:nmode) {
          if (A[i] > 2*A_scale) reject("A", i-1, " > 2*A_scale");
      }
  }

  if ( only_prior == 0 ) {
    for (i in 1:nobs) {
      real torigin;
      h_det[i] = rep_vector(0.0, nsamp);

      if (i > 1) {
        torigin = t0[i] + dts[i-1];
      } else {
        torigin = t0[i];
      }

    for (j in 1:nmode) {
      h_det_mode[i, j] = rd(times[i] - torigin, f[j], gamma[j], Apx[j], Apy[j], Acx[j], Acy[j], FpFc[i][1], FpFc[i][2]);
      h_det[i] = h_det[i] + h_det_mode[i,j];
    }
  }
  }
}

model {
  /* drift ~ lognormal(0, drift_scale) */
  log_drift_unit ~ std_normal();

  /* Amplitude prior */
  if (flat_A_ellip) {
      for (i in 1:nmode) {
        target += -3*log(A[i]) - log1m(ellip[i]^2);
      }
  } else {
    Apx_unit ~ std_normal();
    Apy_unit ~ std_normal();
    Acx_unit ~ std_normal();
    Acy_unit ~ std_normal();
  }

  /* Flat prior on M, chi */

  /* Flat prior on the delta-fs, delta-taus. */

  /* Likelihood */
  if ( only_prior == 0 ) {
      for (i in 1:nobs) {
        strain[i] ~ multi_normal_cholesky(h_det[i], drift[i]*L[i]);
      }
  }
}

generated quantities {
  vector[nmode] tau = 1.0 ./ gamma;
  vector[nmode] Q = pi() * f .* tau;
  vector[nmode] phiR;
  vector[nmode] phiL;
  vector[nmode] theta;
  vector[nmode] phi;
  real Q_charge = sqrt(q)
  
  for (i in 1:nmode) {
    phiR[i] = atan2(-Acx[i] + Apy[i], Acy[i] + Apx[i]);
    phiL[i] = atan2(-Acx[i] - Apy[i], -Acy[i] + Apx[i]);
  }
  theta = -0.5*(phiR + phiL);
  phi = 0.5*(phiR - phiL);
}
