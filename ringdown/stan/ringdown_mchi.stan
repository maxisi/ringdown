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

  vector rd(vector t, real f, real gamma, real A, real ellip, real theta, real phi, real Fp, real Fc) {
    int n = rows(t);
    vector[n] hc = cos(2*pi()*f*t - phi);
    vector[n] hs = sin(2*pi()*f*t - phi);
    vector[n] p = exp(-gamma*t).*(hc*cos(theta) - ellip*hs*sin(theta));
    vector[n] c = exp(-gamma*t).*(hc*sin(theta) + ellip*hs*cos(theta));
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

  real drift_scale;

  real dt_min;
  real dt_max;

  real df_max;
  real dtau_max;

  /* boolean arrays indicating whether to perturb n-th mode */
  vector[nmode] perturb_f;
  vector[nmode] perturb_tau;

  int flat_A;
  int flat_A_ellip;
  int only_prior;
}

transformed data {
  if (flat_A && flat_A_ellip) reject("at most one of `flat_A` or `flat_A_ellip` can be true");
}

parameters {
  real log_drift_unit[nobs];
  real<lower=M_min, upper=M_max> M;
  real<lower=chi_min, upper=chi_max> chi;

  vector<lower=-1,upper=1>[nmode] ellip;
  unit_vector[2] phi_plus_vec[nmode];
  unit_vector[2] phi_minus_vec[nmode];

  vector<lower=dt_min, upper=dt_max>[nobs-1] dts;

  vector<lower=-df_max,upper=df_max>[nmode] df;
  vector<lower=-dtau_max,upper=dtau_max>[nmode] dtau;
}

transformed parameters {
  real drift[nobs];
  vector[nmode] gamma;
  vector[nmode] f;
  vector[nmode] theta;
  vector[nmode] phi;
  vector[nsamp] h_det_mode_unit[nobs,nmode];

  vector[nmode] mu_amplitudes;
  matrix[nmode,nmode] amplitudes_R;

  real log_det_amplitudes;
  real rss;

  for (i in 1:nobs) {
    drift[i] = exp(log_drift_unit[i]*drift_scale);
  }

  for (i in 1:nmode) {
    real phi_plus = atan2(phi_plus_vec[i][2], phi_plus_vec[i][1]);
    real phi_minus = atan2(phi_minus_vec[i][2], phi_minus_vec[i][1]);

    theta[i] = -(phi_plus + phi_minus)/2;
    phi[i] = (phi_plus - phi_minus)/2;
  }

  {
    real fref = 2985.668287014743;
    real mref = 68.0;

    real f0 = fref*mref/M;

    f = f0*chi_factors(chi, nmode, f_coeffs) .* exp(df .* perturb_f);
    gamma = f0*chi_factors(chi, nmode, g_coeffs) .* exp(-dtau .* perturb_tau);
  }

  if ( only_prior == 0 ) {
    matrix[nobs*nsamp, nmode] Mw;
    matrix[nobs*nsamp, nmode] amplitudes_Q;
    vector[nobs*nsamp] dw;
    vector[nobs*nsamp] rw;

    for (i in 1:nobs) {
      real torigin;

      if (i > 1) {
        torigin = t0[i] + dts[i-1];
      } else {
        torigin = t0[i];
      }

      for (j in 1:nmode) {
        h_det_mode_unit[i, j] = rd(times[i] - torigin, f[j], gamma[j], 1.0, ellip[j], theta[j], phi[j], FpFc[i][1], FpFc[i][2]);

        Mw[(i-1)*nsamp+1:i*nsamp, j] = mdivide_left_tri_low(L[i], h_det_mode_unit[i,j]);
      }
      dw[(i-1)*nsamp+1:i*nsamp] = mdivide_left_tri_low(L[i], strain[i]);
    }

    amplitudes_R = qr_thin_R(Mw);
    amplitudes_Q = qr_thin_Q(Mw); 
    mu_amplitudes = mdivide_right_tri_low(dw' * amplitudes_Q, amplitudes_R')';
    rw = dw - Mw * mu_amplitudes;
    rss = rw' * rw;
    log_det_amplitudes = nmode*log(2*pi()) - 2*sum(log(diagonal(amplitudes_R)));
  }
}

model {
  vector[nsamp] resid[nobs];
  /* drift ~ lognormal(0, drift_scale) */
  log_drift_unit ~ std_normal();

  /* Amplitude prior is flat, ellip prior is flat, angle prior is isotropic. */

  /* Flat prior on M, chi */

  /* Flat prior on the delta-fs, delta-taus. */

  /* Marginalized likelihood */

  /* Likelihood */
  if ( only_prior == 0 ) {
      target += -rss/2 + log_det_amplitudes/2;
  }
}

generated quantities {
  vector[nmode] A = mu_amplitudes + (mdivide_right_tri_low(multi_normal_rng(rep_vector(0, nmode), diag_matrix(rep_vector(1, nmode)))', amplitudes_R'))';
  vector[nsamp] h_det_mode[nobs, nmode];
  vector[nsamp] h_det[nobs];
  vector[nmode] tau = 1.0 ./ gamma;
  vector[nmode] Q = pi() * f .* tau;
  vector[nmode] phiR;
  vector[nmode] phiL;
  for (i in 1:nmode) {
    phiR[i] = atan2(phi_plus_vec[i][2], phi_plus_vec[i][1]);
    phiL[i] = atan2(phi_minus_vec[i][2], phi_minus_vec[i][1]);
  }

  for (i in 1:nobs) {
    h_det[i] = rep_vector(0, nsamp);
    for (j in 1:nmode) {
      h_det_mode[i,j] = A[j]*h_det_mode_unit[i,j];
      h_det[i] = h_det[i] + h_det_mode[i,j];
    }
  }

}
