functions {
  vector rd(vector t, real f, real gamma, real Ax, real Ay) {
    int n = rows(t);
    return exp(-gamma*t).*(Ax*cos(2*pi()*f*t) + Ay*sin(2*pi()*f*t));
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

  /* Priors on f and tau are flat */
  real f_min;
  real f_max;
  real gamma_min;
  real gamma_max;


  real A_scale;

  int only_prior;
}

transformed data {
  real tstep = times[1][2]-times[1][1];
  real T = times[1][nsamp]-times[1][1];

  real logit_offsets[nmode];

  for (i in 1:nmode) {
    logit_offsets[i] = logit(1.0/(nmode+2.0-i));
  }
}

parameters {
  vector<lower=f_min, upper=f_max>[nmode] f;
  real gamma_raw[nmode];

  vector<lower=-A_scale,upper=A_scale>[nmode] Ax;
  vector<lower=-A_scale,upper=A_scale>[nmode] Ay;
}

transformed parameters {
  vector[nmode] gamma;
  real gamma_logjac;
  vector[nmode] A;
  vector[nmode] phi0;
  vector[nsamp] h_det_mode[nobs,nmode];
  vector[nsamp] h_det[nobs];

  for (i in 1:nmode) {
    phi0[i] = atan2(Ay[i], Ax[i]);
    A[i] = sqrt(Ax[i]*Ax[i] + Ay[i]*Ay[i]);
  }

  {
    real ljs[nmode];
    for (i in 1:nmode) {
      if (i == 1) {
        real r = inv_logit(gamma_raw[i] + logit_offsets[i]);
        gamma[i] = gamma_min + (gamma_max - gamma_min)*r;
        ljs[i] = log(gamma_max - gamma_min) + log(r) + log1p(-r);
      } else {
        real r = inv_logit(gamma_raw[i] + logit_offsets[i]);
        gamma[i] = gamma[i-1] + (gamma_max - gamma[i-1])*r;
        ljs[i] = log(gamma_max - gamma[i-1]) + log(r) + log1p(-r);
      }
    }

    gamma_logjac = sum(ljs);
  }

  for (i in 1:nobs) {
    h_det[i] = rep_vector(0.0, nsamp);
    for (j in 1:nmode) {
      h_det_mode[i, j] = rd(times[i] - t0[i], f[j], gamma[j], Ax[j], Ay[j]);
      h_det[i] = h_det[i] + h_det_mode[i,j];
    }
  }
}

model {
  /* We want a flat prior on A, phi; thus need Jacobian to d(A, phi) / d(Ax, Ay) = 1/r */
  for (i in 1:nmode) {
    target += -log(A[i]);
  }

  /* Flat prior on gamma, need Jacobian d(gamma) / d(gamma_raw). */
  target += gamma_logjac;
  /* Flat prior on tau for each gamma. */
  //target += -2*sum(log(gamma));
  // /* Flat prior on log(gamma) for each gamma. */
  target += -sum(log(gamma));

  if ( only_prior == 0 ) {
      for (i in 1:nobs) {
        strain[i] ~ multi_normal_cholesky(h_det[i], L[i]);
      }
  }
}

generated quantities {
  vector[nmode] tau = 1.0 ./ gamma;
  vector[nmode] Q = pi()* f .* tau;
}
