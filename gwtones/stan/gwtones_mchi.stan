functions {
  real get_snr(vector h, vector d, matrix L, int nsamp) {
    vector[nsamp] wh = mdivide_left_tri_low(L, h);
    return dot_product(wh, mdivide_left_tri_low(L, d)) / sqrt(dot_self(wh));
  }

  vector f_factors(real chi, int nmode) {
    /* return f's for first `nmode` modes */
    real log1mc = log1p(-chi);

    matrix[8,6] coeffs = [
    [-0.00823557, 0.05994978, -0.00106621,  0.08354181, -0.15165638,  0.11021346],
    [-0.00817443, 0.05566216,  0.00174642,  0.08531863, -0.15465231,  0.11326354],
    [-0.00803510, 0.04842476,  0.00545740,  0.09300492, -0.16959796,  0.12487077],
    [-0.00779142, 0.04067346,  0.00491459,  0.12084976, -0.22269851,  0.15991054],
    [-0.00770094, 0.03418526, -0.00823914,  0.20643478, -0.37685018,  0.24917989],
    [ 0.00303002, 0.02558406,  0.06756237, -0.15655673,  0.36731757, -0.20880323],
    [-0.00948223, 0.02209137, -0.00671374,  0.22389539, -0.36335472,  0.21967326],
    [-0.00931548, 0.01429318,  0.03356735,  0.11195758, -0.20533169,  0.14109002]
    ];

    vector[nmode] f;
    for (i in 1:nmode) {
      row_vector[6] c = row(coeffs, i);
      f[i] = c[1]*log1mc + c[2] + chi*(c[3] + chi*(c[4] + chi*(c[5] + chi*c[6])));
    }
    return f;
  }

  vector g_factors(real chi, int nmode) {
    /* return g's for first `nmode` modes */
    real log1mc = log1p(-chi);

    matrix[8,6] coeffs = [
    [ 0.01180702,  0.08838127,  0.02528302, -0.09002286,  0.18245511, -0.12162592],
    [ 0.03360470,  0.27188580,  0.07460669, -0.31374292,  0.62499252, -0.4116911 ],
    [ 0.05754774,  0.47487353,  0.10275994, -0.52484007,  1.03658076, -0.67299196],
    [ 0.08300547,  0.70003289,  0.11521228, -0.77083409,  1.48332672, -0.93350403],
    [ 0.11438483,  0.94036953,  0.10326999, -0.89912932,  1.62233654, -0.96391884],
    [-0.01888617,  1.20407042, -0.49651606,  1.04793870, -2.02319930,  0.88102107],
    [ 0.10530775,  1.43868390, -0.05621762, -1.38317450,  3.05769954, -2.25940348],
    [ 0.14280084,  1.69019137, -0.25210715, -0.67029321,  2.09513036, -1.8255968 ]
    ];

    vector[nmode] g;
    for (i in 1:nmode) {
      row_vector[6] c = row(coeffs, i);
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

  /* Priors on m and chi are flat */
  real MMin;
  real MMax;

  vector[2] FpFc[nobs];

  real Amax;

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
  real<lower=MMin, upper=MMax> M;
  real<lower=0, upper=0.99> chi;

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
    Ap[i] = Amax*sqrt(Ap_x[i]^2 + Ap_y[i]^2);
    Ac[i] = Amax*sqrt(Ac_x[i]^2 + Ac_y[i]^2);
    phip[i] = atan2(Ap_y[i], Ap_x[i]);
    phic[i] = atan2(Ac_y[i], Ac_x[i]);

    A[i] = 0.5*Amax*(sqrt((Ac_y[i] + Ap_x[i])^2 + (Ac_x[i] - Ap_y[i])^2) + sqrt((Ac_y[i] - Ap_x[i])^2 + (Ac_x[i] + Ap_y[i])^2));
    ellip[i] = (sqrt((Ac_y[i] + Ap_x[i])^2 + (Ac_x[i] - Ap_y[i])^2) -  sqrt((Ac_y[i] - Ap_x[i])^2 + (Ac_x[i] + Ap_y[i])^2))/( sqrt((Ac_y[i] + Ap_x[i])^2 + (Ac_x[i] - Ap_y[i])^2) +  sqrt((Ac_y[i] - Ap_x[i])^2 + (Ac_x[i] + Ap_y[i])^2));

    # impose constraint on total amplitude
    if (A[i] > 10*Amax) reject("A", i, " > Amax");
  }

  {
    real fref = 2985.668287014743;
    real mref = 68.0;

    real f0 = fref*mref/M;

    f = f0*f_factors(chi, nmode) .* (1 + df .* perturb_f);
    gamma = f0*g_factors(chi, nmode) ./ (1 + dtau .* perturb_tau);
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
  Ap_x ~ std_normal();
  Ap_y ~ std_normal();
  Ac_x ~ std_normal();
  Ac_y ~ std_normal();

  for (i in 1:nmode) {
    //target += -log(A[i]); // + 0.5*Ap[i]^2 / (0.25*Amax^2);
    target += 0.5*Ap_x[i]^2;
    target += 0.5*Ap_y[i]^2;
    target += 0.5*Ac_x[i]^2;
    target += 0.5*Ac_y[i]^2;
    target += -3*log(A[i]) - log1m(ellip[i]^2);
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
