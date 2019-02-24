#include "ffnn/train/NNTrainerGSL.hpp"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

#include <cstddef> // NULL

namespace nn_trainer_gsl_details {

    // --- Workspace struct methods

    void training_workspace::copyData(const NNTrainingData &tdata)
    {
        ndata = tdata.ndata;
        ntraining = tdata.ntraining;
        nvalidation = tdata.nvalidation;
        xndim = tdata.xndim;
        yndim = tdata.yndim;
        x = tdata.x;
        y = tdata.y;
        yd1 = tdata.yd1;
        yd2 = tdata.yd2;
        w = tdata.w;
    }

    void training_workspace::copyConfig(const NNTrainingConfig &tconfig)
    {
        lambda_r = tconfig.lambda_r;
        lambda_d1 = tconfig.lambda_d1;
        lambda_d2 = tconfig.lambda_d2;
        maxn_steps = tconfig.maxn_steps;
        maxn_novali = tconfig.maxn_novali;

        flag_r = (lambda_r > 0);
        flag_d1 = (lambda_d1 > 0);
        flag_d2 = (lambda_d2 > 0);
    }

    void training_workspace::copyDatConf(const NNTrainingData &tdata, const NNTrainingConfig &tconfig)
    {
        copyData(tdata);
        copyConfig(tconfig);
    }

    // --- Helper functions

    // set new NN betas
    int setVP(FeedForwardNeuralNetwork * const ffnn, const gsl_vector * const betas)
    {
        const int npar = ffnn->getNVariationalParameters();
        for (int i=0; i<npar; ++i){
            ffnn->setVariationalParameter(i, gsl_vector_get(betas, i));
        }
        return npar;
    };

    // counts total residual vector size
    // set npar or xndim to >0 to count regularization and derivative residual terms, respectively
    // set nderiv = 1 if only one of both deriv residuals should be counted
    int calcNData(const int &nbase, const int &yndim, const int &npar, const int &xndim, const int &nderiv)
    {
        return (nbase > 0) ? nbase*yndim + npar + nderiv * nbase*xndim*yndim : 0;
    };

    // store (root) square sum of residual vector f in chisq (chi)
    void calcRSS(const gsl_vector * const f, double &chi, double &chisq)
    {
        gsl_blas_ddot(f, f, &chisq);
        chi = sqrt(chisq);
    };

    // calculate all costs from the two residual vectors
    void calcCosts(const gsl_vector * const f, double &chi, double &chisq, const gsl_vector * const fvali, double &chi_vali, double &chisq_vali)
    {
        calcRSS(f, chi, chisq);
        if (fvali) calcRSS(fvali, chi_vali, chisq_vali);
    };

    // calculate all costs (from workspace and vali vector)
    void calcCosts(gsl_multifit_nlinear_workspace * const w, double &chi, double &chisq, const gsl_vector * const fvali, double &chi_vali, double &chisq_vali)
    {
        calcCosts(gsl_multifit_nlinear_residual(w), chi, chisq, fvali, chi_vali, chisq_vali);
    };

    // calculate fit and error arrays
    void calcFitErr(gsl_multifit_nlinear_workspace * const w, double * const fit, double * const err, const int &ndata, const int &npar, const double &chisq)
    {
        const double c = GSL_MAX_DBL(1, sqrt(chisq / (ndata-npar)));
        const gsl_matrix * const J = gsl_multifit_nlinear_jac(w);
        gsl_matrix * const covar = gsl_matrix_alloc (npar, npar);

        gsl_multifit_nlinear_covar(J, 0.0, covar);
        for (int i = 0; i<npar; ++i) {
            fit[i] = gsl_vector_get(w->x, i);
            err[i] = c*sqrt(gsl_matrix_get(covar,i,i));
        }
        gsl_matrix_free(covar);
    };


    // --- Cost functions

    int ffnn_f(const gsl_vector * betas, training_workspace * const tws, gsl_vector * f, const bool flag_r, const bool flag_d)
    {
        const int n = tws->ntraining + tws->nvalidation;
        const bool flag_vali = tws->nvalidation>0;
        FeedForwardNeuralNetwork * ffnn = tws->ffnn;

        const int npar = ffnn->getNVariationalParameters();
        const double lambda_d1_red = tws->lambda_d1/sqrt(tws->xndim), lambda_d2_red = tws->lambda_d2/sqrt(tws->xndim), lambda_r_red = tws->lambda_r / sqrt(npar);

        gsl_vector * fnow = f;
        double scale_now = 1./sqrt(tws->ntraining);

        setVP(ffnn, betas);
        if (flag_vali) gsl_vector_set_zero(tws->fvali); // set fvali to zero

        int idx = 0;
        for (int i=0; i<n; ++i) {
            ffnn->setInput(tws->x[i]);
            ffnn->FFPropagate();

            if (i==tws->ntraining) { // now setup the validation calculation
                idx = 0;
                fnow = tws->fvali;
                scale_now = 1./sqrt(tws->nvalidation);
            }

            for (int j=0; j<tws->yndim; ++j) {
                gsl_vector_set(fnow, idx, scale_now * tws->w[i][j] * (ffnn->getOutput(j) - tws->y[i][j])); // the "pure" residual
                ++idx;

                if (flag_d) { // derivative residual
                    for (int k=0; k<tws->xndim; ++k) {
                        gsl_vector_set(fnow, idx, tws->flag_d1 ? scale_now * tws->w[i][j] * lambda_d1_red * (ffnn->getFirstDerivative(j, k) - tws->yd1[i][j][k]) : 0.0);
                        ++idx;
                        gsl_vector_set(fnow, idx, tws->flag_d2 ? scale_now * tws->w[i][j] * lambda_d2_red * (ffnn->getSecondDerivative(j, k) - tws->yd2[i][j][k]) : 0.0);
                        ++idx;
                    }
                }
            }
        }

        if (flag_r) { //append regularization residual
            int offr_train = calcNData(tws->ntraining, tws->yndim, 0, flag_d ? tws->xndim : 0, 2);
            int offr_vali = calcNData(tws->nvalidation, tws->yndim, 0, flag_d ? tws->xndim : 0, 2);
            for (int ib=0; ib<npar; ++ib) {
                gsl_vector_set(f, offr_train + ib, lambda_r_red * gsl_vector_get(betas, ib));
                if (flag_vali) gsl_vector_set(tws->fvali, offr_vali + ib, lambda_r_red * gsl_vector_get(betas, ib));
            }
        }

        return GSL_SUCCESS;
    }

    int ffnn_df(const gsl_vector * betas, training_workspace * const tws, gsl_matrix * J, const bool flag_r, const bool flag_d)
    {
        FeedForwardNeuralNetwork * ffnn = tws->ffnn_vderiv;

        const int npar = ffnn->getNVariationalParameters();
        const double scale = 1./sqrt(tws->ntraining), lambda_d1_red = scale*tws->lambda_d1/sqrt(tws->xndim), lambda_d2_red = scale*tws->lambda_d2/sqrt(tws->xndim), lambda_r_red = tws->lambda_r / sqrt(npar);

        setVP(ffnn, betas);

        int idx = 0;
        for (int i=0; i<tws->ntraining; ++i) {
            ffnn->setInput(tws->x[i]);
            ffnn->FFPropagate();

            for (int j=0; j<tws->yndim; ++j) {
                for (int ib=0; ib<npar; ++ib) gsl_matrix_set(J, idx, ib, scale * tws->w[i][j] * ffnn->getVariationalFirstDerivative(j, ib)); // the "pure" gradient
                ++idx;

                if (flag_d) { // derivative residual gradient
                    for (int k=0; k<tws->xndim; ++k) {
                        for (int ib=0; ib<npar; ++ib) gsl_matrix_set(J, idx, ib, tws->flag_d1? tws->w[i][j] * lambda_d1_red * ffnn->getCrossFirstDerivative(j, k, ib) : 0.0);
                        ++idx;
                        for (int ib=0; ib<npar; ++ib) gsl_matrix_set(J, idx, ib, tws->flag_d2? tws->w[i][j] * lambda_d2_red * ffnn->getCrossSecondDerivative(j, k, ib) : 0.0);
                        ++idx;
                    }
                }
            }
        }

        if (flag_r) {//append regularization gradient
            int offr = calcNData(tws->ntraining, tws->yndim, 0, flag_d ? tws->xndim : 0, 2);
            for (int ib=0; ib<npar; ++ib) {
                for (int ib2=0; ib2<npar; ++ib2) {
                    gsl_matrix_set(J, offr+ib, ib2, 0.0);
                }
                gsl_matrix_set(J, offr+ib, ib, lambda_r_red);
            }
        }

        return GSL_SUCCESS;
    }

    // cost function without regularization and derivative terms
    int ffnn_f_pure(const gsl_vector * betas, void * const tws, gsl_vector * f) {
        return ffnn_f(betas, static_cast<training_workspace *>(tws), f, false, false);
    }

    // gradient of cost function without regularization and derivative terms
    int ffnn_df_pure(const gsl_vector * betas, void * const tws, gsl_matrix * J) {
        return ffnn_df(betas, static_cast<training_workspace *>(tws), J, false, false);
    }

    // cost function with derivative but without regularization
    int ffnn_f_deriv(const gsl_vector * betas, void * const tws, gsl_vector * f) {
        return ffnn_f(betas, static_cast<training_workspace *>(tws), f, false, true);
    }

    // gradient of cost function with derivative but without regularization
    int ffnn_df_deriv(const gsl_vector * betas, void * const tws, gsl_matrix * J) {
        return ffnn_df(betas, static_cast<training_workspace *>(tws), J, false, true);
    };

    // cost function for fitting, without derivative but with regularization
    int ffnn_f_pure_reg(const gsl_vector * betas, void * const tws, gsl_vector * f) {
        return ffnn_f(betas, static_cast<training_workspace *>(tws), f, true, false);
    };

    // gradient of cost function without derivatives but with regularization
    int ffnn_df_pure_reg(const gsl_vector * betas, void * const tws, gsl_matrix * J) {
        return ffnn_df(betas, static_cast<training_workspace *>(tws), J, true, false);
    };

    // cost function for fitting, with derivative and regularization
    int ffnn_f_deriv_reg(const gsl_vector * betas, void * const tws, gsl_vector * f) {
        return ffnn_f(betas, static_cast<training_workspace *>(tws), f, true, true);
    };

    // gradient of cost function with derivatives and regularization
    int ffnn_df_deriv_reg(const gsl_vector * betas, void * const tws, gsl_matrix * J) {
        return ffnn_df(betas, static_cast<training_workspace *>(tws), J, true, true);
    };


    // --- Custom driver routines

    // if verbose, this is used to print info on every fit iteration
    void printStepInfo(const gsl_multifit_nlinear_workspace * const w, const training_workspace * const tws, const int &status) {
        gsl_vector *f = gsl_multifit_nlinear_residual(w);
        gsl_vector *x = gsl_multifit_nlinear_position(w);
        double rcond = 0.0;

        // compute reciprocal condition number of J(x)
        gsl_multifit_nlinear_rcond(&rcond, w);

        // print
        fprintf(stderr, "status = %s\n", gsl_strerror(status));
        fprintf(stderr, "iter %zu: cond(J) = %8.4f, |f(x)| = %.8f (train), %.8f (vali)\n", gsl_multifit_nlinear_niter(w), 1.0 / rcond, gsl_blas_dnrm2(f), (tws->fvali) ? gsl_blas_dnrm2(tws->fvali) : 0.);
        for (size_t i=0; i<x->size; ++i) fprintf(stderr, "b%zu: %f, ", i,  gsl_vector_get(x, i));
        fprintf(stderr, "\n");
    };

    // solve the system with a maximum of tws->max_nsteps iterations, stopping early when validation error doesn't decrease for too long
    void earlyStopDriver(gsl_multifit_nlinear_workspace * const w, const training_workspace * const tws, const int &verbose, int &status, int &info)
    {
        double bestvali = -1.;
        int count_novali = 0;

        const int n = calcNData(tws->nvalidation, tws->yndim, 0, (tws->flag_d1 || tws->flag_d2) ? tws->xndim : 0, 2);
        gsl_vector * fvali_noreg =  gsl_vector_alloc(n);

        while (true) {
            status = gsl_multifit_nlinear_iterate(w); // iterate workspace
            if (verbose > 1) printStepInfo(w, tws, status);

            if (((int)gsl_multifit_nlinear_niter(w)) >= tws->maxn_steps) {  // check if we reached maxnsteps
                info = 0;
                break;
            }

            if (tws->nvalidation > 0) {
                // then check if validation residual went down (we ignore the regularization part for that)
                for (int i=0; i<n; ++i) gsl_vector_set(fvali_noreg, i, gsl_vector_get(tws->fvali, i));
                double resih = gsl_blas_dnrm2(fvali_noreg);

                if (resih == 0 || std::isnan(resih)) { // if it is 0 or nan, stop
                    info = 0;
                    if (verbose>1) fprintf(stderr, "Unregularized validation residual reached 0 (or NaN). Stopping early.\n\n");
                    break;
                }

                if (bestvali >= 0. && resih >= bestvali) { // count how long it didn't go down
                    if (verbose>1) fprintf(stderr, "Unregularized validation residual %.4f did not decrease from previous minimum %.4f. No new minimum since %i iteration(s).\n\n", resih, bestvali, count_novali);

                    ++count_novali;
                    if (count_novali < tws->maxn_novali) continue;
                    else { // if too long, break
                        info = 1;
                        if (verbose>1) fprintf(stderr, "Reached maximal number of iterations (%i) without new validation minimum. Stopping early.\n\n", count_novali);
                        break;
                    }
                }
                else { // new validation minimum found
                    count_novali = 0;
                    bestvali = resih;
                }
            }

            if (verbose>1) fprintf(stderr, "\n");
        }

        gsl_vector_free(fvali_noreg);
    };
};

// --- Class method implementation

void NNTrainerGSL::findFit(FeedForwardNeuralNetwork * const ffnn, double * const fit, double * const err, const int &verbose) {
    //   Fit NN ffnn with the following passed variables:
    //   fit: holds the to be fitted variables, (i.e. betas)
    //   err: holds the corresponding fit error
    //   verbose: print verbose output while fitting
    //
    //   Everything else is already configured via the
    //   _tdata, _tconfig and optional _gsl_params
    //   structs passed at creation to constructor

    using namespace nn_trainer_gsl_details; // to use local workspace / functions above

    int npar = ffnn->getNVariationalParameters(), ntrain = _tdata.ntraining, nvali = _tdata.nvalidation;
    const gsl_multifit_nlinear_type *T_full = gsl_multifit_nlinear_trust, *T_noreg = gsl_multifit_nlinear_trust, *T_pure = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_fdf fdf_full, fdf_noreg, fdf_pure;
    gsl_multifit_nlinear_workspace * w_full, * w_noreg, * w_pure;
    gsl_vector_view gx = gsl_vector_view_array (fit, npar);

    const int dof = ntrain - npar;
    const bool flag_d = _flag_d1 || _flag_d2;
    int ntrain_pure, ntrain_noreg, nvali_noreg, ntrain_full, nvali_full;
    int status, info;
    double resih, chisq, chi0, chi0_vali = 0.;
    double resi_full, resi_noreg, resi_pure;
    double resi_vali_full = 0., resi_vali_noreg = 0., resi_vali_pure = 0.;

    // make sure the ffnn is configured
    _configureFFNN(ffnn, false); // without vderiv substrates!

    // set fit to initial betas
    for (int i=0; i<npar; ++i) fit[i] = ffnn->getVariationalParameter(i);

    // configure training workspace
    training_workspace tws;
    tws.copyDatConf(_tdata, _tconfig);
    tws.ffnn = ffnn; // set the to-be-fitted FFNN
    tws.ffnn_vderiv = _createVDerivFFNN(ffnn); // set the copy with vderivs

    // configure all three fdf objects

    ntrain_pure = calcNData(ntrain, tws.yndim);

    // first the pure fdf
    fdf_pure.f = ffnn_f_pure;
    fdf_pure.df = ffnn_df_pure;
    fdf_pure.fvv = NULL;
    fdf_pure.n = ntrain_pure;
    fdf_pure.p = npar;
    fdf_pure.params = &tws;

    if (flag_d) {
        ntrain_noreg = calcNData(ntrain, tws.yndim, 0, tws.xndim);
        nvali_noreg = calcNData(nvali, tws.yndim, 0, tws.xndim);

        // deriv fdf without regularization
        fdf_noreg.f = ffnn_f_deriv;
        fdf_noreg.df = ffnn_df_deriv;
        fdf_noreg.fvv = NULL;
        fdf_noreg.n = ntrain_noreg;
        fdf_noreg.p = npar;
        fdf_noreg.params = &tws;
    }
    else {
        ntrain_noreg = calcNData(ntrain, tws.yndim);
        nvali_noreg = calcNData(nvali, tws.yndim);
        fdf_noreg = fdf_pure;
    };

    if (tws.flag_r) {
        ntrain_full = ntrain_noreg + npar;
        nvali_full = nvali_noreg + npar;

        if (flag_d) {
            // deriv with regularization
            fdf_full.f = ffnn_f_deriv_reg;
            fdf_full.df = ffnn_df_deriv_reg;
        }
        else {
            // pure fdf with regularization
            fdf_full.f = ffnn_f_pure_reg;
            fdf_full.df = ffnn_df_pure_reg;
        }
        fdf_full.fvv = NULL;
        fdf_full.n = ntrain_full;
        fdf_full.p = npar;
        fdf_full.params = &tws;
    }
    else {
        ntrain_full = ntrain_noreg;
        nvali_full = nvali_noreg;
        fdf_full = fdf_noreg;
    };

    // allocate workspace with default parameters, also allocate space for validation
    w_full = gsl_multifit_nlinear_alloc (T_full, &_gsl_params, ntrain_full, npar);
    w_noreg = gsl_multifit_nlinear_alloc (T_noreg, &_gsl_params, ntrain_noreg, npar);
    w_pure = gsl_multifit_nlinear_alloc (T_pure, &_gsl_params, ntrain_pure, npar);
    if (_flag_vali) {
        tws.fvali = gsl_vector_alloc(nvali_full);
    }
    else {
        tws.fvali = NULL;
        if (verbose > 1) fprintf(stderr, "[NNTrainerGSL] Warning: Validation residual calculation disabled, i.e. no early stopping.\n");
    }

    // initialize solver with starting point and calculate initial cost
    gsl_multifit_nlinear_init(&gx.vector, &fdf_full, w_full);
    calcCosts(w_full, chi0, resih, tws.fvali, chi0_vali, resih);

    // run driver to find fit
    earlyStopDriver(w_full, &tws, verbose, status, info);

    // compute again final full cost and error of best fit parameters
    calcCosts(w_full, resi_full, chisq, tws.fvali, resi_vali_full, resih);
    calcFitErr(w_full, fit, err, ntrain, npar, chisq);

    // final unregularized cost calculation
    gsl_multifit_nlinear_init(&gx.vector, &fdf_noreg, w_noreg);
    calcCosts(w_noreg, resi_noreg, resih, tws.fvali, resi_vali_noreg, resih);

    // final pure (no deriv, no reg) cost calculation
    gsl_multifit_nlinear_init(&gx.vector, &fdf_pure, w_pure);
    calcCosts(w_pure, resi_pure, resih, tws.fvali, resi_vali_pure, resih);

    if (verbose > 1) {
        fprintf(stderr, "summary from method '%s/%s'\n", gsl_multifit_nlinear_name(w_full), gsl_multifit_nlinear_trs_name(w_full));
        fprintf(stderr, "number of iterations: %zu\n", gsl_multifit_nlinear_niter(w_full));
        fprintf(stderr, "function evaluations: %zu\n", fdf_full.nevalf);
        fprintf(stderr, "Jacobian evaluations: %zu\n", fdf_full.nevaldf);
        fprintf(stderr, "reason for stopping: %s\n", (info == 1) ? "failed validation" : "max steps || 0 residual");
        fprintf(stderr, "status = %s\n", gsl_strerror (status));

        fprintf(stderr, "initial |f(x)| = %f (train), %f (vali)\n", chi0, chi0_vali);
        fprintf(stderr, "final   |f(x)| = %f (train), %f (vali)\n", resi_full, resi_vali_full);
        fprintf(stderr, "w/o reg |f(x)| = %f (train), %f (vali)\n", resi_noreg, resi_vali_noreg);
        fprintf(stderr, "pure    |f(x)| = %f (train), %f (vali)\n", resi_pure, resi_vali_pure);
        fprintf(stderr, "chisq/dof = %g\n", chisq / dof);

        for(int i=0; i<npar; ++i) fprintf(stderr, "b%i      = %.5f +/- %.5f\n", i, fit[i], err[i]);
        fprintf(stderr, "\n");
    }

    // free allocations
    gsl_multifit_nlinear_free(w_full);
    gsl_multifit_nlinear_free(w_noreg);
    gsl_multifit_nlinear_free(w_pure);
    if (_flag_vali) {
        gsl_vector_free(tws.fvali);
    }
    delete tws.ffnn_vderiv;
};

