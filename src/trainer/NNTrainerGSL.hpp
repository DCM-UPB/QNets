#ifndef NN_TRAINER_GSL
#define NN_TRAINER_GSL

#include "NNTrainer.hpp"
#include "NNTrainingData.hpp"
#include "NNTrainingConfig.hpp"
#include "GSLFitStruct.hpp"
#include "FeedForwardNeuralNetwork.hpp"

#include <gsl/gsl_multifit_nlinear.h>

// expose the numerous hidden functions in details namespace, for testing
namespace nn_trainer_gsl_details {
    int setBetas(FeedForwardNeuralNetwork * const ffnn, const gsl_vector * const betas);
    int calcNData(const int &nbase, const int &yndim, const int &nbeta = 0, const int &xndim = 0, const int &nderiv = 2);
    void calcOffset1(const int &nbase, const int &yndim, int &off);
    void calcOffset12(const int &nbase, const int &yndim, const int &xndim, int &offd1, int &offd2);
    void calcOffset123(const int &nbase, const int &yndim, const int &xndim, int &offd1, int &offd2, int &offr);
    void calcRSS(const gsl_vector * const f, double &chi, double &chisq);
    void calcCosts(const gsl_vector * const f, double &chi, double &chisq, const gsl_vector * const fvali, double &chi_vali, double &chisq_vali);
    void calcCosts(gsl_multifit_nlinear_workspace * const w, double &chi, double &chisq, const gsl_vector * const fvali, double &chi_vali, double &chisq_vali);
    void calcFitErr(gsl_multifit_nlinear_workspace * const w, double * const fit, double * const err, const int &ndata, const int &npar, const double &chisq);

    int ffnn_f_pure(const gsl_vector * betas, void * const tstruct, gsl_vector * f);
    int ffnn_df_pure(const gsl_vector * betas, void * const tstruct, gsl_matrix * J);
    int ffnn_f_deriv(const gsl_vector * betas, void * const tstruct, gsl_vector * f);
    int ffnn_df_deriv(const gsl_vector * betas, void * const tstruct, gsl_matrix * J);
    int ffnn_f_pure_reg(const gsl_vector * betas, void * const tstruct, gsl_vector * f);
    int ffnn_df_pure_reg(const gsl_vector * betas, void * const tstruct, gsl_matrix * J);
    int ffnn_f_deriv_reg(const gsl_vector * betas, void * const tstruct, gsl_vector * f);
    int ffnn_df_deriv_reg(const gsl_vector * betas, void * const tstruct, gsl_matrix * J);

    void printStepInfo(const gsl_multifit_nlinear_workspace * const w, const GSLFitStruct * const tstruct, const int &status);
    void earlyStopDriver(gsl_multifit_nlinear_workspace * const w, const GSLFitStruct * const tstruct, const int &verbose, int &status, int &info);
};

// actual class
class NNTrainerGSL: public NNTrainer
{
protected:
    GSLFitStruct _tstruct;
    const gsl_multifit_nlinear_parameters _gsl_params;
public:
    NNTrainerGSL(const NNTrainingData &tdata, const NNTrainingConfig &tconfig, const gsl_multifit_nlinear_parameters &gsl_params = gsl_multifit_nlinear_default_parameters()): NNTrainer(tdata, tconfig), _gsl_params(gsl_params) {_tstruct.copyDatConf(_tdata, _tconfig);}
    ~NNTrainerGSL(){}

    // we implement findFit
    void findFit(FeedForwardNeuralNetwork * const ffnn, double * const fit, double * const err, const int &verbose = 0);
};


#endif
