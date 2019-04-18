#ifndef FFNN_TRAIN_NNTRAINERGSL_HPP
#define FFNN_TRAIN_NNTRAINERGSL_HPP

#include "qnets/FeedForwardNeuralNetwork.hpp"
#include "qnets/train/NNTrainer.hpp"
#include "qnets/train/NNTrainingConfig.hpp"
#include "qnets/train/NNTrainingData.hpp"

#include <gsl/gsl_multifit_nlinear.h>

// expose the numerous hidden functions in details namespace, for testing
namespace nn_trainer_gsl_details
{
// full struct for training, to be passed to residual functions for GSL fit
struct training_workspace: public NNTrainingData, public NNTrainingConfig
{
    FeedForwardNeuralNetwork * ffnn = nullptr; // Storing a pointer to the to-be-trained FFNN
    FeedForwardNeuralNetwork * ffnn_vderiv = nullptr; // Storing a pointer to the to-be-trained FFNN with vderivs

    // validation residuals
    gsl_vector * fvali = nullptr;

    // residual flags
    bool flag_r{};
    bool flag_d1{};
    bool flag_d2{};

    void copyData(const NNTrainingData &tdata);
    void copyConfig(const NNTrainingConfig &tconfig);
    void copyDatConf(const NNTrainingData &tdata, const NNTrainingConfig &tconfig);
};

// helpers
int setBetas(FeedForwardNeuralNetwork * ffnn, const gsl_vector * betas);
int calcNData(const int &nbase, const int &yndim, const int &npar = 0, const int &xndim = 0, const int &nderiv = 2);
void calcRSS(const gsl_vector * f, double &chi, double &chisq);
void calcCosts(const gsl_vector * f, double &chi, double &chisq, const gsl_vector * fvali, double &chi_vali, double &chisq_vali);
void calcCosts(gsl_multifit_nlinear_workspace * w, double &chi, double &chisq, const gsl_vector * fvali, double &chi_vali, double &chisq_vali);
void calcFitErr(gsl_multifit_nlinear_workspace * w, double * fit, double * err, const int &ndata, const int &npar, const double &chisq);

// common residual functions
int ffnn_f(const gsl_vector * betas, training_workspace * tws, gsl_vector * f, bool flag_r, bool flag_d);
int ffnn_df(const gsl_vector * betas, training_workspace * tws, gsl_matrix * J, bool flag_r, bool flag_d);

// the residual functions passed to GSL
int ffnn_f_pure(const gsl_vector * betas, void * tws, gsl_vector * f);
int ffnn_df_pure(const gsl_vector * betas, void * tws, gsl_matrix * J);
int ffnn_f_deriv(const gsl_vector * betas, void * tws, gsl_vector * f);
int ffnn_df_deriv(const gsl_vector * betas, void * tws, gsl_matrix * J);
int ffnn_f_pure_reg(const gsl_vector * betas, void * tws, gsl_vector * f);
int ffnn_df_pure_reg(const gsl_vector * betas, void * tws, gsl_matrix * J);
int ffnn_f_deriv_reg(const gsl_vector * betas, void * tws, gsl_vector * f);
int ffnn_df_deriv_reg(const gsl_vector * betas, void * tws, gsl_matrix * J);

// driver routines
void printStepInfo(const gsl_multifit_nlinear_workspace * w, const training_workspace * tws, const int &status);
void earlyStopDriver(gsl_multifit_nlinear_workspace * w, const training_workspace * tws, const int &verbose, int &status, int &info);
} // namespace nn_trainer_gsl_details

// actual class
class NNTrainerGSL: public NNTrainer
{
protected:
    const gsl_multifit_nlinear_parameters _gsl_params; // to fine tune the gsl multifit algorithm, defaults to gsl default
public:
    NNTrainerGSL(const NNTrainingData &tdata, const NNTrainingConfig &tconfig, const gsl_multifit_nlinear_parameters &gsl_params = gsl_multifit_nlinear_default_parameters()):
            NNTrainer(tdata, tconfig), _gsl_params(gsl_params) {}
    ~NNTrainerGSL() override = default;

    // we implement findFit
    void findFit(FeedForwardNeuralNetwork * ffnn, double * fit, double * err, const int &verbose = 0) override;
};


#endif
