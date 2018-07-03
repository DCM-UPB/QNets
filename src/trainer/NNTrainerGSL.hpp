#ifndef NN_TRAINER_GSL
#define NN_TRAINER_GSL

#include "NNTrainer.hpp"
#include "NNTrainingData.hpp"
#include "NNTrainingConfig.hpp"
#include "FeedForwardNeuralNetwork.hpp"

#include <gsl/gsl_multifit_nlinear.h>

// expose the numerous hidden functions in details namespace, for testing
namespace nn_trainer_gsl_details {
    // full struct for training, to be passed to residual functions for GSL fit
    struct training_workspace: public NNTrainingData, public NNTrainingConfig
    {
        FeedForwardNeuralNetwork * ffnn = NULL; // Storing a pointer to the to-be-trained FFNN
        FeedForwardNeuralNetwork * ffnn_vderiv = NULL; // Storing a pointer to the to-be-trained FFNN with vderivs

        // validation residuals
        gsl_vector * fvali_full = NULL;
        gsl_vector * fvali_noreg = NULL;
        gsl_vector * fvali_pure = NULL;

        // residual flags
        bool flag_r;
        bool flag_d1;
        bool flag_d2;

        void copyData(const NNTrainingData &tdata);
        void copyConfig(const NNTrainingConfig &tconfig);
        void copyDatConf(const NNTrainingData &tdata, const NNTrainingConfig &tconfig);
    };

    // helpers
    int setBetas(FeedForwardNeuralNetwork * const ffnn, const gsl_vector * const betas);
    int calcNData(const int &nbase, const int &yndim, const int &nbeta = 0, const int &xndim = 0, const int &nderiv = 2);
    void calcOffset1(const int &nbase, const int &yndim, int &off);
    void calcOffset12(const int &nbase, const int &yndim, const int &xndim, int &offd1, int &offd2);
    void calcOffset123(const int &nbase, const int &yndim, const int &xndim, int &offd1, int &offd2, int &offr);
    void calcRSS(const gsl_vector * const f, double &chi, double &chisq);
    void calcCosts(const gsl_vector * const f, double &chi, double &chisq, const gsl_vector * const fvali, double &chi_vali, double &chisq_vali);
    void calcCosts(gsl_multifit_nlinear_workspace * const w, double &chi, double &chisq, const gsl_vector * const fvali, double &chi_vali, double &chisq_vali);
    void calcFitErr(gsl_multifit_nlinear_workspace * const w, double * const fit, double * const err, const int &ndata, const int &npar, const double &chisq);

    // residuals
    int ffnn_f_pure(const gsl_vector * betas, void * const tws, gsl_vector * f);
    int ffnn_df_pure(const gsl_vector * betas, void * const tws, gsl_matrix * J);
    int ffnn_f_deriv(const gsl_vector * betas, void * const tws, gsl_vector * f);
    int ffnn_df_deriv(const gsl_vector * betas, void * const tws, gsl_matrix * J);
    int ffnn_f_pure_reg(const gsl_vector * betas, void * const tws, gsl_vector * f);
    int ffnn_df_pure_reg(const gsl_vector * betas, void * const tws, gsl_matrix * J);
    int ffnn_f_deriv_reg(const gsl_vector * betas, void * const tws, gsl_vector * f);
    int ffnn_df_deriv_reg(const gsl_vector * betas, void * const tws, gsl_matrix * J);

    // driver routines
    void printStepInfo(const gsl_multifit_nlinear_workspace * const w, const training_workspace * const tws, const int &status);
    void earlyStopDriver(gsl_multifit_nlinear_workspace * const w, const training_workspace * const tws, const int &verbose, int &status, int &info);
};

// actual class
class NNTrainerGSL: public NNTrainer
{
protected:
    const gsl_multifit_nlinear_parameters _gsl_params; // to fine tune the gsl multifit algorithm, defaults to gsl default
public:
    NNTrainerGSL(const NNTrainingData &tdata, const NNTrainingConfig &tconfig, const gsl_multifit_nlinear_parameters &gsl_params = gsl_multifit_nlinear_default_parameters()): NNTrainer(tdata, tconfig), _gsl_params(gsl_params) {}
    ~NNTrainerGSL(){}

    // we implement findFit
    void findFit(FeedForwardNeuralNetwork * const ffnn, double * const fit, double * const err, const int &verbose = 0);
};


#endif
