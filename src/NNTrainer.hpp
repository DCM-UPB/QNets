#ifndef NN_TRAINER
#define NN_TRAINER

#include "FeedForwardNeuralNetwork.hpp"
#include "NNTrainingData.hpp"

class NNTrainer
{

protected:
    FeedForwardNeuralNetwork * _ffnn;
    NNTrainingData * _tdata;

public:
     NNTrainer(NNTrainingData * tdata, FeedForwardNeuralNetwork * ffnn) {_tdata = tdata; _ffnn = ffnn;};
    //~NNTrainer();

    void bestFit(const int nsteps, const int nfits, const double maxresi, const bool verbose) {
        int npar = _ffnn->getNBeta();
        double fit[npar], bestfit[npar], err[npar], bestfit_err[npar];
        double resi_pure, resi_noreg, resi_full, bestresi_pure, bestresi_noreg = -1.0, bestresi_full;

        int ifit = 0;
        while(true) {
            // initial parameters
            _ffnn->randomizeBetas();
            for (int i = 0; i<npar; ++i) {
                fit[i] = _ffnn->getBeta(i);
            }

            NNTrainer::findFit(nsteps, fit, err, resi_full, resi_noreg, resi_pure, verbose);

            if(ifit < 1 || resi_noreg < bestresi_noreg) {
                for(int i = 0; i<npar; ++i){
                    bestfit[i] = fit[i];
                    bestfit_err[i] = err[i];
                }
                bestresi_full = resi_full;
                bestresi_noreg = resi_noreg;
                bestresi_pure = resi_pure;
            }

            ++ifit;

            if (resi_noreg <= maxresi) {
                if (verbose) fprintf(stderr, "Unregularized fit residual %f (full: %f, pure: %f) meets tolerance %f. Exiting with good fit.\n\n", resi_noreg, resi_full, resi_pure, maxresi);
                break;
            } else {
                if (verbose) fprintf(stderr, "Unregularized fit residual %f (full: %f, pure: %f) above tolerance %f.\n", resi_noreg, resi_full, resi_pure, maxresi);
                if (ifit >= nfits) {
                    if (verbose) fprintf(stderr, "Maximum number of fits reached (%i). Exiting with best unregularized fit residual %f.\n\n", nfits, bestresi_noreg);
                    break;
                }
                if (verbose) fprintf(stderr, "Let's try again.\n");
            }
        }

        if (verbose) {
            fprintf(stderr, "best fit summary:\n");
            for(int i=0; i<npar; ++i) fprintf(stderr, "b%i      = %.5f +/- %.5f\n", i, bestfit[i], bestfit_err[i]);
            fprintf(stderr, "|f(x)| = %f (w/o reg: %f, pure: %f)\n", bestresi_full, bestresi_noreg, bestresi_pure);
        }

    };

    void findFit(const int nsteps, double * const fit, double * const err, double &resi_full, double &resi_noreg, double &resi_pure, const bool verbose); // to be implemented by child
};


#endif
