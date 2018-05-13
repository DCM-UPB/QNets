#ifndef NN_TRAINER
#define NN_TRAINER

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"
#include "NNTrainingData.hpp"

class NNTrainer
{

protected:
    NNTrainingData * const _tdata;
    FeedForwardNeuralNetwork * const _ffnn;

public:
    NNTrainer(NNTrainingData * const tdata) : _tdata(tdata), _ffnn(tdata->ffnn) {};
    //~NNTrainer();

    void bestFit(const int nsteps, const int nfits, const double tolresi, const bool verbose) {
        int npar = _ffnn->getNBeta();
        double fit[npar], bestfit[npar], err[npar], bestfit_err[npar];
        double resi_pure = -1.0, resi_noreg = -1.0, resi_full = -1.0, bestresi_pure = -1.0, bestresi_noreg = -1.0, bestresi_full = -1.0;

        int ifit = 0;
        while(true) {
            // initial parameters
            _ffnn->randomizeBetas();
            for (int i = 0; i<npar; ++i) {
                fit[i] = _ffnn->getBeta(i);
            }

            findFit(fit, err, resi_full, resi_noreg, resi_pure, nsteps, verbose);

            for (int i = 0; i<npar; ++i) {
                fit[i] = _ffnn->getBeta(i);
            }

            if(ifit < 1 || abs(resi_noreg) < bestresi_noreg) {
                for(int i = 0; i<npar; ++i){
                    bestfit[i] = fit[i];
                    bestfit_err[i] = err[i];
                }
                bestresi_full = resi_full;
                bestresi_noreg = resi_noreg;
                bestresi_pure = resi_pure;
            }

            ++ifit;

            if (abs(resi_noreg) <= tolresi) {
                if (verbose) fprintf(stderr, "Unregularized fit residual %f (full: %f, pure: %f) meets tolerance %f. Exiting with good fit.\n\n", resi_noreg, resi_full, resi_pure, tolresi);
                break;
            } else {
                if (verbose) fprintf(stderr, "Unregularized fit residual %f (full: %f, pure: %f) above tolerance %f.\n", resi_noreg, resi_full, resi_pure, tolresi);
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
    /*
    // compute fit distance for best betas
    double getFitDistance() {
        double dist = 0.0;
        for(int i=0; i<_ndata; ++i) {
            _ffnn->setInput(0, _xdata[i]);
            _ffnn->FFPropagate();
            dist += pow(_ydata[i]-_ffnn->getOutput(0), 2);
        }
        return dist / _ndata / pow(_yscale, 2);
    }

    // compare NN to data from index i0 to ie in increments di
    void compareFit(const int i0=0, const int ie=-1, const int di = 1) {
        using namespace std;

        const int realie = (ie<0)? _ndata-1:ie; //set default ie although _ndata is not const

        int j=i0;
        while(j<_ndata && j<=realie){
            _ffnn->setInput(0, _xdata[j]);
            _ffnn->FFPropagate();
            cout << "x: " << _xdata[j] / _xscale - _xshift << " f(x): " << _ydata[j] / _yscale - _yshift << " nn(x): " << _ffnn->getOutput(0) / _yscale - _yshift << endl;
            j+=di;
        }
        cout << endl;
    }
    */

    // print output of fitted NN to file
    void printFitOutput(const double min, const double max, const int npoints, const double xscale, const double yscale, const double xshift, const double yshift, const bool print_d1 = false, const bool print_d2 = false) {
        double base_input = 0.0;
        writePlotFile(_ffnn, &base_input, 0, 0, min, max, npoints, "getOutput", "v.txt", xscale, yscale, xshift, yshift);
        if (print_d1 && _tdata->flag_d1) writePlotFile(_ffnn, &base_input, 0, 0, min, max, npoints, "getFirstDerivative", "d1.txt", xscale, yscale, xshift, yshift);
        if (print_d2 && _tdata->flag_d2) writePlotFile(_ffnn, &base_input, 0, 0, min, max, npoints, "getSecondDerivative", "d2.txt", xscale, yscale, xshift, yshift);
    }

    // store fitted NN in file
    void printFitNN() {_ffnn->storeOnFile("nn.txt");}

    virtual void findFit(double * const fit, double * const err, double &resi_full, double &resi_noreg, double &resi_pure, const int nsteps, const bool verbose) = 0; // to be implemented by child
};


#endif
