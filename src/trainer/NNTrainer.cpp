#include "NNTrainer.hpp"

void NNTrainer::bestFit(const int nsteps, const int nfits, const double tolresi, const int verbose)
{
    int npar = _tstruct.ffnn->getNBeta();
    double fit[npar], bestfit[npar], err[npar], bestfit_err[npar];
    double resi_pure = -1.0, resi_noreg = -1.0, resi_full = -1.0, bestresi_pure = -1.0, bestresi_noreg = -1.0, bestresi_full = -1.0;

    int ifit = 0;
    while(true) {
        // initial parameters
        _tstruct.ffnn->randomizeBetas();
        for (int i = 0; i<npar; ++i) {
            fit[i] = _tstruct.ffnn->getBeta(i);
        }

        findFit(fit, err, resi_full, resi_noreg, resi_pure, nsteps, verbose);

        for (int i = 0; i<npar; ++i) {
            fit[i] = _tstruct.ffnn->getBeta(i);
        }

        if(ifit < 1 || (resi_noreg>=0 && resi_noreg < bestresi_noreg)) {
            for(int i = 0; i<npar; ++i){
                bestfit[i] = fit[i];
                bestfit_err[i] = err[i];
            }
            bestresi_full = resi_full;
            bestresi_noreg = resi_noreg;
            bestresi_pure = resi_pure;
        }

        ++ifit;

        if (resi_noreg>=0 && resi_noreg <= tolresi) {
            if (verbose > 0) fprintf(stderr, "Unregularized fit residual %f (full: %f, pure: %f) meets tolerance %f. Exiting with good fit.\n\n", resi_noreg, resi_full, resi_pure, tolresi);
            break;
        } else {
            if (verbose > 0) fprintf(stderr, "Unregularized fit residual %f (full: %f, pure: %f) above tolerance %f.\n", resi_noreg, resi_full, resi_pure, tolresi);
            if (ifit >= nfits) {
                if (verbose > 0) fprintf(stderr, "Maximum number of fits reached (%i). Exiting with best unregularized fit residual %f.\n\n", nfits, bestresi_noreg);
                break;
            }
            if (verbose > 0) fprintf(stderr, "Let's try again.\n");
        }
    }

    if (verbose > 0) {
        fprintf(stderr, "best fit summary:\n");
        for(int i=0; i<npar; ++i) fprintf(stderr, "b%i      = %.5f +/- %.5f\n", i, bestfit[i], bestfit_err[i]);
        fprintf(stderr, "|f(x)| = %f (w/o reg: %f, pure: %f)\n", bestresi_full, bestresi_noreg, bestresi_pure);
    }

}

/*
    // compute fit distance for best betas
    double getFitDistance() {
        double dist = 0.0;
        for(int i=0; i<_ndata; ++i) {
            _tstruct.ffnn->setInput(0, _xdata[i]);
            _tstruct.ffnn->FFPropagate();
            dist += pow(_ydata[i]-_tstruct.ffnn->getOutput(0), 2);
        }
        return dist / _ndata / pow(_yscale, 2);
    }

    // compare NN to data from index i0 to ie in increments di
    void compareFit(const int i0=0, const int ie=-1, const int di = 1) {
        using namespace std;

        const int realie = (ie<0)? _ndata-1:ie; //set default ie although _ndata is not const

        int j=i0;
        while(j<_ndata && j<=realie){
            _tstruct.ffnn->setInput(0, _xdata[j]);
            _tstruct.ffnn->FFPropagate();
            cout << "x: " << _xdata[j] / _xscale - _xshift << " f(x): " << _ydata[j] / _yscale - _yshift << " nn(x): " << _tstruct.ffnn->getOutput(0) / _yscale - _yshift << endl;
            j+=di;
        }
        cout << endl;
    }
    */

// print output of fitted NN to file
void NNTrainer::printFitOutput(const double min, const double max, const int npoints, const double xscale, const double yscale, const double xshift, const double yshift, const bool print_d1, const bool print_d2)
{
    using namespace std;
    double base_input = 0.0;

    for (int i = 0; i<_tstruct.xndim; ++i) {
        for (int j = 0; j<_tstruct.yndim; ++j) {
            stringstream ss;
            ss << i << "_" << j << ".txt";
            writePlotFile(_tstruct.ffnn, &base_input, i, j, min, max, npoints, "getOutput", "v_" + ss.str(), xscale, yscale, xshift, yshift);
            if (print_d1) writePlotFile(_tstruct.ffnn, &base_input, i, j, min, max, npoints, "getFirstDerivative", "d1_" + ss.str(), xscale, yscale, xshift, yshift);
            if (print_d2) writePlotFile(_tstruct.ffnn, &base_input, i, j, min, max, npoints, "getSecondDerivative", "d2_" + ss.str(), xscale, yscale, xshift, yshift);
        }
    }
}
