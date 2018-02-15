
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>


/*
Example: Train FFNN via GSL's Non-Linear Fit with Levenberg-Marquardt solver

In this example we want to demonstrate how a FFNN can be trained via GSL's fit routines to fit target data (in this case a gaussian).

The fit is achieved by minimizing the mean square distance of NN to data (by Levenberg-Marquardt), where the distance and gradient are integrated on a unifrom grid.
Hence, we have to create the NN and target data, define functions for distance and gradient, and pass all the information to the GSL lib.

For convenience, a NNFitter1D class is created, which handles the whole process in a user-friendly way, including normalization of the data.
*/


double Gaussian(const double &x, const double &a, const double &b) {
  return exp(-a*pow(x-b, 2));
};

struct data {
  const int n;
  double * x;
  double * y;
  FeedForwardNeuralNetwork * ffnn;
};

int ffnn_f(const gsl_vector * betas, void * data, gsl_vector * f) {
  const int n = ((struct data *)data)->n;
  const double * x = ((struct data *)data)->x;
  const double * y = ((struct data *)data)->y;
  FeedForwardNeuralNetwork * ffnn = ((struct data *)data)->ffnn;

  //set new NN betas
  for (int i=0; i<ffnn->getNBeta(); ++i){
    ffnn->setBeta(i, gsl_vector_get(betas, i));
  }

  //get NN output
  for (int i=0; i<n; ++i) {
    ffnn->setInput(0, x[i]);
    ffnn->FFPropagate();
    gsl_vector_set(f, i, ffnn->getOutput(0) - y[i]);
  }

  return GSL_SUCCESS;
};

int ffnn_df(const gsl_vector * betas, void * data, gsl_matrix * J) {
  const int n = ((struct data *)data)->n;
  const double * x = ((struct data *)data)->x;
  FeedForwardNeuralNetwork * ffnn = ((struct data *)data)->ffnn;

  //set new NN betas
  for (int i=0; i<ffnn->getNBeta(); ++i){
    ffnn->setBeta(i, gsl_vector_get(betas, i));
  }

  for (int i=0; i<n; ++i) {
    ffnn->setInput(0, x[i]);
    ffnn->FFPropagate();
    for (int j=0; j<ffnn->getNBeta(); ++j){
      gsl_matrix_set(J, i, j, ffnn->getVariationalFirstDerivative(0, j));
    }
  }

  return GSL_SUCCESS;
};

void callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w) {
  double rcond;

  /* compute reciprocal condition number of J(x) */
  gsl_multifit_nlinear_rcond(&rcond, w);
};


void callback_verbose(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w) {
  gsl_vector *f = gsl_multifit_nlinear_residual(w);
  gsl_vector *x = gsl_multifit_nlinear_position(w);
  double rcond;

  /* compute reciprocal condition number of J(x) */
  gsl_multifit_nlinear_rcond(&rcond, w);

  fprintf(stderr, "iter %2zu: cond(J) = %8.4f, |f(x)| = %.4f\n", iter, 1.0 / rcond, gsl_blas_dnrm2(f));

  for (int i=0; i<x->size; ++i) fprintf(stderr, "b%i: %f, ", i,  gsl_vector_get(x, i));
  fprintf(stderr, "\n");
};

// hardcoded target values of logistic actf
#define ACTF_X0 0.0
#define ACTF_XS 1.0
#define ACTF_XD ACTF_XS*3.464101615 //uniform distribution: sigma=(b-a)/sqrt(12)
#define ACTF_Y0 0.5
#define ACTF_YD 1.0
class NNFitter1D {
private:
  int _ndata, _npar;

  double * _xdata;
  double * _ydata;
  double * _weights;

  double _xscale,  _yscale;
  double _xshift,  _yshift;

  bool _flag_d1, _flag_d2;

  FeedForwardNeuralNetwork * _ffnn;

public:
  NNFitter1D(const int &nhlayer, int * nhunits, const int &ndata, const double * xdata, const double * ydata, double * weights, const bool flag_d1 = false, const bool flag_d2 = false) {
    using namespace std;

    _ndata = ndata;

    _xdata = new double[ndata]; // we need own copies to normalize the data
    _ydata = new double[ndata];
    _weights = weights;

    // calculate values for normalization
    auto mimax = minmax_element(xdata, xdata + ndata);
    auto mimay = minmax_element(ydata, ydata + ndata);
    double x0 = 0.5*(*mimax.first + *mimax.second);
    double y0 = 0.5*(*mimay.first + *mimay.second);
    double xd = *mimax.second - *mimax.first;
    double yd = *mimay.second - *mimay.first;

    _xscale = ACTF_XD/xd;
    _yscale = ACTF_YD/yd;
    _xshift = ACTF_X0 - x0;
    _yshift = ACTF_Y0 - y0;

    for (int i = 0; i < ndata; ++i) {
      _xdata[i] = (xdata[i] + _xshift) * _xscale;
      _ydata[i] = (ydata[i] + _yshift) * _yscale;
    }

    _flag_d1 = flag_d1;
    _flag_d2 = flag_d2;

    _ffnn = new FeedForwardNeuralNetwork(2, nhunits[0], 2);
    for (int i = 1; i<nhlayer; ++i) _ffnn->pushHiddenLayer(nhunits[i]);
    _ffnn->connectFFNN();
    if (flag_d1) _ffnn->addFirstDerivativeSubstrate();
    if (flag_d2) _ffnn->addSecondDerivativeSubstrate();
    _ffnn->addVariationalFirstDerivativeSubstrate();
    _ffnn->randomizeBetas();

    _npar = _ffnn->getNBeta();
    //    for (int i= 1; i<_npar; ++i) _ffnn->setBeta(i, 0.1);
  }

  ~NNFitter1D(){
    delete _ffnn;
    delete [] _xdata;
    delete [] _ydata;
  }

  void findFit(const int &nsteps, const int &nfits, const double &maxchi, const bool &doprint) {
  /*
  Fit NN to data with following parameters:
  nsteps : number of fitting iterations
  nfits : maximum number of fits to achieve good fit
  maxchi : maximum tolerable residual to consider a fit good
  doprint: print verbose output while fitting
  */

  // build data struct
  struct data d = { _ndata, _xdata, _ydata, _ffnn};

  //things for gsl multifit
  const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
  gsl_multifit_nlinear_fdf fdf;
  gsl_multifit_nlinear_workspace * w;
  gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();

  gsl_vector *f;
  gsl_matrix *J;
  gsl_matrix * covar = gsl_matrix_alloc (_npar, _npar);

  double x_init[_npar], x_best[_npar], x_best_err[_npar];
  gsl_vector_view gx = gsl_vector_view_array (x_init, _npar);
  gsl_vector_view wts = gsl_vector_view_array(_weights, _ndata);
  double chisq, chisq0, chi0, chi, c, dof = _ndata - _npar, bestchi = -1.0;
  int status, info;
  size_t i, ifit;

  double xtol = 0.0;
  double gtol = 0.0;
  double ftol = 0.0;
  //


  // define the function to be minimized
  fdf.f = ffnn_f;
  fdf.df = ffnn_df;   // set to NULL for finite-difference Jacobian
  fdf.fvv = NULL;     // not using geodesic acceleration
  fdf.n = _ndata;
  fdf.p = _npar;
  fdf.params = &d;

  // allocate workspace with default parameters
  w = gsl_multifit_nlinear_alloc (T, &fdf_params, _ndata, _npar);

  #define FIT(i) gsl_vector_get(w->x, i)
  #define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

  ifit = 0;
  while(true) {
    // initial parameters
    for (i = 0; i<_npar; ++i) {
      x_init[i] = _ffnn->getBeta(i);
    }

    // initialize solver with starting point and weights
    gsl_multifit_nlinear_winit(&gx.vector, &wts.vector, &fdf, w);

    // compute initial cost function
    f = gsl_multifit_nlinear_residual(w);
    gsl_blas_ddot(f, f, &chisq0);
    chi0 = sqrt(chisq0);

    // solve the system with a maximum of nsteps iterations
    if (doprint) status = gsl_multifit_nlinear_driver(nsteps, xtol, gtol, ftol, callback_verbose, NULL, &info, w);
    else status = gsl_multifit_nlinear_driver(nsteps, xtol, gtol, ftol, callback, NULL, &info, w);

    // compute covariance of best fit parameters
    J = gsl_multifit_nlinear_jac(w);
    gsl_multifit_nlinear_covar(J, 0.0, covar);

    // compute final cost
    gsl_blas_ddot(f, f, &chisq);
    chi = sqrt(chisq);
    c = GSL_MAX_DBL(1, sqrt(chisq / dof));

    if (doprint) {
      fprintf(stderr, "summary from method '%s/%s'\n", gsl_multifit_nlinear_name(w), gsl_multifit_nlinear_trs_name(w));
      fprintf(stderr, "number of iterations: %zu\n", gsl_multifit_nlinear_niter(w));
      fprintf(stderr, "function evaluations: %zu\n", fdf.nevalf);
      fprintf(stderr, "Jacobian evaluations: %zu\n", fdf.nevaldf);
      fprintf(stderr, "reason for stopping: %s\n", (info == 1) ? "small step size" : "small gradient");
      fprintf(stderr, "initial |f(x)| = %f\n", chi0);
      fprintf(stderr, "final   |f(x)| = %f\n", chi);
      fprintf(stderr, "chisq/dof = %g\n", chisq / dof);

      for(i=0; i<_npar; ++i) fprintf(stderr, "b%zu      = %.5f +/- %.5f\n", i, FIT(i), c*ERR(i));

      fprintf(stderr, "status = %s\n", gsl_strerror (status));
    }

    if(ifit < 1 || chi < bestchi) {
      for(i = 0; i<_npar; ++i){
        x_best[i] = FIT(i);
        x_best_err[i] = c*ERR(i);
      }
      bestchi = chi;
    }

    ++ifit;

    if (bestchi <= maxchi) {
      fprintf(stderr, "Fit residual %f meets tolerance %f. Exiting with good fit.\n\n", bestchi, maxchi);
      break;
    } else if (ifit >= nfits) {
      fprintf(stderr, "Maximum number of fits reached (%zu). Exiting with best fit residual %f.\n\n", nfits, bestchi);
      break;
    } else {
      _ffnn->randomizeBetas();
      fprintf(stderr, "Fit residual %f above tolerance %f. Let's try again.\n", chi, maxchi);
    }
  }

  if (doprint) {
    fprintf(stderr, "best fit summary:\n");
    for(i=0; i<_npar; ++i) fprintf(stderr, "b%zu      = %.5f +/- %.5f\n", i, x_best[i], x_best_err[i]);
    fprintf(stderr, "|f(x)| = %f\n", bestchi);
    fprintf(stderr, "chisq/dof = %g\n", bestchi*bestchi / dof);
  }

  for (i=0; i<_npar; ++i) {
    _ffnn->setBeta(i, x_best[i]); //set ffnn to best fit
  }

  gsl_multifit_nlinear_free(w);
  gsl_matrix_free(covar);

  }

  // compute fit distance for CG's best betas
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
    //if (ie<0) realie = _ndata-1;
    //else realie = ie;

    int j=i0;
    while(j<_ndata && j<=realie){
      _ffnn->setInput(0, _xdata[j]);
      _ffnn->FFPropagate();
      cout << "x: " << _xdata[j] / _xscale - _xshift << " f(x): " << _ydata[j] / _yscale - _yshift << " nn(x): " << _ffnn->getOutput(0) / _yscale - _yshift << endl;
      j+=di;
    }
    cout << endl;
  }

  // print output of fitted NN to file
  void printFitOutput(const double &min, const double &max, const int &npoints, const bool &print_d1 = false, const bool &print_d2 = false) {
    double base_input = 0.0;
    writePlotFile(_ffnn, &base_input, 0, 0, min, max, npoints, "getOutput", "v.txt", _xscale, _yscale, _xshift, _yshift);
    if (print_d1 && _flag_d1) writePlotFile(_ffnn, &base_input, 0, 0, min, max, npoints, "getFirstDerivative", "d1.txt", _xscale, _yscale, _xshift, _yshift);
    if (print_d2 && _flag_d2) writePlotFile(_ffnn, &base_input, 0, 0, min, max, npoints, "getSecondDerivative", "d2.txt", _xscale, _yscale, _xshift, _yshift);
  }

  // store fitted NN in file
  void printFitNN() {_ffnn->storeOnFile("nn.txt");}

  FeedForwardNeuralNetwork * getFFNN() {return _ffnn;}
};


int main (void) {
  using namespace std;

  NNFitter1D * fitter;

  double lb = -10;
  double ub = 10;
  int ndata = 2001;
  double xdata[ndata];
  double ydata[ndata];
  double weights[ndata];

  int nl, nhl, nhu[2], nfits = 1;
  double maxchi;

  cout << "Let's start by creating a Feed Forward Artificial Neural Network (FFANN)" << endl;
  cout << "========================================================================" << endl;
  cout << endl;
  cout << "How many units should the first hidden layer(s) have? ";
  cin >> nhu[0];
  cout << "How many units should the second hidden layer(s) have? (<=1 for none) ";
  cin >> nhu[1];
  cout << endl;

  // NON I/O CODE
  nl = (nhu[1]>1)? 4:3;
  nhl = nl-2;
  //

  cout << "We generate a FFANN with " << nl << " layers and 2, " << nhu[0];
  if (nhu[1]>0) { cout << ", " << nhu[1];}
  cout << ", 2 units respectively" << endl;
  cout << "========================================================================" << endl;
  cout << endl;
  cout << "In the following we use GSL non-linear fit to minimize the mean-squared-distance of NN vs. target function, i.e. find optimal betas." << endl;
  cout << endl;
  cout << "Please enter the the maximum tolerable fit residual. ";
  cin >> maxchi;
  cout << "Please enter the maximum number of fitting runs. ";
  cin >> nfits;
  cout << endl << endl;
  cout << "Now we find the best fit ... " << endl;


  // NON I/O CODE

  // this is the data to be fitted
  double dx = (ub-lb) / (ndata-1);
  for (int i = 0; i < ndata; ++i) {
    xdata[i] = lb + i*dx;
    ydata[i] = Gaussian(xdata[i], 1, 0);
    weights[i] = 1.0;
    //printf ("data: %zu %g %g\n", i, x[i], y[i]);
  };


  fitter = new NNFitter1D(nhl, nhu, ndata, xdata, ydata, weights, true, true);
  fitter->findFit(100, nfits, maxchi, false);
  //

  cout << "Done." << endl;
  cout << "========================================================================" << endl;
  cout << endl;
  cout << "Finally we compare the best fit NN to the target function:" << endl << endl;

  // NON I/O CODE
  fitter->compareFit(0, ndata, 100);
  //

  cout << endl;
  cout << "And print the output/NN to a file. The end." << endl;

  // NON I/O CODE
  fitter->printFitOutput(-10, 10, 200, true, true);
  fitter->printFitNN();
  //

  delete fitter;

  return 0;
}
