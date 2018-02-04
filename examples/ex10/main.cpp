
#include <iostream>
#include <cmath>
#include <math.h>

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>


/*
Example: Fit FFNN via NFM's Conjugate Gradient

In this example we want to demonstrate how a FFNN can be trained via Conjuage Gradient (CG) to fit a target function (in this case a gaussian).

This is done by minimizing the quadratic distance of NN to target function via class CG from NFM. The distance and gradient functions are integrated by MCI.
Hence, we have to create the NN and target function, MCIObservableFunctionInterfaces for distance and gradient, and put everything together into a NoisyFunctionWithGradient for CG.

The whole process is handled by the NNFitter1D class. As a little speciality, the fitting is done multiple times in parallel threads and the best fit overall is presented as result.
*/


//1D Target Function
class Function1D {
public:
  //Function1D(){}
  virtual ~Function1D(){}

  // Function evaluation
  virtual double f(const double &) = 0;
  //                    ^input
};

// exp(-a*(x-b)^2)
class Gaussian: public Function1D{
private:
  double _a;
  double _b;

public:
  Gaussian(const double &a, const double &b): Function1D(){
    _a = a;
    _b = b;
  }

  double f(const double &x){
    return exp(-_a*pow(x-_b, 2));
  }
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
    ffnn->setInput(1, &x[i]);
    ffnn->FFPropagate();
    gsl_vector_set(f, i, ffnn->getOutput(1) - y[i]);
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
    ffnn->setInput(1, &x[i]);
    ffnn->FFPropagate();
    for (int j=0; j<ffnn->getNBeta(); ++j){
      gsl_matrix_set(J, i, j, ffnn->getVariationalFirstDerivative(1, j));
    }
  }

  return GSL_SUCCESS;
};

void callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w) {
  gsl_vector *f = gsl_multifit_nlinear_residual(w);
  gsl_vector *x = gsl_multifit_nlinear_position(w);
  double rcond;

  using namespace std;

  /* compute reciprocal condition number of J(x) */
  gsl_multifit_nlinear_rcond(&rcond, w);

  fprintf(stderr, "iter %2zu: cond(J) = %8.4f, |f(x)| = %.4f\n", iter, 1.0 / rcond, gsl_blas_dnrm2(f));

  for (int i=0; i<x->size; ++i) cout << "b" << i << ": " << gsl_vector_get(x, i) << ", ";
  cout << endl << endl;
};

/* number of data points to fit */
#define N 2000

int main (void) {

  Gaussian * gauss = new Gaussian(1,0);
  // variables for data struct
  const int n = N;
  double x[n], y[n], weights[n];
  FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(2, 14, 2);
  ffnn->connectFFNN();
  ffnn->addVariationalFirstDerivativeSubstrate();

  struct data d = { n, x, y, ffnn};
  //
  int p = ffnn->getNBeta();

  //things for gsl multifit
  const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
  gsl_multifit_nlinear_workspace *w;
  gsl_multifit_nlinear_fdf fdf;
  gsl_multifit_nlinear_parameters fdf_params =
    gsl_multifit_nlinear_default_parameters();

  gsl_vector *f;
  gsl_matrix *J;
  gsl_matrix *covar = gsl_matrix_alloc (p, p);

  double x_init[ffnn->getNBeta()];
  gsl_vector_view gx = gsl_vector_view_array (x_init, p);
  gsl_vector_view wts = gsl_vector_view_array(weights, n);
  double chisq, chisq0;
  int status, info;
  size_t i;

  const double xtol = 1e-8;
  const double gtol = 1e-8;
  const double ftol = 0.0;
  //


  // define the function to be minimized
  fdf.f = ffnn_f;
  fdf.df = ffnn_df;   // set to NULL for finite-difference Jacobian
  fdf.fvv = NULL;     // not using geodesic acceleration
  fdf.n = n;
  fdf.p = p;
  fdf.params = &d;

  // this is the data to be fitted
  double lb = -10;
  double ub = 10;
  double dx = (ub-lb) / (n-1);
  for (i = 0; i < n; ++i) {

    x[i] = lb + i*dx;
    y[i] = gauss->f(x[i]);
    weights[i] = 1.0;
    //printf ("data: %zu %g %g\n", i, x[i], y[i]);
  };

  // initial parameters
  for (i = 0; i<ffnn->getNBeta(); ++i) {
    x_init[i] = ffnn->getBeta(i);
  }

  // allocate workspace with default parameters
  w = gsl_multifit_nlinear_alloc (T, &fdf_params, n, p);

  // initialize solver with starting point and weights
  gsl_multifit_nlinear_winit (&gx.vector, &wts.vector, &fdf, w);

  // compute initial cost function
  f = gsl_multifit_nlinear_residual(w);
  gsl_blas_ddot(f, f, &chisq0);

  // solve the system with a maximum of 50 iterations
  status = gsl_multifit_nlinear_driver(50, xtol, gtol, ftol, callback, NULL, &info, w);

  // compute covariance of best fit parameters
  J = gsl_multifit_nlinear_jac(w);
  gsl_multifit_nlinear_covar (J, 0.0, covar);

  // compute final cost
  gsl_blas_ddot(f, f, &chisq);

#define FIT(i) gsl_vector_get(w->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

  fprintf(stderr, "summary from method '%s/%s'\n",
          gsl_multifit_nlinear_name(w),
          gsl_multifit_nlinear_trs_name(w));
  fprintf(stderr, "number of iterations: %zu\n",
          gsl_multifit_nlinear_niter(w));
  fprintf(stderr, "function evaluations: %zu\n", fdf.nevalf);
  fprintf(stderr, "Jacobian evaluations: %zu\n", fdf.nevaldf);
  fprintf(stderr, "reason for stopping: %s\n",
          (info == 1) ? "small step size" : "small gradient");
  fprintf(stderr, "initial |f(x)| = %f\n", sqrt(chisq0));
  fprintf(stderr, "final   |f(x)| = %f\n", sqrt(chisq));

  {
    double dof = n - p;
    double c = GSL_MAX_DBL(1, sqrt(chisq / dof));

    fprintf(stderr, "chisq/dof = %g\n", chisq / dof);

    for(int i=0; i<p; ++i) fprintf (stderr, "b%zu      = %.5f +/- %.5f\n", i, FIT(i), c*ERR(i));
  }

  fprintf (stderr, "status = %s\n", gsl_strerror (status));

  double * base_input = new double[ffnn->getNInput()];
  writePlotFile(ffnn, base_input, 0, 1, lb, ub, 200, "getOutput", "v.txt");
  delete [] base_input;

  gsl_multifit_nlinear_free (w);
  gsl_matrix_free (covar);

  delete ffnn;

  return 0;
}

/*
// creates instances and holds the necessary data for multiple fit threads in parallel
class NNFitter1D {
private:
  int _nhlayer;
  int * _nhunits;
  long _nmc;
  double * _irange;

  Function1D * _ftarget;
  FeedForwardNeuralNetwork * _ffnn;
  FitNN1D * _fitnn;
  ConjGrad * _conj;

public:
  NNFitter1D(Function1D * ftarget, const int &nhlayer, int * nhunits, const long &nmc, double * irange) {
    _ftarget = ftarget;
    _nhlayer = nhlayer;
    _nhunits = nhunits;
    _nmc = nmc;
    _irange = irange;
  }

  ~NNFitter1D(){
    delete _ffnn;
    delete _fitnn;
    delete _conj;
  }

  void init() { // capsule this init part away for threading
    _ffnn = new FeedForwardNeuralNetwork(2, _nhunits[0], 2);
    for (int i = 1; i<_nhlayer; ++i) _ffnn->pushHiddenLayer(_nhunits[i]);
    _ffnn->connectFFNN();
    _ffnn->addVariationalFirstDerivativeSubstrate();

    _fitnn = new FitNN1D(_ffnn, _ftarget, _nmc, _irange);

    _conj = new ConjGrad(_fitnn);
    for(int i = 0; i<_ffnn->getNBeta(); ++i) _conj->setX(i, _ffnn->getBeta(i));
  }

  void findFit() {
    _conj->findMin();
  }

  // compute fit distance for CG's best betas
  double getFitDistance() {
    double betas[_ffnn->getNBeta()];
    double f,df;

    for (int i=0; i<_ffnn->getNBeta(); ++i) {
      betas[i] = _conj->getX(i);
    }
    _fitnn->f(betas, f, df);
    return f;
  }

  // compare NN to target at nx points starting from x=x0 in increments dx
  void compareFit(const double x0, const double dx, const int nx) {
    using namespace std;
    double x=x0;
    for(int i=0; i<nx; ++i) {
      _ffnn->setInput(1, &x);
      _ffnn->FFPropagate();
      cout << "x: " << x << " f(x): " << _ftarget->f(x) << " nn(x): " << _ffnn->getOutput(1) << endl;
      x+=dx;
    }
    cout << endl;
  }

  // print output of fitted NN to file
  void printFitOutput(int input_i, int output_i, const double &min, const double &max, const int &npoints) {
    double * base_input = new double[_ffnn->getNInput()];
    writePlotFile(_ffnn, base_input, input_i, output_i, min, max, npoints, "getOutput", "v.txt");
    delete [] base_input;
  }

  // store fitted NN in file
  void printFitNN() {_ffnn->storeOnFile("nn.txt");}

  FeedForwardNeuralNetwork * getFFNN() {return _ffnn;}
  FitNN1D * getFitNN() {return _fitnn;}
  ConjGrad * getConj() {return _conj;}
};

// code which runs in parallel
void findFit(void * voidPtr) {
  using namespace std;
  NNFitter1D * fitter = static_cast<NNFitter1D*>(voidPtr);
  fitter->init();
  try {
    fitter->findFit();
  }
  catch (runtime_error e) {
    cout << "Warning: Fit thread aborted because of too long bracketing." << endl;
  }
}

void *findFit_thread(void * voidPtr) {
  findFit(voidPtr);
  pthread_exit(NULL);
}*/

/*
int main() {
  using namespace std;

  int nl, nhl, nhu[2], nthread;
  const long nmc = 40000l;
  double irange[2];
  irange[0] = -10.;
  irange[1] = 10.;

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
  if (nhu[1]>1) { cout << ", " << nhu[1];}
  cout << ", 2 units respectively" << endl;
  cout << "========================================================================" << endl;
  cout << endl;
  cout << "In the following we use CG to minimize the mean-squared-distance of NN vs. target function, i.e. find optimal betas." << endl;
  cout << endl;
  cout << "How many fitting threads do you want to spawn? ";
  cin >> nthread;
  cout << endl << endl;
  cout << "Now we spawn fitting threads and find the best fit ... " << endl;;

  // NON I/O CODE

  // Preparing list of NNFitters for the threads
  pthread_t * my_threads = new pthread_t[nthread];
  int ret[nthread];

  Gaussian * gauss = new Gaussian(0.5,0.);

  NNFitter1D * fit_list[nthread];
  for (int i=0; i<nthread; ++i){
    fit_list[i] = new NNFitter1D(gauss, nhl, nhu, nmc, irange);
  }

  // Spawn minimzation threads
  for (int i=0; i<nthread; ++i) {
    ret[i] =  pthread_create(&my_threads[i], NULL, &findFit_thread, fit_list[i]);
    if(ret[i] != 0) {
      printf("Error: pthread_create() failed\n");
      exit(EXIT_FAILURE);
    }
  }

  // Join the threads
  for (int i=0; i<nthread; ++i) pthread_join(my_threads[i], (void **) &ret[i]);

  // Find best fit index
  int bfi = 0;
  double fdist, bdist = fit_list[0]->getFitDistance();

  for(int i=1; i<nthread; ++i) {
    fdist = fit_list[i]->getFitDistance();
    if(fdist < bdist) {
      bfi = i;
      bdist = fdist;
    }
  }

  //

  cout << "Done." << endl;
  cout << "========================================================================" << endl;
  cout << endl;
  cout << "Finally we compare the best fit NN to the target function:" << endl << endl;

  // NON I/O CODE
  fit_list[bfi]->compareFit(-10., 1., 21);
  //

  cout << endl;
  cout << "And print the output/NN to a file. The end." << endl;

  // NON I/O CODE
  fit_list[bfi]->printFitOutput(0, 1, -10, 10, 200);
  fit_list[bfi]->printFitNN();
  //

  // cleanup

  delete [] my_threads;
  delete gauss;
  for (int i=0; i<nthread; ++i) delete fit_list[i];

  // end
  return 0;
}
*/

/* no-thread main for debug
int main() {
  using namespace std;

  int nl, nhl, nhu[2];
  nhu[0] = 15;
  nhu[1] = 0;
  nl = (nhu[1]>1)? 4:3;
  nhl = nl-2;


  const long nmc = 1000l;
  double irange[2];
  irange[0] = -10.;
  irange[1] = 10.;

  Gaussian * gauss = new Gaussian(0.5,0.);

  NNFitter1D * fitter = new NNFitter1D(gauss, nhl, nhu, nmc, irange);
  fitter->init();
  fitter->findFit();
  fitter->printFitOutput(0, 1, -10, 10, 200);



  delete gauss;
  delete fitter;
};
*/
