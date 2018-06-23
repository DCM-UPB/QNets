#include "ActivationFunctionManager.hpp"
#include "ActivationFunctionInterface.hpp"

#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>


int main(){
    using namespace std;

    const double TINY = 0.0001;
    const double dx = 0.0001;
    vector<double> x_to_test = {-3., -2.5, -2., -1.5, -1.0, -0.5, -0.25, -0.001, 0.001, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    for (ActivationFunctionInterface * actf : std_actf::supported_actf){
        // cout << "actf = " << actf->getIdCode() << endl;

        for (double x : x_to_test){
            // cout << "    x = " << x << endl;
            const double f = actf->f(x);


            // --- first derivative
            const double f1d = actf->f1d(x);

            const double xdx = x + dx;
            const double fdx = actf->f(xdx);
            const double num_f1d = (fdx - f)/dx;

            // cout << "        f1d     = " << f1d << endl;
            // cout << "        num_f1d = " << num_f1d << endl;
            assert( abs(num_f1d-f1d) < TINY );


            // --- second derivative
            const double f2d = actf->f2d(x);

            const double xmdx = x - dx;
            const double fmdx = actf->f(xmdx);
            const double num_f2d = (fdx - 2.*f + fmdx)/(dx*dx);

            // cout << "        f2d     = " << f2d << endl;
            // cout << "        num_f2d = " << num_f2d << endl;
            assert( abs(num_f2d-f2d) < TINY );


            // --- third derivative
            const double f3d = actf->f3d(x);

            const double fdxdx = actf->f(x+dx+dx);
            const double num_f3d = (fdxdx - 3.*fdx +3*f - fmdx)/(dx*dx*dx);

            // cout << "        f3d     = " << f3d << endl;
            // cout << "        num_f3d = " << num_f3d << endl;
            assert( abs(num_f3d-f3d) < TINY*20. );


            // -- check the fad function
            double fad_f, fad_f1d, fad_f2d, fad_f3d;
            actf->fad(x, fad_f, fad_f1d, fad_f2d, fad_f3d, true, true, true);
            assert( abs(fad_f-f) < TINY );
            assert( abs(fad_f1d-f1d) < TINY );
            assert( abs(fad_f2d-f2d) < TINY );
            assert( abs(fad_f3d-f3d) < TINY );
        }

        // cout << endl;
    }

    return 0;
}
