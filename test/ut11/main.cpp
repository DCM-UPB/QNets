#include "qnets/actf/NoOp.hpp"
#include "qnets/actf/ReLU.hpp"
#include "qnets/actf/Sigmoid.hpp"
#include "qnets/actf/TanSig.hpp"
#include "qnets/actf/Exp.hpp"

#include <vector>
#include <cassert>

template <class ACTF>
void checkACTFDerivatives(const std::vector<double> &x_to_test, double dx, double TINY)
{
    ACTF actf;
    const size_t ntest = x_to_test.size();
    std::vector<double> xlv(ntest), xrv(ntest);
    std::vector<double> d1v(ntest), d2v(ntest);
    std::vector<double> num_d1v(ntest), num_d2v(ntest);

    // setup xl, xr
    for (size_t i = 0; i < ntest; ++i) {
        xlv[i] = x_to_test[i] - dx;
        xrv[i] = x_to_test[i] + dx;
    }
    // copy x values to in/out vectors
    std::vector<double> flv(xlv), fmv(x_to_test), frv(xrv);

    // compute flv, fmv, frv
    actf.f(flv.data(), flv.data()+ntest);
    actf.f(fmv.data(), fmv.data()+ntest);
    actf.f(frv.data(), frv.data()+ntest);

    // compute num derivs
    for (size_t i = 0; i < ntest; ++i) {
        num_d1v[i] = (frv[i] - fmv[i])/dx; // first deriv
        num_d2v[i] = (frv[i] - 2.*fmv[i] + flv[i])/(dx*dx);
    }

    // test d1 call
    fmv = x_to_test; // reset fmv
    actf.fd1(fmv.data(), fmv.data()+ntest, d1v.data());
    for (size_t i = 0; i < ntest; ++i) {
        // std::cout << "    x = " << x_to_test[i] << std::endl;

        // std::cout << "        f1d     = " << d1v[i] << std::endl;
        // std::cout << "        num_f1d = " << num_d1v[i] << std::endl;
        assert(fabs(num_d1v[i] - d1v[i]) < TINY);
    }

    // test d2 call
    fmv = x_to_test; // reset fmv
    std::fill(d1v.begin(), d1v.end(), 0.); // reset d1
    actf.fd12(fmv.data(), fmv.data()+ntest, d1v.data(), d2v.data());
    for (size_t i = 0; i < ntest; ++i) {
        // std::cout << "    x = " << x_to_test[i] << std::endl;

        // std::cout << "        f1d     = " << d1v[i] << std::endl;
        // std::cout << "        num_f1d = " << num_d1v[i] << std::endl;
        assert(fabs(num_d1v[i] - d1v[i]) < TINY);

        // std::cout << "        f2d     = " << d2v[i] << std::endl;
        // std::cout << "        num_f2d = " << num_d2v[i] << std::endl;
        assert(fabs(num_d2v[i] - d2v[i]) < TINY);
    }

    // std::cout << std::endl;
}

int main()
{
    using namespace std;

    const double TINY_DEFAULT = 0.0001;
    const double dx = 0.0001;
    vector<double> x_to_test = {-3., -2.5, -2., -1.5, -1.0, -0.5, -0.25, -0.001, 0.001, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    using namespace actf;
    checkACTFDerivatives<NoOp>(x_to_test, dx, TINY_DEFAULT);
    checkACTFDerivatives<ReLU>(x_to_test, dx, TINY_DEFAULT);
    checkACTFDerivatives<Sigmoid>(x_to_test, dx, TINY_DEFAULT);
    checkACTFDerivatives<TanSig>(x_to_test, dx, TINY_DEFAULT);
    checkACTFDerivatives<Exp>(x_to_test, dx, 0.002);

    return 0;
}
