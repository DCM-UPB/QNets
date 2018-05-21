#ifndef ACTIVATION_FUNCTION_INTERFACE
#define ACTIVATION_FUNCTION_INTERFACE


#include <string>


class ActivationFunctionInterface
{
protected:

public:

    //return a 3-characters identification string
    virtual std::string getIdCode() = 0;

    // compute the activation function value
    virtual double f(const double &) = 0;

    // first derivative of the activation function
    virtual double f1d(const double &) = 0;

    // second derivative of the activation function
    virtual double f2d(const double &) = 0;

    // third derivative of the activation function
    virtual double f3d(const double &) = 0;

    // function to calculate all needed derivatives together, which potentially allows for speedup
    virtual void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false)
    {
        // generic implementation
        v = f(in);

        if (flag_d1) v1d = this->f1d(in);
        else v1d = 0.;

        if (flag_d2) v2d = this->f2d(in);
        else v2d = 0.;

        if (flag_d3) v3d = this->f3d(in);
        else v3d = 0.;
    };
};


#endif
