#ifndef NETWORK_UNIT
#define NETWORK_UNIT

#include "ffnn/serial/SerializableComponent.hpp"
#include "ffnn/feed/FeederInterface.hpp"

#include <string>
#include <cstddef> // for NULL

// Generalized Network Unit
class NetworkUnit: public SerializableComponent
{
protected:
    // Unit core elements
    double _pv;    // protovalue, i.e. the signal value on in the input side (calculated by feeder in computeFeed)
    double _v;    // output side value of the unit (this is meant to be calculated by computeActivation)

    // coordinate derivatives
    int _nx0;    // number of first derivatives (i.e. the number of inputs of the NN)
    double * _v1d;    // first derivatives
    double * _v2d;    // second derivative

    // internal variables which store the activation (computeActivation)
    double _a1d, _a2d, _a3d;

    // internal variables which store the feed derivatives (computeFeed)
    double * _first_der;    // _feeder->getFirstDerivativeFeed(i)
    double * _second_der;    // _feeder->getSecondDerivativeFeed(i)
    double * _first_var_der;    // _feeder->getVariationalFirstDerivativeFeed(i)
    double ** _cross_first_der;    // _feeder->getCrossFirstDerivativeFeed(i, j)
    double ** _cross_second_der; //  _feeder->getCrossSecondDerivativeFeed(i, j)

    // variational derivatives
    int _nvp;
    double * _v1vd;    // variational first derivatives

    // cross derivatives d/dx d/dbeta
    double ** _v1d1vd;   // first index input derivative, second index variational parameter
    double ** _v2d1vd;   // first index input derivative, second index variational parameter

public:
    // Constructor and destructor
    NetworkUnit();
    virtual ~NetworkUnit();

    // return the ideal mean value (mu) and standard deviation (sigma) of the proto value (pv)
    // (if the derived unit applies e.g. an activation function to the pv, overwrite this accordingly)
    virtual double getIdealProtoMu(){return 0;}
    virtual double getIdealProtoSigma(){return 1.;}

    // return the final output mu and sigma
    // (here pretending a constant pv input)
    virtual double getOutputMu(){return _pv;}
    virtual double getOutputSigma(){return 0;}

    // BaseComponent IdCodes
    std::string getClassIdCode(){return "UNIT";}
    virtual std::string getIdCode() = 0; // virtual class

    // Setters
    void setProtoValue(const double &pv){_pv=pv;}

    // Getters
    double getValue(){return _v;}
    double getProtoValue(){return _pv;}

    // Coordinate derivatives
    void setFirstDerivativeSubstrate(const int &nx0);
    void setFirstDerivativeValue(const int &i1d, const double &v1d){_v1d[i1d]=v1d;}
    double getFirstDerivativeValue(const int &i1d){return _v1d[i1d];}  // return first derivative value
    void setSecondDerivativeSubstrate(const int &nx0);
    void setSecondDerivativeValue(const int &i2d, const double &v2d){_v2d[i2d]=v2d;}
    double getSecondDerivativeValue(const int &i2d){return _v2d[i2d];}  // return second derivative value

    // Variational derivatives
    void setVariationalFirstDerivativeSubstrate(const int &nvp);
    void setVariationalFirstDerivativeValue(const int &i1vd, const double &v1vd){_v1vd[i1vd]=v1vd;}
    double getVariationalFirstDerivativeValue(const int &i1vd){return _v1vd[i1vd];}  // return first derivative value

    // Cross derivatives
    void setCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp);
    void setCrossFirstDerivative(const int &i1d, const int &i1vd, const double &v1d1vd){_v1d1vd[i1d][i1vd]=v1d1vd;}
    double getCrossFirstDerivativeValue(const int &i1d, const int &i1vd){return _v1d1vd[i1d][i1vd];}
    void setCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp);
    void setCrossSecondDerivative(const int &i2d, const int &i1vd, const double &v2d1vd){_v2d1vd[i2d][i1vd]=v2d1vd;}
    double getCrossSecondDerivativeValue(const int &i2d, const int &i1vd){return _v2d1vd[i2d][i1vd];}

    // Computation, may be changed by child
    virtual void computeFeed() {};
    virtual void computeOutput(); // _pv->_v, defaults to identity
    virtual void computeDerivatives() {};

    // should execute the methods above (default implementation), but may be extended
    virtual void computeValues();
};


#endif
