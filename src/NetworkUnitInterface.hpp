#ifndef NETWORK_UNIT_INTERFACE
#define NETWORK_UNIT_INTERFACE


// Generalized Network Unit
class NetworkUnitInterface
{
protected:
    // Unit core elements
    double _pv;    // protovalue, i.e. the raw signal value on in the input side
    double _v;    // output side value of the unit

    // coordinate derivatives
    int _nx0;    // number of first derivatives (i.e. the number of inputs of the NN)
    double * _v1d;    // first derivatives
    double * _v2d;    // second derivative

    // internal variables, for numerical optimization. They store some values during the computeValues() method
    double * _first_der;    // _feeder->getFirstDerivativeFeed(i)
    double * _second_der;    // _feeder->getSecondDerivativeFeed(i)
    double * _first_var_der;    // _feeder->getVariationalFirstDerivativeFeed(i)
    double ** _cross_first_der;    // _feeder->getCrossFirstDerivativeFeed(i, j)

    // variational derivatives
    int _nvp;
    double * _v1vd;    // variational first derivatives

    // cross derivatives d/dx d/dbeta
    double ** _v1d1vd;   // first index input derivative, second index variational parameter
    double ** _v2d1vd;   // first index input derivative, second index variational parameter

public:
    // Constructor and destructor
    NetworkUnitInterface();
    virtual ~NetworkUnitInterface();

    // Setters
    void setProtoValue(const double &pv){_pv=pv;}

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

    // Getters
    double getValue(){return _v;}
    double getProtoValue(){return _pv;}

    // Computation, must be implemented by child
    virtual void computeValues() = 0;
};


#endif
