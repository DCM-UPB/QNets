#ifndef NETWORK_LAYER_INTERFACE
#define NETWORK_LAYER_INTERFACE

#include "NetworkUnit.hpp"
#include "OffsetUnit.hpp"

#include <vector>

class NetworkLayerInterface
{
protected:

    OffsetUnit * _U_off;
    std::vector<NetworkUnit *> _U; // this vector stores units of all derived types

public:

    // --- Constructor

    NetworkLayerInterface()
    {
        _U_off = new OffsetUnit();
        _U.push_back(_U_off);
    }

    virtual void construct(const int &nunits) = 0; // should add non-offset units until a total of nunits units of any type are present


    // --- Destructor

    virtual ~NetworkLayerInterface()
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
            delete _U[i];
        }
        _U.clear();
    }

    virtual void deconstruct() // should remove the non-offset units
    {
        for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
            {
                delete _U[i];
            }
        _U.clear();
        _U.push_back(_U_off);
    }


    // --- Getters

    int getNUnits(){return _U.size();}
    NetworkUnit * getUnit(const int & i){return _U[i];}
    OffsetUnit * getOffsetUnit(){return _U_off;}


    // --- Modify structure

    void setSize(const int &nunits) // used deconstruct and then construct to resize the layer
    {
        this->deconstruct();
        this->construct(nunits);
    }


    // --- Variational Parameters

    virtual bool setVariationalParameter(const int &id, const double &vp) {return false;}; 
    virtual bool getVariationalParameter(const int &id, double &vp) {return false;}
    virtual int getNVariationalParameters() {return 0;};


    // --- Values to compute

    virtual int setVariationalParametersID(const int &id_vp) { return -1;};


    void addCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
            _U[i]->setCrossSecondDerivativeSubstrate(nx0, nvp);
        }
    }


    void addCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
            _U[i]->setCrossFirstDerivativeSubstrate(nx0, nvp);
        }
    }


    void addVariationalFirstDerivativeSubstrate(const int &nvp)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
            _U[i]->setVariationalFirstDerivativeSubstrate(nvp);
        }
    }


    void addSecondDerivativeSubstrate(const int &nx0)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
            _U[i]->setSecondDerivativeSubstrate(nx0);
        }
    }


    void addFirstDerivativeSubstrate(const int &nx0)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i){
            _U[i]->setFirstDerivativeSubstrate(nx0);
        }
    }


    // --- Computation

    virtual void computeValues()
    {
#ifdef OPENMP
#pragma omp for schedule(static, 1)
#endif
        for (NetworkUnit * u : _U) u->computeValues();
    }
};

#endif
