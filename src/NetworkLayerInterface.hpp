#ifndef NETWORK_LAYER_INTERFACE
#define NETWORK_LAYER_INTERFACE

#include "NetworkUnit.hpp"
#include "NetworkUnitRay.hpp"

#include "NNLayer.hpp"
#include "NNUnit.hpp"

#include <vector>
#include <memory>

template<typename UnitType>
class NetworkLayerInterface
{
protected:
    std::vector<UnitType *> _U; // T is expected to be at least NetworkUnit

public:
    /* Pseudo header
    virtual ~NetworkLayerInterface();

    // --- Getters
    int getNUnits(){return _U.size();}
    T * getUnit(const int & i){return _U[i];}

    // --- Modify structure
    virtual void setSize(const int &nunits) = 0;

    // --- Values to compute
    void addFirstDerivativeSubstrate(const int &nx0);
    // add the first derivative substrate to all units
    // nx0 is the number of units used as input, i.e.
    // the number of derivatives that will be computed

    void addSecondDerivativeSubstrate(const int &nx0);
    // add the second derivative substrate to all units
    // nx0 is the number of units used as input, i.e.
    // the number of derivatives that will be computed

    void addVariationalFirstDerivativeSubstrate(const int &nvp);
    // add the variational first derivative substrate to all units.
    // nvp is the number of variational parameters in the NN

    void addCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp);
    // add the cross first derivative substrate to all units.
    // nx0 is the number of units used as input
    // nvp is the number of variational parameters in the NN

    void addCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp);
    // add the cross second derivative substrate to all units.
    // nx0 is the number of units used as input
    // nvp is the number of variational parameters in the NN

    int setVariationalParametersID(const int &id_vp);   // assign the id of the variational parameters to the feeders

    // --- Connection
    template <class U, class F>
    void connectOnTopOfLayer(NetworkLayerInterface<U> * nl);
    virtual void disconnect();

    // --- Computation
    void computeValues();

    // --- Variational Parameters
    int getNVariationalParameters();
    bool getVariationalParameter(const int &id, double &vp);
    bool setVariationalParameter(const int &id, const double &vp);

    */

    // Due to template usage the implementation code must be inside the header (or not all required versions will be compiled when including)

    // --- Destructor

    virtual ~NetworkLayerInterface()
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
            {
                delete _U[i];
            }
        _U.clear();
    }

    // --- Getters

    int getNUnits(){return _U.size();}
    UnitType * getUnit(const int & i){return _U[i];}


    // --- Modify structure

    virtual void setSize(const int &nunits) = 0;


    // --- Variational Parameters

    bool setVariationalParameter(const int &id, const double &vp)
    {
        std::vector<NetworkUnit *>::size_type i=1;
        NetworkUnitFeederInterface * feeder;
        bool flag = false;
        while ( (!flag) && (i<_U.size()) )
            {
                feeder = _U[i]->getFeeder();
                if (feeder)
                    {
                        flag = feeder->setVariationalParameterValue(id,vp);
                    }
                i++;
            }
        return flag;
    }


    bool getVariationalParameter(const int &id, double &vp)
    {
        std::vector<NetworkUnit *>::size_type i=1;
        NetworkUnitFeederInterface * feeder;
        bool flag = false;
        while ( (!flag) && (i<_U.size()) )
            {
                feeder = _U[i]->getFeeder();
                if (feeder)
                    {
                        flag = feeder->getVariationalParameterValue(id, vp);
                    }
                i++;
            }
        return flag;
    }


    int getNVariationalParameters()
    {
        int nvp=0;
        NetworkUnitFeederInterface * feeder;
        for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
            {
                feeder = _U[i]->getFeeder();
                if (feeder)
                    {
                        nvp += feeder->getNVariationalParameters();
                    }
            }
        return nvp;
    }


    // --- Values to compute

    int setVariationalParametersID(const int &id_vp)
    {
        int id = id_vp;
        for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
            {
                id = _U[i]->getFeeder()->setVariationalParametersIndexes(id);
            }
        return id;
    }


    void addCrossSecondDerivativeSubstrate(const int &nx0, const int &nvp)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
            {
                _U[i]->setCrossSecondDerivativeSubstrate(nx0, nvp);
            }
    }


    void addCrossFirstDerivativeSubstrate(const int &nx0, const int &nvp)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
            {
                _U[i]->setCrossFirstDerivativeSubstrate(nx0, nvp);
            }
    }


    void addVariationalFirstDerivativeSubstrate(const int &nvp)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
            {
                _U[i]->setVariationalFirstDerivativeSubstrate(nvp);
            }
    }


    void addSecondDerivativeSubstrate(const int &nx0)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
            {
                _U[i]->setSecondDerivativeSubstrate(nx0);
            }
    }


    void addFirstDerivativeSubstrate(const int &nx0)
    {
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i)
            {
                _U[i]->setFirstDerivativeSubstrate(nx0);
            }
    }


    // --- Computation

    void computeValues()
    {
        for (UnitType * u : _U) u->computeValues();
    }


    // --- Connection

    template <typename LayerUnitType>
    void connectOnTopOfLayer(NetworkLayerInterface<LayerUnitType> * nl)
    {
        NetworkUnitFeederInterface * ray;
        for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
            {
                ray = new NetworkUnitRay<LayerUnitType>(nl);
                _U[i]->setFeeder(ray);
            }
    }

    void disconnect()
    {
        NetworkUnitFeederInterface * ray;
        for (std::vector<NetworkUnit *>::size_type i=1; i<_U.size(); ++i)
            {
                ray = _U[i]->getFeeder();
                delete ray;
                _U[i]->setFeeder(NULL);
            }
    }
};


#endif
