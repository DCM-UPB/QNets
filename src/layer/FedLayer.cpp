#include "qnets/layer/FedLayer.hpp"


// --- Register Unit

void FedLayer::_registerUnit(NetworkUnit * newUnit)
{
    NetworkLayer::_registerUnit(newUnit);
    if (auto * fu = dynamic_cast<FedUnit *>(newUnit)) {
        _U_fed.push_back(fu);
    }
}


// --- Variational Parameters

bool FedLayer::setVariationalParameter(const int &id, const double &vp)
{
    std::vector<FedUnit *>::size_type i = 0;
    bool flag = false;
    while ((!flag) && (i < _U_fed.size())) {
        FeederInterface * feeder = _U_fed[i]->getFeeder();
        if (feeder != nullptr) {
            flag = feeder->setVariationalParameterValue(id, vp);
        }
        i++;
    }
    return flag;
}


bool FedLayer::getVariationalParameter(const int &id, double &vp)
{
    std::vector<FedUnit *>::size_type i = 0;
    bool flag = false;
    while ((!flag) && (i < _U_fed.size())) {
        FeederInterface * feeder = _U_fed[i]->getFeeder();
        if (feeder != nullptr) {
            flag = feeder->getVariationalParameterValue(id, vp);
        }
        i++;
    }
    return flag;
}


int FedLayer::getNVariationalParameters()
{
    int nvp = 0;
    for (auto &i : _U_fed) {
        FeederInterface * feeder = i->getFeeder();
        if (feeder != nullptr) {
            nvp += feeder->getNVariationalParameters();
        }
    }
    return nvp;
}

int FedLayer::getMaxVariationalParameterIndex()
{
    int max_index = -1;
    for (auto &i : _U_fed) {
        FeederInterface * feeder = i->getFeeder();
        if (feeder != nullptr) {
            int index = feeder->getMaxVariationalParameterIndex();
            if (index > max_index) {
                max_index = index;
            }
        }
    }
    return max_index;
}


// --- Values to compute

int FedLayer::setVariationalParametersID(const int &id_vp)
{
    int id = id_vp;
    for (auto &i : _U_fed) {
        FeederInterface * feeder = i->getFeeder();
        if (feeder != nullptr) {
            id = i->getFeeder()->setVariationalParametersIndexes(id);
        }
    }
    return id;
}


// --- Connection

void FedLayer::connectOnTopOfLayer(NetworkLayer * nl)
{
    for (std::vector<FedUnit *>::size_type i = 0; i < _U_fed.size(); ++i) {
        FeederInterface * feeder = this->connectUnitOnTopOfLayer(nl, i); // note that i==0 means the first non-offset unit
        if (feeder != nullptr) {
            _U_fed[i]->setFeeder(feeder);
        }
    }
}

void FedLayer::disconnect()
{
    for (auto &i : _U_fed) {
        delete i->getFeeder();
        i->setFeeder(nullptr);
    }
}


// --- Compute Values (with OMP pragma)

void FedLayer::computeValues()
{
#ifdef OPENMP
    // compile with -DOPENMP -fopenmp flags to use parallelization here

    if (this->getNUnits()>2) {
#pragma omp for schedule(static, 1)
        for (std::vector<NetworkUnit *>::size_type i=0; i<_U.size(); ++i) _U[i]->computeValues();
    }
    else {
#pragma omp single
#endif

    for (auto &i : _U) {
        i->computeValues();
    }

#ifdef OPENMP
    }
#endif
}
