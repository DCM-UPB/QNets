#include "qnets/poly/FeedForwardNeuralNetwork.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>


// --- Beta

int FeedForwardNeuralNetwork::getNBeta() const
{
    using namespace std;
    int nbeta = 0;
    for (auto &i : _L_fed) {
        for (int j = 0; j < i->getNFedUnits(); ++j) {
            if (i->getFedUnit(j)->getFeeder() != nullptr) {
                nbeta += i->getFedUnit(j)->getFeeder()->getNBeta();
            }
        }
    }
    return nbeta;
}


double FeedForwardNeuralNetwork::getBeta(const int &ib) const
{
    using namespace std;
    if (ib < 0 || ib >= getNBeta()) {
        cout << endl << "ERROR FeedForwardNeuralNetwork::getBeta : index out of boundaries" << endl;
        cout << ib << " against the maximum allowed " << this->getNBeta() - 1 << endl << endl;
    }
    else {
        int idx = 0;
        for (auto &i : _L_fed) {
            for (int j = 0; j < i->getNFedUnits(); ++j) {
                if (i->getFedUnit(j)->getFeeder() != nullptr) {
                    for (int k = 0; k < i->getFedUnit(j)->getFeeder()->getNBeta(); ++k) {
                        if (idx == ib) {
                            return i->getFedUnit(j)->getFeeder()->getBeta(k);
                        }
                        idx++;
                    }
                }
            }
        }
    }
    cout << endl << "ERROR FeedForwardNeuralNetwork::getBeta : index " << ib << " not found" << endl << endl;
    return -666.;
}


void FeedForwardNeuralNetwork::getBeta(double * beta) const
{
    using namespace std;
    int idx = 0;
    for (auto &i : _L_fed) {
        for (int j = 0; j < i->getNFedUnits(); ++j) {
            if (i->getFedUnit(j)->getFeeder() != nullptr) {
                for (int k = 0; k < i->getFedUnit(j)->getFeeder()->getNBeta(); ++k) {
                    beta[idx] = i->getFedUnit(j)->getFeeder()->getBeta(k);
                    idx++;
                }
            }
        }
    }
}


void FeedForwardNeuralNetwork::setBeta(const int &ib, const double &beta)
{
    using namespace std;
    if (ib < 0 || ib >= this->getNBeta()) {
        cout << endl << "ERROR FeedForwardNeuralNetwork::setBeta : index out of boundaries" << endl;
        cout << ib << " against the maximum allowed " << this->getNBeta() - 1 << endl << endl;
    }
    else {
        int idx = 0;
        for (auto &i : _L_fed) {
            for (int j = 0; j < i->getNFedUnits(); ++j) {
                if (i->getFedUnit(j)->getFeeder() != nullptr) {
                    for (int k = 0; k < i->getFedUnit(j)->getFeeder()->getNBeta(); ++k) {
                        if (idx == ib) {
                            i->getFedUnit(j)->getFeeder()->setBeta(k, beta);
                            return;
                        }
                        idx++;
                    }
                }
            }
        }
    }
}


void FeedForwardNeuralNetwork::setBeta(const double * beta)
{
    using namespace std;
    int idx = 0;
    for (auto &i : _L_fed) {
        for (int j = 0; j < i->getNFedUnits(); ++j) {
            if (i->getFedUnit(j)->getFeeder() != nullptr) {
                for (int k = 0; k < i->getFedUnit(j)->getFeeder()->getNBeta(); ++k) {
                    i->getFedUnit(j)->getFeeder()->setBeta(k, beta[idx]);
                    idx++;
                }
            }
        }
    }
}


void FeedForwardNeuralNetwork::randomizeBetas()
{
    using namespace std;

    random_device rdev;
    mt19937_64 rgen = std::mt19937_64(rdev());
    uniform_real_distribution<double> rd;

    int nsource;
    double bah;

    for (auto &i : _L_fed) {
        for (int j = 0; j < i->getNFedUnits(); ++j) {
            if (i->getFedUnit(j)->getFeeder() != nullptr) {
                nsource = i->getFedUnit(j)->getFeeder()->getNBeta();
                // target sigma to keep sum of weighted inputs in range [-4,4], assuming uniform distribution
                // sigma = 8/sqrt(12) = (b-a)/sqrt(12) * n^(1/2) , where n is nsource
                bah = 4*pow(nsource, -0.5); // (b-a)/2
                rd = uniform_real_distribution<double>(-bah, bah);

                for (int k = 0; k < nsource; ++k) {
                    i->getFedUnit(j)->getFeeder()->setBeta(k, rd(rgen));
                }
            }
        }
    }
}


// --- Variational Parameters

void FeedForwardNeuralNetwork::_updateNVP()
{
    // count the total number of variational parameters
    _nvp = 0;
    for (auto &i : _L) {
        _nvp += i->getNVariationalParameters();
    }
}


void FeedForwardNeuralNetwork::assignVariationalParameters(const int &starting_layer_index)
{
    // make betas variational parameters, starting from starting_layer
    int id_vp = 0;
    for (std::vector<NetworkLayer *>::size_type i = starting_layer_index; i < _L.size(); ++i) {
        id_vp = _L[i]->setVariationalParametersID(id_vp);
    }
    _updateNVP();
}


double FeedForwardNeuralNetwork::getVariationalParameter(const int &ivp) const
{
    using namespace std;

    if (ivp < 0 || ivp >= getNVariationalParameters()) {
        cout << endl << "ERROR FeedForwardNeuralNetwork::getVariationalParameter : index out of boundaries" << endl;
        cout << ivp << " against the maximum allowed " << this->getNVariationalParameters() - 1 << endl << endl;
    }
    for (auto &i : _L) {
        if (ivp <= i->getMaxVariationalParameterIndex()) {
            double result;
            bool status = i->getVariationalParameter(ivp, result);
            if (status) {
                return result;
            }
            {
                break;
            }
        }
    }
    cout << endl << "ERROR FeedForwardNeuralNetwork::getVariationalParameter : index " << ivp << " not found" << endl << endl;
    return -666.;
}


void FeedForwardNeuralNetwork::getVariationalParameter(double * vp) const
{
    using namespace std;

    int ivp = 0;
    for (vector<NetworkLayer *>::size_type i = 0; i < _L.size(); ++i) {
        int idmax = _L[i]->getMaxVariationalParameterIndex();
        for (; ivp <= idmax; ++ivp) {
            bool status = _L[i]->getVariationalParameter(ivp, vp[ivp]);
            if (!status) {
                cout << endl << "ERROR FeedForwardNeuralNetwork::getVariationalParameter : index " << ivp << " not found in layer " << i << " with max index " << idmax << endl << endl;
            }
        }
    }
}


void FeedForwardNeuralNetwork::setVariationalParameter(const int &ivp, const double &vp)
{
    if (ivp < 0 || ivp >= getNVariationalParameters()) {
        cout << endl << "ERROR FeedForwardNeuralNetwork::setVariationalParameter : index out of boundaries" << endl << endl;
        cout << ivp << " against the maximum allowed " << this->getNVariationalParameters() - 1 << endl << endl;
    }
    for (auto &i : _L) {
        if (ivp <= i->getMaxVariationalParameterIndex()) {
            bool status = i->setVariationalParameter(ivp, vp);
            if (status) {
                return;
            }
            {
                break;
            }
        }
    }
    cout << endl << "ERROR FeedForwardNeuralNetwork::setVariationalParameter : index " << ivp << " not found" << endl << endl;
}


void FeedForwardNeuralNetwork::setVariationalParameter(const double * vp)
{
    using namespace std;

    int ivp = 0;
    for (vector<NetworkLayer *>::size_type i = 0; i < _L.size(); ++i) {
        int idmax = _L[i]->getMaxVariationalParameterIndex();
        for (; ivp <= idmax; ++ivp) {
            bool status = _L[i]->setVariationalParameter(ivp, vp[ivp]);
            if (!status) {
                cout << endl << "ERROR FeedForwardNeuralNetwork::setVariationalParameter : index " << ivp << " not found in layer " << i << " with max index " << idmax << endl << endl;
            }
        }
    }
}


// --- Computation


double FeedForwardNeuralNetwork::getCrossSecondDerivative(const int &i, const int &i1d, const int &iv1d) const
{
    return _L_out->getUnit(i + 1)->getCrossSecondDerivativeValue(i1d, iv1d);
}


void FeedForwardNeuralNetwork::getCrossSecondDerivative(double *** d2vd1) const
{
    for (int i = 0; i < getNOutput(); ++i) {
        getCrossSecondDerivative(i, d2vd1[i]);
    }
}


void FeedForwardNeuralNetwork::getCrossSecondDerivative(const int &i, double ** d2vd1) const
{
    for (int i2d = 0; i2d < getNInput(); ++i2d) {
        for (int iv1d = 0; iv1d < getNVariationalParameters(); ++iv1d) {
            d2vd1[i2d][iv1d] = getCrossSecondDerivative(i, i2d, iv1d);
        }
    }
}


void FeedForwardNeuralNetwork::getCrossSecondDerivative(const int &i, const int &i2d, double * d2vd1) const
{
    for (int iv1d = 0; iv1d < getNVariationalParameters(); ++iv1d) {
        d2vd1[iv1d] = getCrossSecondDerivative(i, i2d, iv1d);
    }
}


double FeedForwardNeuralNetwork::getCrossFirstDerivative(const int &i, const int &i1d, const int &iv1d) const
{
    return _L_out->getUnit(i + 1)->getCrossFirstDerivativeValue(i1d, iv1d);
}


void FeedForwardNeuralNetwork::getCrossFirstDerivative(double *** d1vd1) const
{
    for (int i = 0; i < getNOutput(); ++i) {
        getCrossFirstDerivative(i, d1vd1[i]);
    }
}


void FeedForwardNeuralNetwork::getCrossFirstDerivative(const int &i, double ** d1vd1) const
{
    for (int i1d = 0; i1d < getNInput(); ++i1d) {
        for (int iv1d = 0; iv1d < getNBeta(); ++iv1d) {
            d1vd1[i1d][iv1d] = getCrossFirstDerivative(i, i1d, iv1d);
        }
    }
}


void FeedForwardNeuralNetwork::getCrossFirstDerivative(const int &i, const int &i1d, double * d1vd1) const
{
    for (int iv1d = 0; iv1d < getNVariationalParameters(); ++iv1d) {
        d1vd1[iv1d] = getCrossFirstDerivative(i, i1d, iv1d);
    }
}


double FeedForwardNeuralNetwork::getVariationalFirstDerivative(const int &i, const int &iv1d) const
{
    return _L_out->getUnit(i + 1)->getVariationalFirstDerivativeValue(iv1d);
}


void FeedForwardNeuralNetwork::getVariationalFirstDerivative(const int &i, double * vd1) const
{
    for (int iv1d = 0; iv1d < getNInput(); ++iv1d) {
        vd1[iv1d] = getVariationalFirstDerivative(i, iv1d);
    }
}


void FeedForwardNeuralNetwork::getVariationalFirstDerivative(double ** vd1) const
{
    for (int i = 0; i < getNOutput(); ++i) {
        for (int iv1d = 0; iv1d < getNBeta(); ++iv1d) {
            vd1[i][iv1d] = getVariationalFirstDerivative(i, iv1d);
        }
    }
}


double FeedForwardNeuralNetwork::getSecondDerivative(const int &i, const int &i2d) const
{
    return (_L_out->getUnit(i + 1)->getSecondDerivativeValue(i2d));
}


void FeedForwardNeuralNetwork::getSecondDerivative(const int &i, double * d2) const
{
    for (int i2d = 0; i2d < getNInput(); ++i2d) {
        d2[i2d] = getSecondDerivative(i, i2d);
    }
}


void FeedForwardNeuralNetwork::getSecondDerivative(double ** d2) const
{
    for (int i = 0; i < getNOutput(); ++i) {
        for (int i2d = 0; i2d < getNInput(); ++i2d) {
            d2[i][i2d] = getSecondDerivative(i, i2d);
        }
    }
}


double FeedForwardNeuralNetwork::getFirstDerivative(const int &i, const int &i1d) const
{
    return (_L_out->getUnit(i + 1)->getFirstDerivativeValue(i1d));
}


void FeedForwardNeuralNetwork::getFirstDerivative(const int &i, double * d1) const
{
    for (int i1d = 0; i1d < getNInput(); ++i1d) {
        d1[i1d] = getFirstDerivative(i, i1d);
    }
}


void FeedForwardNeuralNetwork::getFirstDerivative(double ** d1) const
{
    for (int i = 0; i < getNOutput(); ++i) {
        for (int i1d = 0; i1d < getNInput(); ++i1d) {
            d1[i][i1d] = getFirstDerivative(i, i1d);
        }
    }
}


double FeedForwardNeuralNetwork::getOutput(const int &i) const
{
    return _L_out->getUnit(i + 1)->getValue();
}


void FeedForwardNeuralNetwork::getOutput(double * out) const
{
    for (int i = 1; i < _L_out->getNUnits(); ++i) {
        out[i - 1] = _L_out->getUnit(i)->getValue();
    }
}


void FeedForwardNeuralNetwork::evaluate(const double * in, double * out, double ** d1, double ** d2, double ** vd1)
{
    setInput(in);
    FFPropagate();
    if (out != nullptr) {
        getOutput(out);
    }
    if (hasFirstDerivativeSubstrate() && d1 != nullptr) {
        getFirstDerivative(d1);
    }
    if (hasSecondDerivativeSubstrate() && d2 != nullptr) {
        getSecondDerivative(d2);
    }
    if (hasVariationalFirstDerivativeSubstrate() && vd1 != nullptr) {
        getVariationalFirstDerivative(vd1);
    }
}

void FeedForwardNeuralNetwork::FFPropagate()
{
#ifdef OPENMP
#pragma omp parallel default(none)
#endif
    for (auto &i : _L) {
        i->computeValues();
    }
}


void FeedForwardNeuralNetwork::setInput(const double * in)
{
    // set the protovalues of the first layer units
    for (int i = 0; i < _L_in->getNInputUnits(); ++i) {
        _L_in->getInputUnit(i)->setProtoValue(in[i]);
    }
}


void FeedForwardNeuralNetwork::setInput(const int &i, const double &in)
{
    // set the protovalues of the first layer units
    _L_in->getInputUnit(i)->setProtoValue(in);
}



// --- Substrates


void FeedForwardNeuralNetwork::addCrossSecondDerivativeSubstrate()
{
    using namespace std;

    if (_flag_c2d) {
        return; // nothing to do
    }

    // add dependencies (which themselves add all other required substrates)
    if (!_flag_2d) {
        addSecondDerivativeSubstrate();
    }
    if (!_flag_c1d) {
        addCrossFirstDerivativeSubstrate();
    }

    // set the substrate in the units
    for (auto &i : _L) {
        i->addCrossSecondDerivativeSubstrate(getNInput());
    }

    _flag_c2d = true;
}


void FeedForwardNeuralNetwork::addCrossFirstDerivativeSubstrate()
{
    using namespace std;

    if (_flag_c1d) {
        return; // nothing to do
    }

    // add dependencies
    if (!_flag_1d) {
        addFirstDerivativeSubstrate();
    }
    if (!_flag_v1d) {
        addVariationalFirstDerivativeSubstrate();
    }

    // set the substrate in the units
    for (auto &i : _L) {
        i->addCrossFirstDerivativeSubstrate(getNInput());
    }

    _flag_c1d = true;
}


void FeedForwardNeuralNetwork::addVariationalFirstDerivativeSubstrate()
{
    if (_flag_v1d) {
        return; // nothing to do
    }

    // set the substrate in the units
    for (auto &i : _L) {
        i->addVariationalFirstDerivativeSubstrate();
    }

    _flag_v1d = true;
}


void FeedForwardNeuralNetwork::addSecondDerivativeSubstrate()
{
    if (_flag_2d) {
        return; // nothing to do
    }
    if (!_flag_1d) {
        addFirstDerivativeSubstrate();
    }

    // add the second derivative substrate to all the layers
    for (auto &i : _L) {
        i->addSecondDerivativeSubstrate(getNInput());
    }

    _flag_2d = true;
}


void FeedForwardNeuralNetwork::addFirstDerivativeSubstrate()
{
    if (_flag_1d) {
        return; // nothing to do
    }

    // add the first derivative substrate to all the layers
    for (auto &i : _L) {
        i->addFirstDerivativeSubstrate(getNInput());
    }

    _flag_1d = true;
}

void FeedForwardNeuralNetwork::addSubstrates(const bool flag_d1, const bool flag_d2, const bool flag_vd1, const bool flag_c1d, const bool flag_c2d)
{
    if (flag_d1) {
        addFirstDerivativeSubstrate();
    }
    if (flag_d2) {
        addSecondDerivativeSubstrate();
    }
    if (flag_vd1) {
        addVariationalFirstDerivativeSubstrate();
    }
    if (flag_c1d) {
        addCrossFirstDerivativeSubstrate();
    }
    if (flag_c2d) {
        addCrossSecondDerivativeSubstrate();
    }
}

// --- Connect the neural network

void FeedForwardNeuralNetwork::connectFFNN()
{
    if (_flag_connected) {
        this->disconnectFFNN();
    }

    _L_fed[0]->connectOnTopOfLayer(_L_in); // connect the first fed layer to the input layer
    for (std::vector<FedLayer *>::size_type i = 1; i < _L_fed.size(); ++i) // connect the rest
    {
        _L_fed[i]->connectOnTopOfLayer(_L_fed[i - 1]);
    }
    _flag_connected = true;
}


void FeedForwardNeuralNetwork::disconnectFFNN()
{
    if (!_flag_connected) {
        using namespace std;
        cout << "ERROR: FeedForwardNeuralNetwork::disconnectFFNN() : trying to disconnect an already disconnected FFNN" << endl << endl;
    }

    for (auto &i : _L_fed) {
        i->disconnect();
    }
    _flag_connected = false;
}


// --- Modify NN structure

void FeedForwardNeuralNetwork::setGlobalActivationFunctions(ActivationFunctionInterface * actf)
{
    for (auto &i : _L_nn) {
        i->setActivationFunction(actf);
    }
}


void FeedForwardNeuralNetwork::pushHiddenLayer(const int &size)
{
    if (_flag_connected) {
        using namespace std;
        // count the number of beta before the last (output) layer
        int nbeta = 0;
        for (vector<FedLayer *>::size_type i = 0; i < _L_fed.size() - 1; ++i) {
            for (int j = 0; j < _L_fed[i]->getNFedUnits(); ++j) {
                if (_L_fed[i]->getFedUnit(j)->getFeeder() != nullptr) {
                    nbeta += _L_fed[i]->getFedUnit(j)->getFeeder()->getNBeta();
                }
            }
        }
        int total_nbeta = this->getNBeta();
        // store the beta for the output
        auto * old_beta = new double[total_nbeta - nbeta];
        for (int i = nbeta; i < total_nbeta; ++i) {
            old_beta[i - nbeta] = getBeta(i);
        }

        // disconnect last layer
        _L_out->disconnect();  // disconnect the last (output) layer
        // insert new layer
        _addNewLayer("NNL", size, 1);
        // reconnect the layers
        _L_nn[_L_nn.size() - 2]->connectOnTopOfLayer(_L[_L.size() - 3]);
        _L_nn[_L_nn.size() - 1]->connectOnTopOfLayer(_L[_L.size() - 2]);

        // restore the old beta
        for (int i = nbeta; i < total_nbeta; ++i) {
            this->setBeta(i, old_beta[i - nbeta]);
        }
        // set all the other beta to zero
        for (int i = total_nbeta; i < this->getNBeta(); ++i) {
            this->setBeta(i, 0.);
        }
        for (int i = 0; i < _L_fed[_L_fed.size() - 1]->getNFedUnits(); ++i) {
            _L_fed[_L_fed.size() - 1]->getFedUnit(i)->getFeeder()->setBeta(i, 1.);
        }
        // free memory
        delete[] old_beta;
    }
    else {
        _addNewLayer("NNL", size, 1);
    }
}


void FeedForwardNeuralNetwork::popHiddenLayer()
{
    delete _L[_L.size() - 2];

    auto it = _L.end() - 2;
    auto it_fed = _L_fed.end() - 2;
    auto it_nn = _L_nn.end() - 2;

    _L.erase(it);
    _L_fed.erase(it_fed);
    _L_nn.erase(it_nn);
}


void FeedForwardNeuralNetwork::pushFeatureMapLayer(const int &size, const std::string &params)
{
    if (_flag_connected) {
        using namespace std;
        // count the number of beta up to and including the last feature map layer
        int nbeta = 0;
        for (auto &i : _L_fm) {
            for (int j = 0; j < i->getNFedUnits(); ++j) {
                if (i->getFedUnit(j)->getFeeder() != nullptr) {
                    nbeta += i->getFedUnit(j)->getFeeder()->getNBeta();
                }
            }
        }
        int total_nbeta = this->getNBeta();
        // store the beta for the output
        auto * old_beta = new double[total_nbeta - nbeta];
        for (int i = nbeta; i < total_nbeta; ++i) {
            old_beta[i - nbeta] = getBeta(i);
        }

        // disconnect the layer after the last feature map layer
        _L_fed[_L_fm.size()]->disconnect();  // disconnect the first non-fm fed layer
        // insert new layer
        _addNewLayer("FML", size, _L_nn.size(), params);
        // reconnect the layers
        _L_fm[_L_fm.size() - 1]->connectOnTopOfLayer(_L[_L_fm.size() - 1]);
        _L_fed[_L_fm.size()]->connectOnTopOfLayer(_L_fm[_L_fm.size() - 1]);

        // restore the old beta
        for (int i = nbeta; i < total_nbeta; ++i) {
            this->setBeta(i, old_beta[i - nbeta]);
        }
        // set all the other beta to zero
        for (int i = total_nbeta; i < this->getNBeta(); ++i) {
            this->setBeta(i, 0.);
        }
        for (int i = 0; i < _L_fed[_L_fed.size() - 1]->getNFedUnits(); ++i) {
            _L_fed[_L_fed.size() - 1]->getFedUnit(i)->getFeeder()->setBeta(i, 1.);
        }
        // free memory
        delete[] old_beta;
    }
    else {
        _addNewLayer("FML", size, _L_nn.size(), params);
    }
}


// --- Store FFNN on a file

void FeedForwardNeuralNetwork::storeOnFile(const char * filename, const bool store_betas) const
{
    using namespace std;

    // open file
    ofstream file;
    file.open(filename);
    // store the number of layers
    file << getNLayers() << endl;
    // store the tree code of each layer
    for (int i = 0; i < getNLayers(); ++i) {
        string word, treeCode = _L[i]->getTreeCode();

        if (!store_betas) { // cut out betas
            bool skip = false;
            istringstream iss(treeCode);
            treeCode = ""; // reset
            while (iss >> word) {
                if (!skip) {
                    if (!treeCode.empty()) {
                        treeCode += " ";
                    }
                    treeCode += word;
                }
                if (word == "RAY") {
                    skip = true;
                }
                if (word == ")") {
                    skip = false;
                }
            }
        }

        file << treeCode << endl; // store treeCode
    }
    // store connected flag
    file << _flag_connected << endl;
    // store the flags for the substrates
    file << _flag_1d << " " << _flag_2d << " " << _flag_v1d << " " << _flag_c1d << " " << _flag_c2d << endl;
    file.close();
}


// --- Constructor

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(const char * filename)
{
    // open file
    using namespace std;

    string line, id, params;
    vector<string> layerMemberCodes;
    ifstream file;
    file.open(filename);

    // read the number of layers
    int nlayers;
    file >> nlayers;

    int il = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue; // idk why I get an empty line here in the first iteration
        }
        id = readIdCode(line);
        params = readParams(line);
        layerMemberCodes.push_back(readMemberTreeCode(line));
        _addNewLayer(id, params);

        ++il;
        if (il == nlayers) {
            break;
        }
    }
    if (il != nlayers) {
        throw std::invalid_argument("Stored FFNN file declares to have more layers than it has layer codes.");
    }

    // connect the NN, if connected is found true
    bool connected;
    file >> connected;
    if (connected) {
        connectFFNN();
    }

    // set betas and all other params/actf
    for (int i = 0; i < getNLayers(); ++i) {
        getLayer(i)->setMemberParams(layerMemberCodes[i]);
    }
    _updateNVP();

    // read and set the substrates
    bool flag_1d = false, flag_2d = false, flag_v1d = false, flag_c1d = false, flag_c2d = false;
    file >> flag_1d;
    if (flag_1d) {
        addFirstDerivativeSubstrate();
    }
    file >> flag_2d;
    if (flag_2d) {
        addSecondDerivativeSubstrate();
    }
    file >> flag_v1d;
    if (flag_v1d) {
        addVariationalFirstDerivativeSubstrate();
    }
    file >> flag_c1d;
    if (flag_c1d) {
        addCrossFirstDerivativeSubstrate();
    }
    file >> flag_c2d;
    if (flag_c2d) {
        addCrossSecondDerivativeSubstrate();
    }

    file.close();
}


FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(const FeedForwardNeuralNetwork &ffnn)
{
    auto &other = const_cast<FeedForwardNeuralNetwork &>(ffnn); // lazy hack for const signature

    // copy layer structure
    for (int i = 0; i < other.getNLayers(); ++i) {
        _addNewLayer(other.getLayer(i)->getIdCode(), other.getLayer(i)->getParams());
    }

    // read and set the substrates
    if (other.isConnected()) {
        connectFFNN();
    }

    // now copy the parameter tree (incl. betas) for all layers
    for (int i = 0; i < other.getNLayers(); ++i) {
        _L[i]->setMemberParams(other.getLayer(i)->getMemberTreeCode());
    }
    _updateNVP();

    if (other.hasFirstDerivativeSubstrate()) {
        addFirstDerivativeSubstrate();
    }
    if (other.hasSecondDerivativeSubstrate()) {
        addSecondDerivativeSubstrate();
    }
    if (other.hasVariationalFirstDerivativeSubstrate()) {
        addVariationalFirstDerivativeSubstrate();
    }
    if (other.hasCrossFirstDerivativeSubstrate()) {
        addCrossFirstDerivativeSubstrate();
    }
    if (other.hasCrossSecondDerivativeSubstrate()) {
        addCrossSecondDerivativeSubstrate();
    }
}


FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(const int &insize, const int &hidlaysize, const int &outsize)
{
    _construct(insize, hidlaysize, outsize);
}

void FeedForwardNeuralNetwork::_construct(const int &insize, const int &hidlaysize, const int &outsize)
{
    _addNewLayer("INL", insize);
    _addNewLayer("NNL", hidlaysize);
    _addNewLayer("OUTL", outsize);
}


// --- Register/Create Layer

void FeedForwardNeuralNetwork::_registerLayer(NetworkLayer * newLayer, const int &indexFromBack)
{
    _L.insert(_L.end() - indexFromBack, newLayer);

    if (auto * fnl = dynamic_cast<FedLayer *>(newLayer)) {
        _L_fed.insert(_L_fed.end() - indexFromBack, fnl);
    }

    if (auto * nnl = dynamic_cast<NNLayer *>(newLayer)) {
        _L_nn.insert(_L_nn.end() - indexFromBack, nnl);
    }

    if (auto * fml = dynamic_cast<FeatureMapLayer *>(newLayer)) {
        if (indexFromBack > static_cast<int>(_L_nn.size())) {
            _L_fm.insert(_L_fm.end() - (indexFromBack - _L_nn.size()), fml);
        }
        else {
            _L_fm.insert(_L_fm.end(), fml);
        }
    }
}


void FeedForwardNeuralNetwork::_addNewLayer(const std::string &idCode, const int &nunits, const int &indexFromBack, const std::string &params)
{
    if (idCode == "INL") {
        if (_L_in != nullptr) {
            delete _L_in;
            _L.erase(_L.begin());
        }
        _L_in = new InputLayer(nunits);
        _registerLayer(_L_in, indexFromBack);
    }
    else if (idCode == "NNL") {
        NNLayer * nnl = new NNLayer(nunits);
        _registerLayer(nnl, indexFromBack);
    }
    else if (idCode == "OUTL") {
        if (_L_out != nullptr) {
            delete _L_out;
            _L.erase(_L.end() - 1);
            _L_fed.erase(_L_fed.end() - 1);
            _L_nn.erase(_L_nn.end() - 1);
        }
        _L_out = new OutputNNLayer(nunits);
        _registerLayer(_L_out, indexFromBack);
    }
    else if (idCode == "FML") {
        auto * fml = new FeatureMapLayer(nunits);
        _registerLayer(fml, indexFromBack);
        if (!params.empty()) {
            fml->setParams(params);
        }
    }
    else {
        throw std::invalid_argument("FFNN::_addNewLayer: Unknown layer identifier '" + idCode + "' passed.");
    }
}


void FeedForwardNeuralNetwork::_addNewLayer(const std::string &idCode, const std::string &params, const int &indexFromBack)
{
    int nunits;
    if (!setParamValue(params, "nunits", nunits)) {
        nunits = 1; // if params has no information about nunits
    }
    if (countNParams(params) > 1) {
        _addNewLayer(idCode, nunits, indexFromBack, params);
    }
    else {
        _addNewLayer(idCode, nunits, indexFromBack, "");
    }
}


// --- Destructor

FeedForwardNeuralNetwork::~FeedForwardNeuralNetwork()
{
    for (auto &i : _L) {
        delete i;
    }

    _L.clear();
    _L_fed.clear();
    _L_nn.clear();
    _L_fm.clear();
    _L_in = nullptr;
    _L_out = nullptr;

    _flag_connected = false;
    _flag_1d = false;
    _flag_2d = false;
    _flag_v1d = false;
    _flag_c1d = false;
    _flag_c2d = false;

    _nvp = 0;
}
