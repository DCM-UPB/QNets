#include "OutputNNLayer.hpp"

#include "ActivationFunctionInterface.hpp"
#include "ActivationFunctionManager.hpp"
#include "OutputNNUnit.hpp"

// --- Constructor

void OutputNNLayer::construct(const int &nunits, ActivationFunctionInterface * actf)
{
    for (int i=1; i<nunits; ++i)
        {
            OutputNNUnit * newUnit = new OutputNNUnit(actf->getCopy());
            _U.push_back(newUnit);
            _U_fed.push_back(newUnit);
            _U_nn.push_back(newUnit);
            _U_out.push_back(newUnit);
        }
    delete actf;
}
