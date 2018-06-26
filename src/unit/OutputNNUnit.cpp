#include "OutputNNUnit.hpp"

void OutputNNUnit::setOutputBounds(const double &lbound, const double &ubound)
{
    const double mu = 0.5 * (ubound + lbound), bah = 0.5 * (ubound - lbound);
    const double idmu = _actf->getOutputMu(_actf->getIdealInputMu(), _actf->getIdealInputSigma()); // what mu would actf produce with ideal input
    const double idsig = _actf->getOutputSigma(_actf->getIdealInputMu(), _actf->getIdealInputSigma()); // what sig would actf produce with ideal input
    const double idbah = 0.5 * idsig * sqrt(12);  // assuming the actfs calculate mu/sig according to flat distribution
    _scale = (bah > 0) ? bah / idbah : 1.;
    _shift = mu / _scale - idmu;
};
