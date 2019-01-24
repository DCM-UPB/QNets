#ifndef INPUT_UNIT
#define INPUT_UNIT

#include "ShifterScalerUnit.hpp"
#include <string>

// Input Unit
class InputUnit: public ShifterScalerUnit
{
protected:
    int _index;
    double _inputMu, _inputSigma;

public:
    explicit InputUnit(const int &index, const double inputMu = 0., const double inputSigma = 1.): _index(index) {_inputMu = inputMu; _inputSigma = inputSigma;} // the index of the input unit, i.e. d/dx_index f(_pv) = 1
    virtual ~InputUnit(){}

    // string code methods
    virtual std::string getIdCode(){return "IN";} // return identifier for unit type

    // return the output mean value (mu) and standard deviation (sigma)
    virtual double getOutputMu(){return (_inputMu + _shift ) * _scale;}
    virtual double getOutputSigma(){return _inputSigma * _scale;}

    // set input data mu and sigma, set shift/scale accordingly
    void setInputMu(const double &inputMu, const bool &doShift = true);
    void setInputSigma(const double &inputSigma, const bool &doScale = true);

    // get the input data mu and sigma
    double getInputMu(){return _inputMu;}
    double getInputSigma(){return _inputSigma;}

    // set/get the dedicated input index
    void setInputIndex(int &index){_index=index;}
    int getInputIndex(){return _index;}

    // Computation
    void computeFeed(){}
    void computeActivation(){}
    void computeDerivatives(){}

    void computeValues() {_v = _pv; if (_v1d) _v1d[_index] = 1.;}
};


#endif
