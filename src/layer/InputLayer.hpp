#ifndef INPUT_LAYER
#define INPUT_LAYER

#include "NetworkLayer.hpp"
#include "InputUnit.hpp"

#include <vector>
#include <string>

class InputLayer: public NetworkLayer
{
protected:
    std::vector<InputUnit *> _U_in; // stores pointers to all input units
    bool _doSort; // should we sort the inputs?
    
    void _registerUnit(NetworkUnit * newUnit); // check if newUnit is a/derived from InputUnit and register

public:
    // --- Constructor / Destructor

    explicit InputLayer(const int &nunits = 1, const bool doSort = false): _doSort(doSort) { if (nunits > 1) construct(nunits); }
    void construct(const int &nunits);

    // --- Destructor

    ~InputLayer(){_U_in.clear();}
    void deconstruct(){NetworkLayer::deconstruct(); _U_in.clear();}

    // --- String Codes

    std::string getParams();
    void setParams(const std::string &params);
    std::string getIdCode(){return "INL";}

    // --- Getters

    int getNInputUnits() {return _U_in.size();}
    InputUnit * getInputUnit(const int &i) {return _U_in[i];}

    // sorting config
    bool doesSorting(){return _doSort;}
    void enableSorting(){_doSort = true;}
    void disableSorting(){_doSort = false;}

    // computeValues with sorting
    void computeValues();
};

#endif
