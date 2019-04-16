#ifndef FFNN_NET_TEMPLNET_HPP
#define FFNN_NET_TEMPLNET_HPP

#include <array>
#include <vector>


template <class IteratorT = double *>
class LogACTF // Test Array ACTF
{
public:

    static void f(IteratorT cur, IteratorT end, IteratorT out)
    {
        for (; cur<end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur)));
        }
    }

    static void fd1(IteratorT cur, IteratorT end, IteratorT out)
    {
        for (; cur<end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur))); // f
            *out = *out * (1. - *out); // fd1
        }
    }

    static void fd2(IteratorT cur, IteratorT end, IteratorT out)
    {
        for (; cur<end; ++cur, ++out) {
            *out = 1./(1. + exp(-(*cur))); // f
            *out = *out * (1. - *out) * (1. - 2.*(*out)); // fd2
        }
    }

    static void fad(IteratorT cur, IteratorT end, IteratorT outf, IteratorT outfd1, IteratorT outfd2)
    {
        for (; cur<end; ++cur, ++outf, ++outfd1, ++outfd2) {
            *outf = 1./(1. + exp(-(*cur))); // f
            *outfd1 = *outf * (1. - *outf); // fd1
            *outfd2 = *outfd1 * (1. - 2.*(*outf)); // fd2
        }
    }
};


template<typename T, T ... ts>
constexpr T const_sum() {
    T result = 0;
    for(auto &t : { ts... }) result += t;
    return result;
}


template<typename T, T ... ts>
constexpr T const_prod() {
    T result = 1;
    for(auto &t : { ts... }) result *= t;
    return result;
}

/*template <class SizeT = size_t, SizeT N_IN, SizeT N_OUT, std::initializer_list<SizeT> N_HID>
constexpr calcNVP*/

template <typename SizeT, typename ValueT,
          class HiddenACTF, class OutputACTF,
                  SizeT NU_IN, SizeT NU_OUT, SizeT NL_HID, SizeT ... NUs_HID>
class TemplNet
{
private:
    bool _flag_d1{}, _flag_d2{}, _flag_vd1{};  // flags that indicate if the substrates for the derivatives have been activated or not

    static constexpr SizeT _nvp = 0;  // global number of variational parameters

public:
    static constexpr SizeT NLAYERS = 2 + sizeof...(NUs_HID);
    static constexpr SizeT NUNITS = NU_IN + const_sum<SizeT, NUs_HID...>() + NU_OUT;


    constexpr TemplNet() = default;
    constexpr TemplNet(bool flag_d1, bool flag_d2, bool flag_vd1):
    _flag_d1(flag_d1), _flag_d2(flag_d2), _flag_vd1(flag_vd1) {}


    // --- Get information about the NN structure
    /*
    SizeT getNLayers() { return _L.size(); }
    SizeT getNFedLayers() { return _L_fed.size(); }
    SizeT getNNeuralLayers() { return _L_nn.size(); }
    SizeT getNHiddenLayers() { return _L_nn.size() - 1; }

    SizeT getNInput() { return _L_in->getNInputUnits(); }
    SizeT getNOutput() { return _L_out->getNOutputNNUnits(); }
    SizeT getLayerSize(const SizeT &li) { return _L[li]->getNUnits(); }
    
    bool hasFirstDerivative() { return _flag_d1; }
    bool hasSecondDerivative() { return _flag_d2; }
    bool hasVariationalFirstDerivative() { return _flag_vd1; }
    */

    // --- Manage the betas, which exist only after that the FFNN has been connected
    //SizeT getNBeta();
    /*ValueT getBeta(const SizeT &ib);
    void getBeta(ValueT * beta);
    void setBeta(const SizeT &ib, const ValueT &beta);
    void setBeta(const ValueT * beta);
    void randomizeBetas(); // has to be changed maybe if we add beta that are not "normal" weights*/

    // --- Manage the variational parameters (which may contain a subset of beta and/or non-beta parameters),
    //     which exist only after that they are assigned to actual parameters in the network (e.g. betas)
    //SizeT getNVariationalParameters() { return _nvp; }
    /*ValueT getVariationalParameter(const SizeT &ivp);
    void getVariationalParameter(ValueT * vp);
    void setVariationalParameter(const SizeT &ivp, const ValueT &vp);
    void setVariationalParameter(const ValueT * vp);*/

    // --- Toggle the calculations of derivatives
    void enableFirstDerivatives() { _flag_d1 = true; }  // coordinates first derivatives
    void disableFirstDerivatives() { _flag_d1 = false; }
    void enableSecondDerivatives() { _flag_d2 = true; }  // coordinates second derivatives
    void disableSecondDerivatives() { _flag_d2 = false; }
    void enableVariationalFirstDerivatives() { _flag_vd1 = true; }  // variational first derivatives
    void disableVariationalFirstDerivatives() { _flag_vd1 = false; }

    // shortcut for (connecting and) adding substrates
    //void enableDerivatives(bool flag_d1, bool flag_d2, bool flag_vd1);


    // Set initial parameters
    /*void setInput(const ValueT * in);
    void setInput(const SizeT &i, const ValueT &in);*/

    // --- Computation
    //void FFPropagate();

    // Shortcut for computation: set input and get all values and derivatives with one calculations.
    // If some derivatives are not supported (substrate missing) the values will be leaved unchanged.
    //void evaluate(const ValueT * in, ValueT * out, ValueT * d1 = nullptr, ValueT * d2 = nullptr, ValueT * vd1 = nullptr);


    // --- Get outputs
    /*
    void getOutput(ValueT * out);
    ValueT getOutput(const SizeT &i);

    void getFirstDerivative(ValueT ** d1);
    void getFirstDerivative(const SizeT &iu, ValueT * d1);  // iu is the unit index
    ValueT getFirstDerivative(const SizeT &iu, const SizeT &i1d); // i is the index of the output elemnet (i.e. unit=1, offset unit is meaningless), i1d the index of the input element

    void getSecondDerivative(ValueT ** d2);
    void getSecondDerivative(const SizeT &i, ValueT * d2);  // i is the output index
    ValueT getSecondDerivative(const SizeT &i, const SizeT &i2d); // i is the index of the output element, i2d the index of the input element

    void getVariationalFirstDerivative(ValueT ** vd1);
    void getVariationalFirstDerivative(const SizeT &i, ValueT * vd1);  // i is the output index
    ValueT getVariationalFirstDerivative(const SizeT &i, const SizeT &iv1d);  // i is the index of the output element, iv1d the index of the beta element
    */

    // --- Store FFNN on file
    //void storeOnFile(const char * filename, bool store_betas = true);
};


#endif
