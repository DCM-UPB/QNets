#ifndef FFNN_NET_TEMPLNET_HPP
#define FFNN_NET_TEMPLNET_HPP

#include "qnets/tool/PackTools.hpp"
#include "qnets/tool/TupleTools.hpp"
#include "qnets/templ/Layer.hpp"
#include "qnets/templ/LayerPackTools.hpp"
#include "qnets/templ/DerivConfig.hpp"

#include <array>
#include <tuple>
#include <algorithm>
#include <utility>
#include <type_traits>

namespace templ
{

// NOTE: For technical reasons, we need to use a TemplNetShape class for static creation
//       of arrays like nbeta_shape (which depends on Layer, not LayerConf). Might become obsolete with C++17.
namespace detail
{
template <class LTuplType, class ISeq>
struct TemplNetShape;

template <class LTuplType, int ... Is>
struct TemplNetShape<LTuplType, std::integer_sequence<int, Is...>>
{
private:
    using LayerTuple = LTuplType;

public:
    // Some static arrays to make access easier (needs to be defined again below the class, until C++17)
    static constexpr std::array<int, sizeof...(Is)> nunits{std::tuple_element<Is, LayerTuple>::type::size()...};
    static constexpr std::array<int, sizeof...(Is)> nbetas{std::tuple_element<Is, LayerTuple>::type::nbeta...};
};
template <class LTuplType, int ... Is>
constexpr std::array<int, sizeof...(Is)> TemplNetShape<LTuplType, std::integer_sequence<int, Is...>>::nunits;
template <class LTuplType, int ... Is>
constexpr std::array<int, sizeof...(Is)> TemplNetShape<LTuplType, std::integer_sequence<int, Is...>>::nbetas;
} // detail


// --- The fully templated TemplNet FFNN

template <typename ValueT, DerivConfig DCONF, int N_IN, class ... LayerConfs>
class TemplNet
{
private:
    // --- Static Setup

    // LayerTuple / Shape
    using LayerTuple = typename lpack::LayerPackTuple<ValueT, DCONF, N_IN, LayerConfs...>::type;
    using Shape = detail::TemplNetShape<LayerTuple, std::make_integer_sequence<int, sizeof...(LayerConfs)>>;

    LayerTuple _layers{};

    // store some basics (these stay private so they may be freely removed/replaced by constexpr method)
    static constexpr int _nlayer = tupl::count<int, LayerTuple>();
    static constexpr int _ninput = std::tuple_element<0, LayerTuple>::type::ninput;
    static constexpr int _noutput = std::tuple_element<_nlayer - 1, LayerTuple>::type::noutput;
    static constexpr int _nbeta = lpack::countBetas<N_IN, LayerConfs...>();
    static constexpr int _nunit = lpack::countUnits<LayerConfs...>();


    // Static Output Deriv Array Sizes (depend on DCONF)
    static constexpr int _nd1_out = std::tuple_element<_nlayer - 1, LayerTuple>::type::nd1;
    static constexpr int _nd2_out = std::tuple_element<_nlayer - 1, LayerTuple>::type::nd2;

    // Basic assertions
    static_assert(_nlayer == static_cast<int>(sizeof...(LayerConfs)), ""); // -> BUG!
    static_assert(_ninput == N_IN, ""); // -> BUG!
    static_assert(_noutput == Shape::nunits[_nlayer - 1], ""); // -> BUG!
    static_assert(_nlayer > 1, "[TemplNet] nlayer <= 1");
    static_assert(lpack::hasNoEmptyLayer<(_ninput > 0), LayerConfs...>(), "[TemplNet] LayerConf pack contains empty Layer.");


    // --- Non-statics

    // arrays of array.begin() pointers, for run-time public indexing
    const std::array<const ValueT *, _nlayer> _out_begins;
    const std::array<ValueT *, _nlayer> _beta_begins;

public:
    // static and dynamic derivative config
    static constexpr StaticDFlags<DCONF> dconf{}; // static derivative config
    DynamicDFlags dflags{DCONF}; // dynamic (opt-out) derivative config (default to DCONF or explicit set in ctor)

    // input array
    std::array<ValueT, _ninput> input{};

public:
    explicit constexpr TemplNet(DynamicDFlags init_dflags = DynamicDFlags{DCONF}):
            _out_begins(tupl::make_fcont<std::array<const ValueT *, _nlayer>>(_layers, [](const auto &layer) { return &layer.out().front(); })),
            _beta_begins(tupl::make_fcont<std::array<ValueT *, _nlayer>>(_layers, [](auto &layer) { return &layer.beta.front(); })),
            dflags(init_dflags) {}

    // --- Get information about the NN structure

    static constexpr int getNLayer() { return _nlayer; }
    static constexpr int getNInput() { return _ninput; }
    static constexpr int getNOutput() { return _noutput; }
    static constexpr int getNUnit() { return _nunit; }
    static constexpr int getNUnit(int i) { return Shape::nunits[i]; }
    static constexpr const auto &getShape() { return Shape::nunits; }

    // Read access to LayerTuple / individual layers
    constexpr const LayerTuple &getLayers() const { return _layers; }
    template <int I>
    constexpr const auto &getLayer() const { return std::get<I>(_layers); }

    // --- const get Value Arrays
    constexpr const auto &getInput() const { return input; } // alternative const read of public input array
    constexpr const auto &getOutput() const { return std::get<_nlayer - 1>(_layers).out(); } // get values of output layer
    constexpr const auto &getFirstDerivative() const { return std::get<_nlayer - 1>(_layers).d1(); } // get derivative of output with respect to input
    constexpr const auto &getSecondDerivative() const { return std::get<_nlayer - 1>(_layers).d2(); }

    // --- check derivative setup
    static constexpr bool allowsFirstDerivative() { return dconf.d1; }
    static constexpr bool allowsSecondDerivative() { return dconf.d2; }
    static constexpr bool allowsVariationalFirstDerivative() { return dconf.vd1; }

    constexpr bool hasFirstDerivative() const { return dconf.d1 && dflags.d1(); }
    constexpr bool hasSecondDerivative() const { return dconf.d2 && dflags.d2(); }
    constexpr bool hasVariationalFirstDerivative() const { return dconf.vd1 && dflags.vd1(); }

    // --- Access Network Weights (Betas)
    static constexpr int getNBeta() { return _nbeta; }
    static constexpr int getNBeta(int i) { return Shape::nbetas[i]; }
    static constexpr const auto &getBetaShape() { return Shape::nbetas; }
    static constexpr int getNLink() { return _nbeta - _nunit; } // substract offsets
    static constexpr int getNLink(int i) { return Shape::nbetas[i] - Shape::nunits[i]; }

    constexpr ValueT getBeta(int i) const // get beta by index
    {
        int idx = 0;
        while (i >= Shape::nbetas[idx]) {
            i -= Shape::nbetas[idx];
            ++idx;
        }
        return *(_beta_begins[idx] + i);
    }

    template <class IterT>
    constexpr void getBetas(IterT begin, const IterT end) const // get betas into range
    {
        int idx = 0;
        while (begin < end) {
            const auto blocksize = (end - begin > Shape::nbetas[idx]) ? Shape::nbetas[idx] : end - begin;
            std::copy(_beta_begins[idx], _beta_begins[idx] + blocksize, begin);
            begin += blocksize;
            ++idx;
        }
    }
    // get betas into array
    constexpr void getBetas(std::array<ValueT, _nbeta> &b_arr) const { return getBetas(b_arr.begin(), b_arr.end()); }

    constexpr void setBeta(int i, ValueT beta)
    {
        int idx = 0;
        while (i >= Shape::nbetas[idx]) {
            i -= Shape::nbetas[idx];
            ++idx;
        }
        *(_beta_begins[idx] + i) = beta;
    }

    template <class IterT>
    constexpr void setBetas(IterT begin, const IterT end)
    {
        int idx = 0;
        while (begin < end) {
            const auto blocksize = (end - begin > Shape::nbetas[idx]) ? Shape::nbetas[idx] : end - begin;
            std::copy(begin, begin + blocksize, _beta_begins[idx]);
            begin += blocksize;
            ++idx;
        }
    }
    // set betas from array
    constexpr void setBetas(const std::array<ValueT, _nbeta> &b_arr) { setBetas(b_arr.begin(), b_arr.end()); }
    /*
    ValueT getBeta(const int &ib);
    void getBeta(ValueT * beta);
    void setBeta(const int &ib, const ValueT &beta);
    void setBeta(const ValueT * beta);
    void randomizeBetas(); // has to be changed maybe if we add beta that are not "normal" weights*/

    // --- Manage the variational parameters (which may contain a subset of beta and/or non-beta parameters),
    //     which exist only after that they are assigned to actual parameters in the network (e.g. betas)
    //int getNVariationalParameters() { return _nvp; }
    /*ValueT getVariationalParameter(const int &ivp);
    void getVariationalParameter(ValueT * vp);
    void setVariationalParameter(const int &ivp, const ValueT &vp);
    void setVariationalParameter(const ValueT * vp);
    */

    // shortcut for (connecting and) adding substrates
    //void enableDerivatives(bool flag_d1, bool flag_d2, bool flag_vd1);


    // Set initial parameters
    constexpr void setInput(int i, ValueT val) { input[i] = val; }
    template <class IterT>
    constexpr void setInput(IterT begin, const IterT end) { std::copy(begin, end, input.begin()); }
    constexpr void setInput(const std::array<ValueT, _ninput> &in_arr) { input = in_arr; }


    // --- Propagation

    constexpr void FFPropagate()
    {
        propagateLayers(input, _layers, dflags);
    }

    // Shortcut for computation: set input and get all values and derivatives with one calculations.
    // If some derivatives are not supported (substrate missing) the values will be leaved unchanged.
    //void evaluate(const ValueT * in, ValueT * out, ValueT * d1 = nullptr, ValueT * d2 = nullptr, ValueT * vd1 = nullptr);

    // --- Store FFNN on file
    //void storeOnFile(const char * filename, bool store_betas = true);
};
} // templ

#endif
