#ifndef QNETS_TEMPL_LAYER_HPP
#define QNETS_TEMPL_LAYER_HPP


namespace templ
{
// --- TemplNet Layers

// Derivative Config
//
// To configure "allocation" of derivative-related std::arrays
// You may still opt OUT of the derivative COMPUTATION dynamically
// at run time, but the pre-reserved memory will always be kept.
template <bool RESERVE_D1 = false, bool RESERVE_D2 = false, bool RESERVE_VD1 = false>
struct DerivConfig
{
    static constexpr bool d1 = RESERVE_D1;
    static constexpr bool d2 = RESERVE_D2;
    static constexpr bool vd1 = RESERVE_VD1;
};

// Layer Config
//
// To pass non-input layer configurations as variadic parameter pack
template <typename SizeT, SizeT N_IN, SizeT N_OUT, class ACTF>
struct LayerConfig
{
    static constexpr SizeT ninput = N_IN;
    static constexpr SizeT noutput = N_OUT;
    static constexpr SizeT nbeta = (N_IN + 1)*N_OUT;
    static constexpr SizeT nlink = N_IN*N_OUT;
    using ACTF_Type = ACTF;

    static constexpr SizeT size() { return noutput; }
};


// The actual Layer struct
//
template <class ValueT, class LayerConf, class DerivConf>
struct Layer: LayerConf
{
    static constexpr typename LayerConf::ACTF_Type actf{};
    std::array<ValueT, LayerConf::noutput> out;
    std::array<ValueT, LayerConf::nlink> d1;
    std::array<ValueT, LayerConf::nlink> d2;
    //std::array<ValueT, LayerConf::nvp> vd1;
    std::array<ValueT, LayerConf::nbeta> beta;
};
} // templ

#endif
