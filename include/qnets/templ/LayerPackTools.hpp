#ifndef QNETS_TEMPL_LAYERPACKTOOLS_HPP
#define QNETS_TEMPL_LAYERPACKTOOLS_HPP

#include "qnets/templ/TemplLayer.hpp"
#include "qnets/tool/PackTools.hpp"

namespace templ
{
namespace lpack
{
// --- Layer pack helpers (some might become obsolete with C++17)



// Helpers for LayerPackTuple (below):
namespace detail
{
template <class LConf>
constexpr int net_nout() { return LConf::noutput; } // last layers output
template <class LConf1, class LConf2, class ... Rest> // all lconfs should be passed
constexpr int net_nout() { return net_nout<LConf2, Rest...>(); }

template <class LConf> // last layer
constexpr int nbeta_next() { return 0; }
template <class LConf1, class LConf2, class ... Rest> // LConf2 is "next"
constexpr int nbeta_next() { return (1 + LConf1::noutput)*LConf2::noutput; }

template <typename ValueT, DerivConfig DCONF, int NET_NINPUT, int NET_NOUTPUT, int N_IN, class>
struct LayerPackTuple_rec
{
    using type = std::tuple<>;
};

template <typename ValueT, DerivConfig DCONF, int NET_NINPUT, int NET_NOUTPUT, int N_IN, class LConf, class ... LCONFS>
struct LayerPackTuple_rec<ValueT, DCONF, NET_NINPUT, NET_NOUTPUT, N_IN, std::tuple<LConf, LCONFS...>>
{
private:
    using layer = TemplLayer<ValueT, NET_NINPUT, NET_NOUTPUT, nbeta_next<LConf, LCONFS...>(), N_IN, LConf::noutput, typename LConf::ACTF_Type, DCONF>;
    using rest = typename LayerPackTuple_rec<ValueT, DCONF, NET_NINPUT, NET_NOUTPUT, layer::noutput, std::tuple<LCONFS...>>::type;
public:
    using type = decltype(std::tuple_cat(
            std::declval<std::tuple<layer>>(),
            std::declval<rest>()));
};
} // detail

// -- LayerPackTuple
//
// Helps to determine the full layer tuple type according to LayerConfig pack
//
template <typename ValueT, DerivConfig DCONF, int NET_NINPUT, class LConf, class ... LCONFS>
struct LayerPackTuple
{
private:
    static constexpr int net_noutput = detail::net_nout<LConf, LCONFS...>();
    using layer = TemplLayer<ValueT, NET_NINPUT, net_noutput, detail::nbeta_next<LConf, LCONFS...>(), NET_NINPUT, LConf::noutput, typename LConf::ACTF_Type, DCONF>;
    using rest = typename detail::LayerPackTuple_rec<ValueT, DCONF, NET_NINPUT, net_noutput, layer::noutput, std::tuple<LCONFS...>>::type;
public:
    using type = decltype(std::tuple_cat(
            std::declval<std::tuple<layer>>(),
            std::declval<rest>()));
};


// count total number of units across Layer pack
template <class ... LAYERS>
constexpr int countUnits()
{
    return pack::sum<int, int, LAYERS::size()...>();
}

// count total number of weights across Layer pack (recursive)
template <int prev_nunits>
constexpr int countBetas() { return 0; }  // terminate recursion
template <int prev_nunits, class Layer, class ... LAYERS>
constexpr int countBetas()
{
    return Layer::size()*(prev_nunits + 1/*offset*/) + countBetas<Layer::size(), LAYERS...>(); // recurse
}

// check for empty layers across pack (use it "passing" first argument with initial boolean (usually true)
template <bool init_ret>
constexpr bool hasNoEmptyLayer() { return init_ret; } // termination
template <bool init_ret, class Layer, class ... LAYERS>
constexpr bool hasNoEmptyLayer()
{
    const bool new_ret = init_ret && Layer::size() > 0;
    return new_ret && hasNoEmptyLayer<new_ret, LAYERS...>(); // recurse
}
} // lpack
} // templ

#endif
