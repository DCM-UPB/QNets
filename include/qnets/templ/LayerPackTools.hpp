#ifndef QNETS_TEMPL_LAYERPACKTOOLS_HPP
#define QNETS_TEMPL_LAYERPACKTOOLS_HPP

#include "qnets/templ/TemplLayer.hpp"
#include "qnets/tool/PackTools.hpp"

namespace templ
{
namespace lpack
{
// --- Layer pack helpers (some might become obsolete with C++17)


// -- LayerPackTuple struct
// Helps to determine the full layer tuple type according to LayerConfig pack
//
// recursive helpers for LayerPackTuple:
namespace detail
{
template <typename ValueT, DerivConfig DCONF, int IBETA_PREV_BEGIN, int IBETA_BEGIN, int NET_NINPUT, int N_IN, typename>
struct LayerPackTuple_rec
{
    using type = std::tuple<>;
};

template <typename ValueT, DerivConfig DCONF, int IBETA_PREV_BEGIN, int IBETA_BEGIN, int NET_NINPUT, int N_IN, typename LConf, typename... LCONFS>
struct LayerPackTuple_rec<ValueT, DCONF, IBETA_PREV_BEGIN, IBETA_BEGIN, NET_NINPUT, N_IN, std::tuple<LConf, LCONFS...>>
{
private:
    using layer = TemplLayer<ValueT, IBETA_PREV_BEGIN, IBETA_BEGIN, NET_NINPUT, N_IN, LConf::noutput, typename LConf::ACTF_Type, DCONF>;
    using rest = typename LayerPackTuple_rec<ValueT, DCONF, IBETA_BEGIN, IBETA_BEGIN+layer::nbeta, NET_NINPUT, layer::noutput, std::tuple<LCONFS...>>::type;
public:
    using type = decltype(std::tuple_cat(
            std::declval<std::tuple<layer>>(),
            std::declval<rest>()));
};
} // detail

template <typename ValueT, DerivConfig DCONF, int NET_NINPUT, typename LConf, typename... LCONFS>
class LayerPackTuple
{
private:
    using layer = TemplLayer<ValueT, 0, 0, NET_NINPUT, NET_NINPUT, LConf::noutput, typename LConf::ACTF_Type, DCONF>;
    using rest = typename detail::LayerPackTuple_rec<ValueT, DCONF, 0, layer::nbeta, NET_NINPUT, layer::noutput, std::tuple<LCONFS...>>::type;
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
