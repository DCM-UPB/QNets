#ifndef QNETS_TEMPL_LAYERPACKTOOLS_HPP
#define QNETS_TEMPL_LAYERPACKTOOLS_HPP


namespace templ
{
namespace lpack
{
// --- Layer pack helpers (might be mostly obsolete with C++17)

// count total number of units across Layer pack (recursive)
template <typename SizeT>
constexpr SizeT countUnits() { return 0; } // terminate recursion
template <typename SizeT, class Layer, class ... LAYERS>
constexpr SizeT countUnits()
{
    return Layer::size() + countUnits<SizeT, LAYERS...>(); // recurse until all layer's units counted
}

// count total number of weights across Layer pack (recursive)
template <typename SizeT>
constexpr SizeT countBetas(SizeT/*prev_nunits*/) { return 0; } // terminate recursion
template <typename SizeT, class Layer, class ... LAYERS>
constexpr SizeT countBetas(SizeT prev_nunits)
{
    return Layer::size()*(prev_nunits + 1/*offset*/) + countBetas<SizeT, LAYERS...>(Layer::size()); // recurse
}

// check for empty layers across pack (use it "passing" first argument with initial boolean (usually true)
template <bool ret>
constexpr bool hasNoEmptyLayer() { return ret; } // terminate recursion
template <bool ret, class Layer, class ... LAYERS>
constexpr bool hasNoEmptyLayer()
{
    return hasNoEmptyLayer<(ret && Layer::size() > 0), LAYERS...>();
}
} // lpack
} // templ

#endif
