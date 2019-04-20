#ifndef QNETS_TEMPL_DERIVCONFIG_HPP
#define QNETS_TEMPL_DERIVCONFIG_HPP


namespace templ
{
// --- TemplNet Derivative Config

// In TemplNet all derivatives that should potentially be used at run-time
// need to be enabled at compile-time. This configuration is handled via an
// enum that enumerates possible derivative combinations. The enum is mapped
// to a set of boolean flags via the DerivFlags template.
//
// You may still opt OUT of the derivative COMPUTATION dynamically
// at run time, but the pre-reserved memory will always be kept.

// Enumeration of allowed combinations
enum DerivConfig { OFF, D1, D12, VD1, D1_VD1, D12_VD1 };


// These mapping functions work both at compile and runtime

constexpr bool isD1Enabled(DerivConfig dconf) noexcept
{
    using DC = DerivConfig;
    return (dconf == DC::D1 || dconf == DC::D12 || dconf == DC::D1_VD1 || dconf == DC::D12_VD1);
}

constexpr bool isD2Enabled(DerivConfig dconf) noexcept
{
    using DC = DerivConfig;
    return (dconf == DC::D12 || dconf == DC::D12_VD1);
}

constexpr bool isVD1Enabled(DerivConfig dconf) noexcept
{
    using DC = DerivConfig;
    return (dconf == DC::VD1 || dconf == DC::D1_VD1 || dconf == DC::D12_VD1);
}


// Static Mapping to bool flags

template <DerivConfig DCONF>
struct StaticDFlags
{
    static constexpr bool d1 = isD1Enabled(DCONF);
    static constexpr bool d2 = isD2Enabled(DCONF);
    static constexpr bool vd1 = isVD1Enabled(DCONF);
};

// Runtime Mapping class

class DynamicDFlags
{
private:
    bool _d1{};
    bool _d2{};
    bool _vd1{};

    constexpr DynamicDFlags(bool flag_d1, bool flag_d2, bool flag_vd1):
            _d1(flag_d1), _d2(flag_d2), _vd1(flag_vd1) {}

public:
    constexpr DynamicDFlags() = default;
    explicit constexpr DynamicDFlags(DerivConfig dconf)
    {
        this->set(dconf);
    }

    constexpr void set(DerivConfig dconf) // set from DerivConfig enum
    {
        _d1 = isD1Enabled(dconf);
        _d2 = isD2Enabled(dconf);
        _vd1 = isVD1Enabled(dconf);
    }

    template <class DFlags>
    constexpr DynamicDFlags AND(DFlags other)
    {   // Logical AND template, meant for StaticDFlags other
        return DynamicDFlags{_d1 && other.d1, _d2 && other.d2, _vd1 && other.vd1};
    }

    constexpr DynamicDFlags AND(const DynamicDFlags other)
    {   // Logical AND for DynamicDFlags other
        return DynamicDFlags{_d1 && other.d1(), _d2 && other.d2(), _vd1 && other.vd1()};
    }

    constexpr bool d1() const { return _d1; }
    constexpr bool d2() const { return _d2; }
    constexpr bool vd1() const { return _vd1; }
};
} // templ

#endif
