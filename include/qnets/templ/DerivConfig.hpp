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
enum DerivConfig { OFF, D1, D12, VD1, VD12, D1_VD1, D1_VD12, D12_VD1, D12_VD12 };


// These mapping functions work both at compile and runtime

constexpr bool isD1Enabled(DerivConfig dconf) noexcept
{
    return !(dconf == OFF || dconf == VD1 || dconf == VD12);
}

constexpr bool isD2Enabled(DerivConfig dconf) noexcept
{
    return (dconf == D12 || dconf == D12_VD1 || dconf == D12_VD12);
}

constexpr bool isVD1Enabled(DerivConfig dconf) noexcept
{
    return !(dconf == OFF || dconf == D1 || dconf == D12);
}

constexpr bool isVD2Enabled(DerivConfig dconf) noexcept
{
    return (dconf == VD12 || dconf == D1_VD12 || dconf == D12_VD12);
}


// Static Mapping to bool flags

template <DerivConfig DCONF>
struct StaticDFlags
{
    static constexpr bool d1 = isD1Enabled(DCONF);
    static constexpr bool d2 = isD2Enabled(DCONF);
    static constexpr bool vd1 = isVD1Enabled(DCONF);
    static constexpr bool vd2 = isVD2Enabled(DCONF);

    static constexpr DerivConfig dconf() { return DCONF; }
    static constexpr bool needsAny() { return (d1 || d2 || vd1 || vd2); }
    static constexpr bool needsNone() { return !needsAny(); }
    // check for backprop needs
    static constexpr bool needsBD1() { return (needsBD2() || d1 || vd1); }
    static constexpr bool needsBD2() { return vd2; }
};

// Runtime Mapping class

class DynamicDFlags
{
private:
    bool _d1{};
    bool _d2{};
    bool _vd1{};
    bool _vd2{};

    // private "manual" constructor
    constexpr DynamicDFlags(bool flag_d1, bool flag_d2, bool flag_vd1, bool flag_vd2):
            _d1(flag_d1), _d2(flag_d2), _vd1(flag_vd1), _vd2(flag_vd2) {}

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
        _vd2 = isVD2Enabled(dconf);
    }

    template <class DFlags>
    constexpr DynamicDFlags AND(DFlags other)
    {   // Logical AND template, meant for StaticDFlags other
        return DynamicDFlags{_d1 && other.d1, _d2 && other.d2, _vd1 && other.vd1, _vd2 && other.vd2};
    }

    constexpr DynamicDFlags AND(DynamicDFlags other)
    {   // Logical AND for DynamicDFlags other
        return DynamicDFlags{_d1 && other.d1(), _d2 && other.d2(), _vd1 && other.vd1(), _vd2 && other.vd2()};
    }

    template <class DFlags>
    constexpr DynamicDFlags OR(DFlags other)
    {   // Logical OR template, meant for StaticDFlags other
        return DynamicDFlags{_d1 || other.d1, _d2 || other.d2, _vd1 || other.vd1, _vd2 || other.vd2};
    }

    constexpr DynamicDFlags OR(DynamicDFlags other)
    {   // Logical OR for DynamicDFlags other
        return DynamicDFlags{_d1 || other.d1(), _d2 || other.d2(), _vd1 || other.vd1(), _vd2 || other.vd2()};
    }

    constexpr bool d1() const { return _d1; }
    constexpr bool d2() const { return _d2; }
    constexpr bool vd1() const { return _vd1; }
    constexpr bool vd2() const { return _vd2; }

    constexpr bool needsAny() { return (_d1 || _d2 || _vd1 || _vd2); }
    constexpr bool needsNone() { return !needsAny(); }
    // check for backprop needs
    constexpr bool needsBD1() { return (needsBD2() || _d1 || _vd1); }
    constexpr bool needsBD2() { return _vd2; }
};
} // templ

#endif
