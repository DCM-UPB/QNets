#include <iostream>
#include <random>
#include <cassert>
#include <memory>

#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/Sigmoid.hpp"

int main()
{
    using namespace std;
    using namespace templ;

    //const double TINY = 0.0001;

    const int NU_IN = 2;
    using layer1 = LayerConfig<4, actf::Sigmoid>;
    using layer2 = LayerConfig<2, actf::Sigmoid>;
    const auto dopt = DerivConfig::D1_VD1; // we want to check that D2 arrays are size 0 in that case
    using TestNet = TemplNet<double, dopt, NU_IN, layer1, layer2>;
    const StaticDFlags<dopt> dconf{}; // static flag set according to dopt
    const DynamicDFlags dflags(DerivConfig::D1); // dynamic flag set (to configure deriv calculation at runtime)


    // -- Static type-based tests

    static_assert(TestNet::getNLayer() == 2, "");
    static_assert(TestNet::getNUnit() == 6, "");
    static_assert(TestNet::getNInput() == 2, "");
    static_assert(TestNet::getNUnit(0) == 4, "");
    static_assert(TestNet::getNOutput() == 2, "");

    static_assert(TestNet::getNBeta() == 22, "");
    static_assert(TestNet::getNBeta(0) == 12, "");
    static_assert(TestNet::getNBeta(1) == 10, "");

    static_assert(TestNet::allowsD1() == dconf.d1, "");
    static_assert(TestNet::allowsD2() == dconf.d2, "");
    static_assert(TestNet::allowsVD1() == dconf.vd1, "");


    // -- Create a TemplNet instance
    TestNet test(dflags);

    // -- More basic checks

    // check again for the dynamic dflag setting
    assert(test.hasD1() == dflags.d1());
    assert(test.hasD2() == dflags.d2());
    assert(test.hasVD1() == dflags.vd1());

    constexpr array<int, 2> expected_shape{4, 2};
    constexpr array<int, 2> expected_betashape{12, 10};

    assert(TestNet::getUnitShape() == expected_shape);
    assert(TestNet::getBetaShape() == expected_betashape);

    assert(test.input.size() == TestNet::getNInput());
    assert(test.getOutput().size() == TestNet::getNOutput());


    // -- Check the layers directly

    const auto &l0 = test.getLayer<0>();
    const auto &l1 = test.getLayer<1>();
    assert(l0.size() == 4);
    assert(l0.ninput == 2);
    assert(l0.nbeta == 12);
    assert(l0.nd1 == 8);
    assert(l0.nd2 == 0);
    assert(l0.out().size() == 4);
    assert(l0.d1().size() == 8);
    assert(l0.d2().empty());
    assert(l1.size() == 2);
    assert(l1.ninput == 4);
    assert(l1.nbeta == 10);
    assert(l1.nd1 == 4);
    assert(l1.nd2 == 0);
    assert(l1.out().size() == 2);
    assert(l1.d1().size() == 4);
    assert(l1.d2().empty());


    // -- Betas

    // element-wise getBeta (initially should be 0)
    for (int i = 0; i < TestNet::getNBeta(); ++i) {
        assert(test.getBeta(i) == 0.);
    }

    // try array getBeta
    array<double, 22> cur_beta{};
    const array<double, 22> zeros{};
    array<double, 22> some_zeros{}; // the last part will be like cur_beta
    // fill in some garbage which should become 0 in the end
    iota(cur_beta.begin(), cur_beta.end(), 42.);
    copy(cur_beta.begin() + 15, cur_beta.end(), some_zeros.begin() + 15);
    test.getBetas(cur_beta.begin(), cur_beta.end() - 7);
    assert(std::equal(cur_beta.begin(), cur_beta.end(), some_zeros.begin()));
    iota(cur_beta.begin(), cur_beta.end(), 42.); // again fill some garbage
    test.getBetas(cur_beta); // get all into cur_beta (should be all 0)
    assert(std::equal(cur_beta.begin(), cur_beta.end(), zeros.begin()));

    array<double, 15> rand_beta{};
    array<double, 22> comp_beta{};
    for (int i = 0; i < 15; ++i) {
        rand_beta[i] = rand()*(1./RAND_MAX);
        comp_beta[i] = rand_beta[i];
    }
    // rest of comp_beta stays 0


    for (int i = 0; i < 15; ++i) {
        test.setBeta(i, rand_beta[i]);
        assert(test.getBeta(i) == rand_beta[i]);
    }

    test.setBetas(zeros); // set back to 0 (full array set)
    test.setBetas(rand_beta.begin(), rand_beta.end()); // set betas again, now range-based
    test.getBetas(cur_beta);
    assert(std::equal(cur_beta.begin(), cur_beta.end(), comp_beta.begin()));


    // -- Propagation (properly in another test)

    // create some new test layers
    const auto dopt2 = DerivConfig::D12_VD1; // now we enable all
    Layer<double, 2, 2, 4, actf::Sigmoid, dopt2> myl0{};
    Layer<double, 2, 4, 2, actf::Sigmoid, dopt2> myl1{};
    DynamicDFlags dflags2(dopt2);

    // set beta to random values
    for (auto &b : myl0.beta) { b = rand()*(1./RAND_MAX); }
    for (auto &b : myl1.beta) { b = rand()*(1./RAND_MAX); }

    array<double, 2> foo{-0.5, 0.3};
    myl0.PropagateInput(foo, dflags2);
    myl1.PropagateLayer(myl0.out(), myl0.d1(), myl0.d2(), dflags2);

    return 0;
}
