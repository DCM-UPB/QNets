#include <iostream>
#include <random>
#include <cassert>
#include <memory>

#include "qnets/templ/TemplNet.hpp"
#include "qnets/actf/Sigmoid.hpp"
#include "qnets/actf/ReLU.hpp"
#include "qnets/actf/Exp.hpp"

int main()
{
    using namespace std;
    using namespace templ;

    //const double TINY = 0.0001;

    const int NU_IN = 2;
    using layer1 = LayerConfig<4, actf::Sigmoid>;
    using layer2 = LayerConfig<2, actf::Sigmoid>;
    const auto dopt = DerivConfig::D12_VD1; // we want to check that VD2 arrays are size 0 in that case
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
    static_assert(TestNet::allowsVD2() == dconf.vd2, "");


    // -- Create a TemplNet instance
    TestNet test(dflags);

    // -- More basic checks

    // check again for the dynamic dflag setting
    assert(test.hasD1() == dflags.d1());
    assert(test.hasD2() == dflags.d2());
    assert(test.hasVD1() == dflags.vd1());
    assert(test.hasVD2() == dflags.vd2());

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
    assert(l0.nd2 == 8);
    assert(l0.nbd1 == 8);
    assert(l0.nbd2 == 0);
    assert(l0.out().size() == 4);
    assert(l0.d1().size() == 8);
    assert(l0.d2().size() == 8);
    assert(l0.bd1().size() == 8);
    assert(l0.bd2().size() == 0);
    assert(l1.size() == 2);
    assert(l1.ninput == 4);
    assert(l1.nbeta == 10);
    assert(l1.nd2 == 4);
    assert(l1.nbd1 == 4);
    assert(l1.nbd2 == 0);
    assert(l1.out().size() == 2);
    assert(l1.d1().size() == 4);
    assert(l1.d2().size() == 4);
    assert(l1.bd1().size() == 4);
    assert(l1.bd2().size() == 0);


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


    // check file load / store
    test.storeToFile("templtest.txt");
    TestNet test2;
    test2.loadFromFile("templtest.txt");
    array<double, 22> cur2_beta{};
    test2.getBetas(cur2_beta);
    assert(std::equal(cur_beta.begin(), cur_beta.end(), cur2_beta.begin()));



    // -- Propagation / Derivatives (properly in another test)

    // create some new test layers
    // const auto dopt2 = DerivConfig::D12_VD12; // now we enable all
    // TemplLayer<double, 2/*net_nin*/, 2/*net_nout*/, (4+1)*4/*nbeta_next*/, 2/*nin*/, 4/*nout*/, actf::Sigmoid, dopt2> myl0{};
    // TemplLayer<double, 2/*net_nin*/, 2/*net_nout*/, (4+1)*2/*nbeta_next*/, 4/*nin*/, 4/*nout*/, actf::Sigmoid, dopt2> myl1{};
    // TemplLayer<double, 2/*net_nin*/, 2/*net_nout*/, 0/*nbeta_next*/, 4/*nin*/, 2/*nout*/, actf::Sigmoid, dopt2> myl2{};
    // DynamicDFlags dflags2(dopt2);

    /*
    // set beta to random values
    for (auto &b : myl0.beta) { b = rand()*(1./RAND_MAX); }
    for (auto &b : myl1.beta) { b = rand()*(1./RAND_MAX); }
    for (auto &b : myl2.beta) { b = rand()*(1./RAND_MAX); }

    const array<double, 2> foo{-0.5, 0.3};
    myl0.ForwardInput(foo, dflags2);
    myl1.ForwardLayer(myl0.out(), myl0.d1(), myl0.d2(), dflags2);
    myl2.ForwardLayer(myl1.out(), myl1.d1(), myl1.d2(), dflags2);
    myl2.BackwardOutput(dflags2);
    myl1.BackwardLayer(myl2.bd1(), myl2.bd2(), myl2.beta, dflags2);
    myl0.BackwardLayer(myl1.bd1(), myl1.bd2(), myl1.beta, dflags2);

    std::array<double, 4> ana_d1{};
    std::array<double, 4> ana_d2{};
    std::array<std::array<double, 12>, 2> ana_vd1{};
    std::array<std::array<double, 12>, 2> ana_vd2{};
    myl0.storeInputD1(ana_d1, dflags2);
    ana_d2 = myl2.d2();
    myl0.storeLayerVD1(foo, ana_vd1[0], 0, dflags2);
    myl0.storeLayerVD2(foo, ana_vd2[0], 0, dflags2);
    myl0.storeLayerVD1(foo, ana_vd1[1], 1, dflags2);
    myl0.storeLayerVD2(foo, ana_vd2[1], 1, dflags2);

    cout << endl << "all D1: ";
    for (double d1 : ana_d1) { cout << d1 << " "; }
    cout << endl << "all D2: ";
    for (double d2 : ana_d2) { cout << d2 << " "; }
    cout << endl << "all VD1: ";
    for (double vd1 : ana_vd1[0]) { cout << vd1 << " "; }
    for (double vd1 : ana_vd1[1]) { cout << vd1 << " "; }
    cout << endl << "all D2: ";
    for (double vd2 : ana_vd2[0]) { cout << vd2 << " "; }
    for (double vd2 : ana_vd2[1]) { cout << vd2 << " "; }
    cout << endl;


    const double ana_vd1_0 = foo[1]*myl0.bd1()[0];
    const double ana_vd1_1 = foo[1]*myl0.bd1()[4];
    const double ana_vd2_0 = foo[1]*foo[1]*myl0.bd2()[0];
    const double ana_vd2_1 = foo[1]*foo[1]*myl0.bd2()[4];
    cout << endl;
    cout << "ana_vd1_0: " << ana_vd1_0 << ", ana_vd1_1: " << ana_vd1_1 << endl;
    cout << "ana_vd2_0: " << ana_vd2_0 << ", ana_vd2_1: " << ana_vd2_1 << endl;
    const auto out_l = myl2.out();

    std::array<double, 4> testBD1{};
    std::array<double, 4> testBD2{};
    for (int i = 0; i < 2; ++i) {
        const int d_i0 = i*2;
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 2; ++k) {
                const double bjk = myl0.beta[1 + j*(2 + 1) + k];
                testBD1[d_i0 + k] += bjk*myl0.bd1()[i*4 + j];
                testBD2[d_i0 + k] += bjk*bjk*myl0.bd2()[i*4 + j];
            }
        }
    }

    cout << endl << "IL BD1: ";
    for (double bd1 : testBD1) { cout << bd1 << " "; }
    cout << endl << "IL BD2: ";
    for (double bd2 : testBD2) { cout << bd2 << " "; }
    cout << endl << endl;

    // check input derivs

    const double dx = 0.000001;
    auto foo2 = foo;
    foo2[0] += dx;
    myl0.ForwardInput(foo2, dflags2);
    myl1.ForwardLayer(myl0.out(), myl0.d1(), myl0.d2(), dflags2);
    myl2.ForwardLayer(myl1.out(), myl1.d1(), myl1.d2(), dflags2);
    myl2.BackwardOutput(dflags2);
    myl1.BackwardLayer(myl2.bd1(), myl2.bd2(), myl2.beta, dflags2);
    myl0.BackwardLayer(myl1.bd1(), myl1.bd2(), myl1.beta, dflags2);
    const auto out_r_x = myl2.out();

    std::array<double, 4> ana_d1_r{};
    std::array<double, 4> ana_d2_r{};
    myl0.storeInputD1(ana_d1_r, dflags2);
    ana_d2_r = myl2.d2();

    const double num_d1_0 = (out_r_x[0] - out_l[0])/dx;
    const double num_d1_1 = (out_r_x[1] - out_l[1])/dx;
    const double num_d2_0 = (ana_d1_r[0] - ana_d1[0])/dx;
    const double num_d2_1 = (ana_d1_r[2] - ana_d1[2])/dx;
    cout << "num_d1_0: " << num_d1_0 << ", num_d1_1: " << num_d1_1 << endl;
    cout << "num_d2_0: " << num_d2_0 << ", num_d2_1: " << num_d2_1 << endl;


    // check beta derivs

    const double db = 0.000001;
    myl0.beta[2] += db;
    myl0.ForwardInput(foo, dflags2);
    myl1.ForwardLayer(myl0.out(), myl0.d1(), myl0.d2(), dflags2);
    myl2.ForwardLayer(myl1.out(), myl1.d1(), myl1.d2(), dflags2);
    myl2.BackwardOutput(dflags2);
    myl1.BackwardLayer(myl2.bd1(), myl2.bd2(), myl2.beta, dflags2);
    myl0.BackwardLayer(myl1.bd1(), myl1.bd2(), myl1.beta, dflags2);
    const auto out_r_b = myl2.out();

    const auto ana_vd1_r_0 = foo[1]*myl0.bd1()[0];
    const auto ana_vd1_r_1 = foo[1]*myl0.bd1()[4];

    const double num_vd1_0 = (out_r_b[0] - out_l[0])/db;
    const double num_vd1_1 = (out_r_b[1] - out_l[1])/db;
    const double num_vd2_0 = (ana_vd1_r_0 - ana_vd1_0)/db;
    const double num_vd2_1 = (ana_vd1_r_1 - ana_vd1_1)/db;
    cout << "num_vd1_0: " << num_vd1_0 << ", num_vd1_1: " << num_vd1_1 << endl;
    cout << "num_vd2_0: " << num_vd2_0 << ", num_vd2_1: " << num_vd2_1 << endl;
*/


    return 0;
}
