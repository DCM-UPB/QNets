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
    using layer1 = LayerConfig<int, NU_IN, 4, actf::Sigmoid>;
    using layer2 = LayerConfig<int, layer1::size(), 2, actf::Sigmoid>;
    const auto dconf = DerivConfig::D1_VD1; // we want to check that D2 arrays are size 0 in that case
    using TestNet = TemplNet<int, double, dconf, layer1, layer2>;


    // -- Static type-based tests

    static_assert(TestNet::getNLayer() == 2, "");
    static_assert(TestNet::getNUnit() == 6, "");
    static_assert(TestNet::getNInput() == 2, "");
    static_assert(TestNet::getNUnit(0) == 4, "");
    static_assert(TestNet::getNOutput() == 2, "");

    static_assert(TestNet::getNBeta() == 22, "");
    static_assert(TestNet::getNBeta(0) == 12, "");
    static_assert(TestNet::getNBeta(1) == 10, "");
    static_assert(TestNet::getNLink() == 16, "");
    static_assert(TestNet::getNLink(0) == 8, "");
    static_assert(TestNet::getNLink(1) == 8, "");

    static_assert(TestNet::allowsFirstDerivative(), "");
    static_assert(!TestNet::allowsSecondDerivative(), "");
    static_assert(TestNet::allowsVariationalFirstDerivative(), "");


    // -- Create a TemplNet instance

    //auto test_ptr = make_unique<TestNet>();
    //auto &test = *test_ptr;
    TestNet test{};

    // Some basic checks
    assert(test.hasFirstDerivative());
    assert(!test.hasSecondDerivative());
    assert(test.hasVariationalFirstDerivative());

    constexpr std::array<int, 2> expected_shape{4, 2};
    constexpr std::array<int, 2> expected_betashape{12, 10};

    assert(TestNet::getShape() == expected_shape);
    assert(TestNet::getBetaShape() == expected_betashape);

    assert(test.input.size() == TestNet::getNInput());
    assert(test.output.size() == TestNet::getNOutput());
    assert(test.output.size() == test.getOutput().size());


    // -- Check the layers directly

    auto l0 = test.getLayer<0>();
    auto l1 = test.getLayer<1>();
    static_assert(l0.size()==4, "l0.size() != 4");
    static_assert(l0.ninput==2, "");
    static_assert(l0.nlink==8, "");
    static_assert(l0.nbeta==12, "");
    static_assert(l0.nd1==8, "");
    static_assert(l0.nd2==0, "");
    assert(l0.out.size()==4);
    assert(l0.d1.size()==8);
    assert(l0.d2.empty());
    static_assert(l1.size()==2, "");
    static_assert(l1.ninput==4, "");
    static_assert(l1.nlink==8, "");
    static_assert(l1.nbeta==10, "");
    static_assert(l1.nd1==4, "");
    static_assert(l1.nd2==0, "");
    assert(l1.out.size()==2);
    assert(l1.d1.size()==4);
    assert(l1.d2.empty());


    // -- Betas

    // element-wise getBeta (initially should be 0)
    for (int i = 0; i < TestNet::getNBeta(); ++i) {
        assert(test.getBeta(i) == 0.);
    }

    // try array getBeta
    std::array<double, 22> cur_beta{};
    const std::array<double, 22> zeros{};
    std::array<double, 22> some_zeros{}; // the last part will be like cur_beta
    // fill in some garbage which should become 0 in the end
    std::iota(cur_beta.begin(), cur_beta.end(), 42.);
    std::copy(cur_beta.begin()+15, cur_beta.end(), some_zeros.begin()+15);
    std::cout << "sz ";
    for (double sz : some_zeros) { cout << sz << " "; }
    std::cout << std::endl;
    test.getBetas(cur_beta.begin(), cur_beta.end()-7);
    std::cout << "cb ";
    for (double cb : cur_beta) { cout << cb << " "; }
    std::cout << std::endl;
    assert(std::equal(cur_beta.begin(), cur_beta.end(), some_zeros.begin()));
    std::iota(cur_beta.begin(), cur_beta.end(), 42.); // again fill some garbage
    test.getBetas(cur_beta.begin()); // get all into cur_beta (should be all 0)
    assert(std::equal(cur_beta.begin(), cur_beta.end(), zeros.begin()));

    std::array<double, 15> rand_beta{};
    std::array<double, 22> comp_beta{};
    for (int i = 0; i < 15; ++i) {
        rand_beta[i] = rand()*(1./RAND_MAX);
        comp_beta[i] = rand_beta[i];
    }
    // rest of comp_beta stays 0

    cout << "beta (rand) ";
    for (int i = 0; i < 15; ++i) {
        test.setBeta(i, rand_beta[i]);
        cout << "b" << i << " " << test.getBeta(i) << "  ";
        assert(test.getBeta(i) == rand_beta[i]);
    }
    cout << endl;

    test.setBetas(zeros.begin()); // set back to 0 (implicit full range)
    test.setBetas(rand_beta.begin(), rand_beta.end()); // set betas again, now range-based

    test.getBetas(cur_beta.begin());
    //assert(std::equal(cur_beta.begin(), cur_beta.end(), comp_beta.begin()));

   /*
    actf::Sigmoid actf{};
    std::array<double, 2> foo{-0.5, 0.5};
    std::array<double, 2> bar{};
    actf.f(foo.begin(), foo.end(), bar.begin());
    cout << "actf ";
    for (double out : bar) { cout << out << " "; }
    cout << endl;
*/

    // -- Propagation

    std::array<double, 2> foo{-0.5, 0.3};
    l0.PropagateInput(foo, DynamicDFlags(DerivConfig::D12_VD1));
    l1.PropagateLayer(l0.out, l0.d1, l0.d2, DynamicDFlags(DerivConfig::D12_VD1));

    cout << "layer output ";
    for (double out : l1.out) { cout << out << " "; }
    cout << endl;

/*
       FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 5, 3);
       ffnn->pushHiddenLayer(4);

       ffnn->pushFeatureMapLayer(5);
       ffnn->getFeatureMapLayer(0)->setNMaps(0, 0, 1, 1, 2);

       ffnn->pushFeatureMapLayer(4);
       ffnn->getFeatureMapLayer(1)->setNMaps(1, 1, 0, 1, 0); // we specify only 3 units
       ffnn->getFeatureMapLayer(1)->setSize(6); // now the other 2 should be defaulted to IDMU (generates warning)
       ffnn->getFeatureMapLayer(1)->setNMaps(1, 1, 0, 1, 2); // to suppress further warning on copies

       //printFFNNStructure(ffnn);

       ffnn->connectFFNN();
*/

/*
    // random generator with fixed seed for generating the beta, in order to eliminate randomness of results in the unittest
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-2., 2.);
    for (int i = 0; i < ffnn->getNBeta(); ++i) {
        ffnn->setBeta(i, rd(rgen));
    }
*/
    //printFFNNStructure(ffnn, false, 0);

    return 0;
}