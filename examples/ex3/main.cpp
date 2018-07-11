#include <iostream>
#include <cmath>
#include <fstream>

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"



class MyActivationFunction: public ActivationFunctionInterface
{
public:
    // get copy method
    ActivationFunctionInterface * getCopy(){
        return new MyActivationFunction();
    }

    // get identifier
    std::string getIdCode(){
        return "MyA";
    }

    // range [-2 : 2]
    double getIdealInputMu(){return 0.;};
    double getIdealInputSigma(){return 1.154700538379252;};

    // since this is a monotonic activation function,
    // we can leave the default getOutputMu() and getOutputSigma()


    // computation
    double f(const double &in){
        // function
        return in*in*in;
    }

    double f1d(const double &in){
        // first derivative
        return 3.*in*in;
    }

    double f2d(const double &in){
        // second derivative
        return 6.*in;
    }

    double f3d(const double &in){
        // third derivative
        return 6.;
    }

    // this optional implementation allows to calculate all needed derivatives together, which is usually faster
    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false)
    {
        v = in*in*in;
        v1d = flag_d1 ? 3.*in*in : 0.;
        v2d = flag_d2 ? 6.*in    : 0.;
        v3d = flag_d3 ? 6.       : 0.;
    }
};



int main() {
    using namespace std;


    cout << "Let's start by creating a Feed Forward Artificial Neural Network (FFANN)" << endl;
    cout << "========================================================================" << endl;
    cin.ignore();

    cout << "We generate a FFANN with 4 layers and 3, 4, 5, 2 units respectively" << endl;
    cin.ignore();

    // NON I/O CODE
    FeedForwardNeuralNetwork * ffnn = new FeedForwardNeuralNetwork(3, 4, 2);
    ffnn->pushHiddenLayer(5);
    //

    cout << "Graphically it looks like this" << endl;
    cin.ignore();
    printFFNNStructure(ffnn, true, 0);
    cout << endl << endl;
    cin.ignore();


    cout << "Access an activation function" << endl;
    cout << "=============================" << endl;
    cin.ignore();

    cout << "We want to access the activation function of the 2nd neural unit of the 2nd (hidden) neural layer." << endl;
    cin.ignore();

    ActivationFunctionInterface * actf = ffnn->getNNLayer(1)->getNNUnit(1)->getActivationFunction();
    cout << "The activation function we accessed has id = " << actf->getIdCode() << endl;
    cin.ignore();

    cout << "Now we want to substitute it with our activation function (MyActivationFunction)" << endl;
    cin.ignore();

    MyActivationFunction * myactf = new MyActivationFunction();
    ffnn->getNNLayer(1)->getNNUnit(1)->setActivationFunction(myactf);
    myactf = NULL; // if you want to be sure
    cout << "Done! Note that the unit now has taken over the activation function allocation, so you must not delete it." << endl;
    cin.ignore();

    cout << "Graphically the NN now looks like this" << endl;
    cin.ignore();
    printFFNNStructure(ffnn, true, 0);
    cout << endl << endl;

    cout << endl << endl;


    return 0;
}
