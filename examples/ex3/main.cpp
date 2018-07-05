#include <iostream>
#include <cmath>
#include <fstream>

#include "FeedForwardNeuralNetwork.hpp"
#include "PrintUtilities.hpp"



class MyActivationFunction: public ActivationFunctionInterface
{
protected:

public:

    // get copy method
    ActivationFunctionInterface * getCopy(){
        return new MyActivationFunction();
    }

    // get identifier
    std::string getIdCode(){
        return "MyA";
    }

    // computation
    double f(const double &in){
        // function
        return 2.*in*in;
    }

    double f1d(const double &in){
        // first derivative
        return 4.*in;
    }

    double f2d(const double &in){
        // second derivative
        return 4.;
    }

    double f3d(const double &in){
        // third derivative
        return 0.;
    }

    // this optional implementation allows to calculate all needed derivatives together, which is usually faster
    void fad(const double &in, double &v, double &v1d, double &v2d, double &v3d, const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_d3 = false)
    {
        v = 2.*in*in;
        v1d = flag_d1 ? 4.*in : 0.;
        v2d = flag_d2 ? 4.    : 0.;
        v3d = 0.;
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
