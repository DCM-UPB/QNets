#ifndef NN_TRAINER
#define NN_TRAINER

#include "FeedForwardNeuralNetwork.hpp"
#include "NNTrainingData.hpp"

class NNTrainer
{

protected:
    FeedForwardNeuralNetwork * _ffnn;
    NNTrainingData * _tdata;

public:
     NNTrainer(NNTrainingData * tdata, FeedForwardNeuralNetwork * ffnn) {_tdata = tdata; _ffnn = ffnn;};
    //~NNTrainer();

     void findFit(); // to be implemented by child
};


#endif
