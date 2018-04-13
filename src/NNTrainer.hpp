#ifndef NN_TRAINER
#define NN_TRAINER

class NNTrainer
{

protected:
    FeedForwardNeuralNetwork * _ffnn;
    NNTrainingData * _tdata;

public:
     NNTrainer(NNTrainingData * tdata, FeedForwardNeuralNetwork * ffnn) {_tdata = tdata; _ffnn = ffnn;);
    ~NNTrainer;

    void findFit; // to be implemented by child
};


#endif
