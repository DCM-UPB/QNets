#ifndef NN_TRAINING_CONFIG
#define NN_TRAINING_CONFIG

// holds the required configuration parameters for the trainer
struct NNTrainingConfig {

    bool flag_d1, flag_d2, flag_r; // use derivatives / regularization?
    double lambda_r, lambda_d1, lambda_d2; // regularization / derivative weights
};

#endif
