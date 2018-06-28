#ifndef NN_TRAINING_CONFIG
#define NN_TRAINING_CONFIG

// holds the required configuration parameters for the trainer
struct NNTrainingConfig {
    bool flag_r, flag_d1, flag_d2; // use derivatives / regularization?
    double lambda_r, lambda_d1, lambda_d2; // regularization / derivative weights
    int maxn_steps, maxn_novali; // maximum number of fitting iterations / stop early after how many steps without improved validation (0 -> disabled)
};

#endif
