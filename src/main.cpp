#include <iostream>
#include <sstream>
#include <string>
#include "dataset.cpp"
#include <unordered_map>
#include "matrix.h"
#include "../include/layers/linear.h"
#include "../include/model.h"
#include "../include/activations/relu.h"
#include "../include/optimizers/sgd.h"
#include "../include/optimizers/optimizer.h"
#include "../include/loss_functions/mse_loss.h"


int main() {
    size_t NUM_EPOCHS = 10;
    std::string filename = "../sample_datasets/iris.csv";
    Dataset ds = Dataset(filename);
    DataLoader dataloader = DataLoader(ds, 32, true);


    NN::Model myModel = NN::Model(
            new NN::Linear(4, 128, InitType::Random),  // First linear layer
            new NN::ReLU(),
            new NN::Linear(128, 3, InitType::Random)  // First linear layer
    );
    auto params = myModel.params();
    Optim::SGD sgd(params, 0.01);
    NN::MSELoss mse;



//    NN::Linear layer1 = NN::Linear(4, 3, InitType::Random);
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        double totalLoss = 0.0;
        int batches = 0;
        while (dataloader.hasMoreBatches()) {
            pair<Matrix, Matrix> batch = dataloader.nextBatch();
            Matrix batchX = batch.first;
            Matrix batchY = batch.second;
            Matrix batchYhat = myModel.forward(batchX);
            double loss = mse.compute(batchYhat, batchY);
            totalLoss += loss;
            batches++;
            //Matrix grad = mse.gradient(batchYhat, batchY);
            //myModel.backward(grad);
            //sgd.step();
        }


        dataloader.reset();
        std::cout << "-----EPOCH: " << epoch << "TOTAL LOSS: " << totalLoss << "-----------" << std::endl;
    }
    return 0;
}
