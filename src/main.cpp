#include <iostream>
#include <sstream>
#include <string>
#include "dataset.cpp"
#include <unordered_map>
#include "matrix.h"
#include "../include/layers/linear.h"
#include "../include/model.h"
#include "../include/activations/relu.h"
#include "../include/activations/softmax.h"
#include "../include/optimizers/sgd.h"
#include "../include/optimizers/adam.h"
#include "../include/optimizers/optimizer.h"
#include "../include/loss_functions/mse_loss.h"
#include "../include/loss_functions/crossentropy_loss.h"



//TODO fix update
//TODO fix stepping

int main() {
    size_t NUM_EPOCHS = 100;
    std::string filename = "../sample_datasets/iris.csv";
    Dataset ds = Dataset(filename);
    DataLoader dataloader = DataLoader(ds, 1, true);


    NN::Model myModel = NN::Model(
            new NN::Linear(4, 3, InitType::Xavier)//,  // First linear layer
            //new NN::Softmax()
    );
    auto params = myModel.params();
    Optim::Adam optimizer(params, 0.01);
    NN::CategoricalCrossEntropyLoss lossFunction;



//    NN::Linear layer1 = NN::Linear(4, 3, InitType::Random);
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        double totalLoss = 0.0;
        int batches = 0;
        while (dataloader.hasMoreBatches()) {
            pair<Matrix, Matrix> batch = dataloader.nextBatch();
            Matrix batchX = batch.first;
            Matrix batchY = batch.second;
            Matrix batchYhat = myModel.forward(batchX);
            double loss = lossFunction.compute(batchYhat, batchY);
            totalLoss += loss;
            batches++;
            Matrix grad = lossFunction.gradient(batchYhat, batchY);
            myModel.backward(grad);
            optimizer.step();
        }
        dataloader.reset();
        std::cout << "-----EPOCH: " << epoch << "-----TOTAL LOSS: " << totalLoss << "-----------" << std::endl;
    }
    return 0;
}
