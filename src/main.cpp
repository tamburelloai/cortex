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


class SoftmaxPredictor {
public:
    // Function to apply softmax to logits and return predicted labels
    Matrix predict(const Matrix& logits) {
        int N = logits.numRows();
        int K = logits.numCols();
        Matrix yhat = Matrix(N, K);
        Matrix predictions = Matrix(N, 1); // To store the index of the maximum probability

        // Compute softmax
        for (size_t i = 0; i < N; ++i) {
            double max_logit = *std::max_element(logits.data[i].begin(), logits.data[i].end()); // for numerical stability
            double denominator = 0.0;
            for (size_t k = 0; k < K; ++k) {
                yhat.data[i][k] = std::exp(logits.data[i][k] - max_logit); // shift by max logit for stability
                denominator += yhat.data[i][k];
            }
            for (size_t k = 0; k < K; ++k) {
                yhat.data[i][k] /= denominator; // Normalize to make probabilities sum to 1
            }

            // Argmax to find the predicted class
            double max_prob = yhat.data[i][0];
            int max_index = 0;
            for (size_t k = 1; k < K; ++k) {
                if (yhat.data[i][k] > max_prob) {
                    max_prob = yhat.data[i][k];
                    max_index = k;
                }
            }
            predictions.data[i][0] = max_index;
        }

        return predictions;
    }
};

double getAccuracy(Matrix& preds, Matrix& groundTruthOneHot) {
    double N = preds.numRows();
    double correct = 0;
    double classPred;
    for (size_t i = 0; i < preds.numRows(); i++) {
        classPred = preds.data[i][0];
        if (groundTruthOneHot.data[i][int(classPred)] == 1) {
            correct += 1.0;
        }
    }
    return correct / N;
}

int main() {
    size_t NUM_EPOCHS = 100;
    std::string filename = "../sample_datasets/iris.csv";
    Dataset ds = Dataset(filename);
    DataLoader dataloader = DataLoader(ds, 16, true);


    NN::Model myModel = NN::Model(
            new NN::Linear(4, 3, InitType::Xavier)//,  // First linear layer
    );
    auto params = myModel.params();
    Optim::Adam optimizer(params, 0.00000001);
    NN::CategoricalCrossEntropyLoss lossFunction;



//    NN::Linear layer1 = NN::Linear(4, 3, InitType::Random);
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        double totalLoss = 0.0;
        double avgAccuracy = 0.0;
        int batches = 0;
        while (dataloader.hasMoreBatches()) {
            pair<Matrix, Matrix> batch = dataloader.nextBatch();
            Matrix batchX = batch.first;
            Matrix batchY = batch.second;
            Matrix batchLogits = myModel.forward(batchX);
            double loss = lossFunction.compute(batchLogits, batchY);
            Matrix gradient = lossFunction.backward(batchY);
            myModel.backward(gradient);
            optimizer.step();
            batches++;
            totalLoss += loss;

            SoftmaxPredictor predictor;
            Matrix batchPredictions = predictor.predict(batchLogits);
            avgAccuracy += getAccuracy(batchPredictions, batchY);


        }
        avgAccuracy /= batches;
        dataloader.reset();
        std::cout << "-----EPOCH: " << epoch << "-----TOTAL LOSS: " << totalLoss << "-----------" << avgAccuracy << std::endl;
    }
    return 0;
}
