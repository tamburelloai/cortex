//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_MODEL_H
#define CORTEX_MODEL_H

#include <vector>
#include "layers/layer.h"

namespace NN {
    class Model {
    private:
        std::vector<std::unique_ptr<Layer>> layers;
    public:
        template<typename... Layers>
        Model(Layers&&... layers) {
            (addLayer(std::forward<Layers>(layers)), ...);

        }
        void addLayer(Layer* layer) {
            layers.emplace_back(layer);
        }

        Matrix forward(const Matrix& input) {
            Matrix result = input;
            for (auto& layer : layers) {
                result = layer->forward(result);
            }
            return result;
        }

        void backward(const Matrix& gradFromLoss) {
            Matrix grad = gradFromLoss;
            for (auto it = layers.rbegin(); it != layers.rend(); it++) {
                grad = (*it)->backward(grad);
            }
        };

        std::vector<Parameter*> params() {
            std::vector<Parameter*> allParams;
            for (auto& layer : layers) {
                auto layerParams = layer->params();
                allParams.insert(allParams.end(), layerParams.begin(), layerParams.end());
            }
            return allParams;
        }

    };
}

#endif //CORTEX_MODEL_H
