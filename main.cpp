#include "Tensor.hpp"
#include "Registry.hpp"
#include <iostream>
#include "OpenClContext.hpp"

// TODO: t4 = t1 * t2 + t3 this does not work right now, since t1 * t2 makes a rvalue, when i do add i take its reference but it gets deleted after this line
inline int letter_to_index(const char c) { return c == '*' ? 27 : c - 'a';}
inline char index_to_letter(const int index) {return index == 27 ? '*' : index + 'a';}


#define LOOK_BACK 3
#define VOCAB_SIZE 27
#define EMMBED_SIZE 3
#define NEURON_CNT 200


void mlp_example() {
    const std::vector<std::string> names = {"sava", "ana", "dusan", "nemanja", "petar"};
    std::vector<int> xs = {};
    std::vector<int> ys;

    for(const auto& name: names) {
        // (batch_size, LOOK_BACK, emmbed_size) or just make it like ()
        for(int y = 0; y < name.size() + 1; y++) {
            for(int x = y - LOOK_BACK; x < y; x++) {
                if(x < 0) xs.push_back(letter_to_index('*'));
                else xs.push_back(letter_to_index(name[x]));
            }
            ys.push_back( y < name.size() ? letter_to_index(name[y]) : letter_to_index('*'));
        }

    }

    Tensor Ys(ys, {(uint32_t)ys.size()}, DeviceType::GPU);
    // FLAT
    Tensor Xs(xs, {(uint32_t)xs.size() * LOOK_BACK}, DeviceType::GPU);

    Tensor C = Tensor::rand({VOCAB_SIZE, EMMBED_SIZE}, 42, DeviceType::GPU);
    C.requires_grad = true;
    Tensor W1 = Tensor::rand({EMMBED_SIZE * LOOK_BACK, NEURON_CNT}, 42, DeviceType::GPU);
    W1.requires_grad = true;
    Tensor b1 = Tensor::rand({NEURON_CNT}, 53, DeviceType::GPU);
    b1.requires_grad = true;
    Tensor W2 = Tensor::rand({NEURON_CNT, VOCAB_SIZE}, 42, DeviceType::GPU);
    W2.requires_grad = true;
    Tensor b2 = Tensor::rand({VOCAB_SIZE}, 53, DeviceType::GPU);
    b2.requires_grad = true;

    std::vector<Tensor*> params = {&C, &W1, &b1, &W2, &b2};

    std::cout<< xs.size()<< std::endl;
    std::vector<int> test_xs(xs.begin(), xs.begin() + 27);
    std::vector<int> Y(ys.begin(), ys.begin() + 3);
    Tensor Y_tensor(Y, {(uint32_t)Y.size()}, DeviceType::GPU);
    std::cout<< "STARTING" << std::endl;
    int epoch = 2;
    for(int i = 0; i < epoch; i++) {
        auto X = C.nab_rows(test_xs); 
        for(auto k: X.shape) {
            std::cout<< k << std::endl;
        }
        std::cout<<"nab rows" << std::endl;
        auto reshaped = X.reshape({9, 9});
        std::cout<<"RESHAPE" << std::endl;
        auto temp_val = reshaped.mm(W1);
        std::cout<<temp_val.requires_grad<< std::endl;
        auto next_temp = temp_val + b1;
        std::cout<<next_temp.requires_grad<< std::endl;
        auto hidden_out = next_temp.tanh();
        std::cout<<"tanh" << std::endl;
        auto ok = hidden_out.mm(W2);
        std::cout<<"second mm" << std::endl;
        auto logits = ok + b2;
        std::cout<<"second add" << std::endl;
        auto loss = logits.cross_entropy(Y_tensor);
        std::cout<< "LOSS:" << loss.index({0}) << std::endl;

        for(Tensor* param: params) {
            param->grad = nullptr;
        }

        loss.backward();
        std::cout<<loss.requires_grad<< std::endl;
        std::cout<<"herereser" << std::endl;

        for(Tensor* param: params) {
            if(param->grad == nullptr) std::cout<<"ok shit broke" << std::endl;
            //(*param)+=  *(param->grad) * (-0.1);

            std::cout<<"finished one" << std::endl;
        }

        std::cout<<"got here" << std::endl;
    }
}

int main() {
//    register_all_fill_kernels();
//    register_all_read_kernels();
//    register_all_write_elem();
//    register_all_add_kernels();
//    
//    auto logits = Tensor::ones({5, 20}, DType::Float32, DeviceType::GPU);
//    logits.requires_grad = true;
//    auto targets = Tensor::ones({5}, DType::Int32, DeviceType::GPU);
//
//    auto losses = logits.cross_entropy(targets);
//    losses.backward();
//    std::cout<< losses.shape[0] << std::endl << losses.index({0}) << ", " << losses.index({1}) << std::endl;
//    std::cout << logits.grad->index({0, 0}) << std::endl;
//    return 0;
    mlp_example();
}