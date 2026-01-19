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


void mlp_example() {
    const std::vector<std::string> names = {"sava", "ana", "dusan", "nemanja", "petar"};
    std::vector<std::vector<int>> xs = {};
    std::vector<int> ys;

    const Tensor emmbed_table = Tensor::rand({VOCAB_SIZE, EMMBED_SIZE}, 42, DeviceType::GPU);

    for(const auto& name: names) {
        // (batch_size, LOOK_BACK, emmbed_size) or just make it like (batch_size, LOOK_BACK * emmbed_size)
        for(int y = 0; y < name.size() + 1; y++) {
            std::vector<int> cur_x = {};
            for(int x = y - LOOK_BACK; x < y; x++) {
                if(x < 0) cur_x.push_back(letter_to_index('*'));
                else cur_x.push_back(letter_to_index(name[x]));
            }
            xs.push_back(cur_x);
            ys.push_back( y < name.size() ? letter_to_index(name[y]) : letter_to_index('*'));
        }

    }

    Tensor Ys(ys, {(uint32_t)ys.size()}, DeviceType::GPU);
    Tensor Xs(xs, {(uint32_t)xs.size(), LOOK_BACK}, DeviceType::GPU);

    auto test = Xs.data();
    for(auto val : test) {
        std::cout<< val << std::endl;
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