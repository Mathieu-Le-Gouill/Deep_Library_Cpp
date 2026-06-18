#pragma once
#include "layer.h"

template <::std::size_t... inputDims>
class Sigmoid : public Layer< Sigmoid<inputDims...>, Tensor<inputDims...>, Tensor<inputDims...>>
{
    using InputType  = Tensor<inputDims...>;
    using OutputType = Tensor<inputDims...>;

public:

    constexpr Sigmoid() = default;

    inline OutputType Forward(InputType& input) noexcept
    {
        input.apply_sigmoid();

        _output = input;// Cache the output for backpropagation

        return _output;
    }

    inline InputType Backward(OutputType& upstream_loss_gradient) noexcept
    {
        // Compute the derivative of the sigmoid function with respect to the input
        OutputType loss_gradient = std::move(upstream_loss_gradient) * _output * (_output * (-1.0f) + 1.0f);

        return loss_gradient;
    }

    inline void Update() noexcept
    {
    }

private :

    OutputType _output;
};
