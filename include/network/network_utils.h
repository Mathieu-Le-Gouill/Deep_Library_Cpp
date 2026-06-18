#include "tensor/tensor.h"
#include "compiler/compiler_optimizations.h"
#include <cmath>
#include <random>
#include <cstring>

#ifndef _CMP_GT_OQ
    #define _CMP_GT_OQ 30
#endif

// ------ TENSORS LAYERS FUNCTIONS -------

inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Apply the sigmoid function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_sigmoid()
{
    const PACKAGE_FLOAT zeros = _SETZERO();
    const PACKAGE_FLOAT ones  = _SET1_PS(1.f);

    size_t i = 0;

    UNROLL_LOOP(UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT x = _LOAD(_values + i);
        const PACKAGE_FLOAT e = _EXP(_SUB_PS(zeros, x));           // exp(-x)
        const PACKAGE_FLOAT s = _DIV(ones, _ADD_PS(ones, e));       // 1 / (1 + exp(-x))
        _STORE(_values + i, s);
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT x = _LOAD(_values + i);
        const PACKAGE_FLOAT e = _EXP(_SUB_PS(zeros, x));
        const PACKAGE_FLOAT s = _DIV(ones, _ADD_PS(ones, e));
        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), s);
    }
}


// Apply the ReLu function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_ReLU()
{
    const PACKAGE_FLOAT zeros = _SETZERO();

    size_t i = 0;

    UNROLL_LOOP(UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

        _STORE(_values + i, _MAX_PS(packedValues, zeros)); // Store values in memory
	}

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), _MAX_PS(packedValues, zeros)); // Store values in memory
    }
}


// Apply the ReLu derivative function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_ReLU_derivative()
{
    const PACKAGE_FLOAT zeros = _SETZERO();

    size_t i = 0;

    UNROLL_LOOP(UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

        const PACKAGE_FLOAT mask = _CMPGT_PS(packedValues, zeros); // Create mask

        _STORE(_values + i, _AND(mask, packedValues)); // Store values in memory
	}

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

		const PACKAGE_FLOAT mask = _CMPGT_PS(packedValues, zeros); // Create mask

		_MASKSTORE(_values + i, remainderMaskSI<_offset>(), _AND(mask, packedValues)); // Store values in memory
    }
}


// Apply a normal distribution on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_normalization()
{
    const float mean = this->mean();

    // Maybe there is a better way to compute standard deviation reciprocal
    const float std = std::sqrt(this->variance(mean));
    const float sigmaReciprocal = 1.f / std;

    PACKAGE_FLOAT packedMean = _SET1_PS(mean);
    PACKAGE_FLOAT packedSigmaReciprocal = _SET1_PS(sigmaReciprocal);

    size_t i = 0;

    UNROLL_LOOP(UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

        #ifdef __FMA__
            const PACKAGE_FLOAT result = _FMSUB(packedValues, packedSigmaReciprocal, packedMean); // Normalize
        #else
            const PACKAGE_FLOAT result = _SUB_PS(_MUL(packedValues, packedSigmaReciprocal), packedMean); // Normalize
        #endif

        _STORE(_values + i, result);
	}

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

        #ifdef __FMA__
            const PACKAGE_FLOAT result = _FMSUB(packedValues, packedSigmaReciprocal, packedMean); // Normalize
        #else
            const PACKAGE_FLOAT result = _SUB_PS(_MUL(packedValues, packedSigmaReciprocal), packedMean); // Normalize
        #endif

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), result); // Store values in memory
    }
}


// Normalize the tensor based on a given mean and variance then shift and scale the values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::norm_shift_and_scale(float mean, float variance, float shift, float scale)
{
    debug_assert(variance != 0 && "Error in Tensor apply_shift_and_scale : the given variance is null.");

    // Maybe there is a better way to compute sigma reciprocal
    const float sigmaReciprocal = 1.f / sqrtf(variance);

    PACKAGE_FLOAT packedShift = _SET1_PS(shift);
    PACKAGE_FLOAT packedScale = _SET1_PS(scale);

    PACKAGE_FLOAT packedMean = _SET1_PS(mean);
    PACKAGE_FLOAT packedSigmaReciprocal = _SET1_PS(sigmaReciprocal);

    size_t i = 0;

    UNROLL_LOOP(UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

        #ifdef __FMA__
            const PACKAGE_FLOAT norm = _FMSUB(packedValues, packedSigmaReciprocal, packedMean); // Normalize
            const PACKAGE_FLOAT result = _FMADD(norm, packedScale, packedShift); // Scale and shift
        #else
            const PACKAGE_FLOAT norm = _SUB_PS(_MUL(packedValues, packedSigmaReciprocal), packedMean); // Normalize
            const PACKAGE_FLOAT result = _ADD_PS(_MUL(norm, packedScale), packedShift); // Scale and shift
        #endif

        _STORE(_values + i, result); // Store values in memory
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

        #ifdef __FMA__
            const PACKAGE_FLOAT norm = _FMSUB(packedValues, packedSigmaReciprocal, packedMean); // Normalize
            const PACKAGE_FLOAT result = _FMADD(norm, packedScale, packedShift); // Scale and shift
        #else
            const PACKAGE_FLOAT norm = _SUB_PS(_MUL(packedValues, packedSigmaReciprocal), packedMean); // Normalize
            const PACKAGE_FLOAT result = _ADD_PS(_MUL(norm, packedScale), packedShift); // Scale and shift
        #endif

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), result); // Store values in memory
    }
}


// Apply dropout on the tensor values setting random values to zero based on a given rate
// Return the build mask to be able to apply the same dropout on the backward pass
template<::std::size_t ...Dimensions>
Tensor<Dimensions...> Tensor<Dimensions...>::apply_dropout(float rate)
{
    debug_assert(rate > 0.f && rate < 1.f && "Error in Tensor apply_dropout : the given rate is not between 0 and 1.");

    Tensor<Dimensions...> dropoutMask;
    dropoutMask.init();

	const PACKAGE_FLOAT packedRate = _SET1_PS(rate);

    auto seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
    std::default_random_engine generator(seed);// Create a generator of random numbers

    std::uniform_real_distribution<float> distribution(0, 1);

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        // Generate random values between 0 and 1
        const PACKAGE_FLOAT randomValues = _RAND(generator, distribution);

        const PACKAGE_FLOAT mask = _CMPGT_PS(randomValues, packedRate); // Create mask

        const PACKAGE_FLOAT packedValues = _LOAD(this->_values + i); // Load values from memory

        _STORE(this->_values + i, _AND(packedValues, mask)); // Store values in memory

        _STORE(dropoutMask._values + i, mask); // Save mask
	}

    if constexpr (_offset)
    {
        // Generate random values between 0 and 1
        const PACKAGE_FLOAT randomValues = _RAND(generator, distribution);

		const PACKAGE_FLOAT mask = _CMPGT_PS(randomValues, packedRate); // Create mask

        const PACKAGE_FLOAT packedValues = _LOAD(this->_values + i); // Load values from memory

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _AND(packedValues, mask)); // Store values in memory

        _MASKSTORE(dropoutMask._values + i, remainderMaskSI<_offset>(), mask); // Save mask
	}

    return dropoutMask;
}


// Apply the given mask on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_mask(const Tensor<Dimensions...>& mask)
{
    size_t i = 0;

    UNROLL_LOOP(UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory
		const PACKAGE_FLOAT packedMask = _LOAD(mask._values + i); // Load mask from memory

		_STORE(_values + i, _AND(packedValues, packedMask)); // Apply mask
	}

    if constexpr (_offset)
    {
		const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory
        const PACKAGE_FLOAT packedMask = _LOAD(mask._values + i); // Load mask from memory

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), _AND(packedValues, packedMask)); // Apply mask
	}
}

// Apply the softmax function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_Softmax()
{
    const float maxValue = this->max(); // max values of the tensor

    // Compute sum
    const PACKAGE_FLOAT packedMax = _SET1_PS(maxValue);

    PACKAGE_FLOAT packedSum;

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        packedSum = _EXP( _SUB_PS( _LOAD(_values), packedMax));  // Compute exponent

        size_t i = PACKAGE_LENGTH;

        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

            const PACKAGE_FLOAT expValues = _EXP( _SUB_PS(packedValues, packedMax));  // Compute exponent

			packedSum = _ADD_PS(expValues, packedSum);
        }

        if constexpr (_offset)
        {
            const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

            const PACKAGE_FLOAT expValues = _EXP( _SUB_PS(packedValues, packedMax));  // Compute exponent

            const PACKAGE_FLOAT maskedValues = _AND(expValues, remainderMask<_offset>()); // Apply mask

            packedSum = _ADD_PS(maskedValues, packedSum);
        }
    }
    else
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values); // Load values from memory

        const PACKAGE_FLOAT expValues = _EXP( _SUB_PS(packedValues, packedMax));  // Compute exponent

        packedSum = _AND(expValues, remainderMask<_offset>());
    }

    const float totalSum = _HSUM(packedSum); // Sum all values

    PACKAGE_FLOAT packedRCPSum = _SET1_PS(1.f/totalSum);

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

		const PACKAGE_FLOAT expValues = _EXP( _SUB_PS(packedValues, packedMax));  // Compute exponent
		const PACKAGE_FLOAT softmax = _MUL(expValues, packedRCPSum); // Compute softmax

        _STORE(_values + i, softmax); // Store values in memory
	}

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(_values + i); // Load values from memory

        const PACKAGE_FLOAT expValues = _EXP(_SUB_PS(packedValues, packedMax));  // Compute exponent
        const PACKAGE_FLOAT softmax = _MUL(expValues, packedRCPSum); // Compute softmax

        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), softmax); // Store values in memory
	}
}


// Apply the softmax derivative function on the tensor values
template<::std::size_t ...Dimensions>
void Tensor<Dimensions...>::apply_Softmax_derivative()
{
    float upstreamLossGradient[_size];

    std::memcpy(upstreamLossGradient, _values, _size * sizeof(float));

    float* lossGradientValue = (float*)_values;

    // For each loss gradient values
    for (size_t i = 0; i < _size; i++)
    {
        const float& upstreamLossGradientValue = upstreamLossGradient[i];

        // For each upstream loss gradient values
        for (size_t j = 0; j < _size; j++)
        {
            // Compute the loss gradient
            lossGradientValue[i] += (i == j) ?
                upstreamLossGradientValue * (1 - upstreamLossGradientValue) : - upstreamLossGradientValue * (upstreamLossGradient[j]);
        }

    }
}


// Compute the cross entropy loss function
template<::std::size_t ...Dimensions>
float Cross_Entropy_loss(const Tensor<Dimensions...>& prediction, const Tensor<Dimensions...>& labels)
{
    constexpr size_t offset = Tensor<Dimensions...>::_offset;
    constexpr size_t size = Tensor<Dimensions...>::_size;

    PACKAGE_FLOAT packedSum;

    if constexpr (size >= PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT log = _LOG( _LOAD(prediction._values)); // Load labels from memory

        packedSum = _MUL( _LOAD(labels._values), log);

        size_t i = PACKAGE_LENGTH;

        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT labelsValues = _LOAD(labels._values + i); // Load labels from memory
            const PACKAGE_FLOAT predictionValues = _LOAD(prediction._values + i); // Load prediction from memory

            #ifdef __FMA__
                const PACKAGE_FLOAT values = _FMADD(labelsValues, _LOG(predictionValues), packedSum); // Compute loss
            #else
                const PACKAGE_FLOAT packedSum = _ADD_PS(_MUL(labelsValues, _LOG(predictionValues)), packedSum);
            #endif
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT labelsValues = _MASKLOAD(labels._values + i, remainderMaskSI<offset>()); // Load labels from memory
            const PACKAGE_FLOAT predictionValues = _LOG(prediction._values + i); // Load prediction from memory

            const PACKAGE_FLOAT maskedPredictionsLog = _AND( _LOG(predictionValues), remainderMask<offset>()); // Apply mask on the log of the prediction

            #ifdef __FMA__
                const PACKAGE_FLOAT values = _FMADD(labelsValues, maskedPredictionsLog, packedSum); // Compute loss
            #else
                const PACKAGE_FLOAT packedSum = _ADD_PS(_MUL(labelsValues, maskedPredictionsLog), packedSum);
            #endif
        }
    }
    else 
    {
        const PACKAGE_FLOAT labelsValues = _LOAD(labels._values); // Load labels from memory
		const PACKAGE_FLOAT predictionValues = _LOAD(prediction._values); // Load prediction from memory

		const PACKAGE_FLOAT values = _MUL(labelsValues, _LOG(predictionValues));

		packedSum = _AND(values, remainderMask<offset>()); // Apply mask
	}

	const float ce = _HSUM(packedSum); // Sum all values

	return -ce;
}