#include "tensor.h"
#include "debug.h"
#include "network/network_utils.h"
#include "tensor/kernel_utils.h"
#include "simd/simd_utils.h"

#include <cstring>
#include <cfloat>
#include <optional>
#include <sstream>
#include <iomanip>
#include "compiler/compiler_optimizations.h"
#include <random>
#include <chrono>


// TODO : Look for OpenMP for threading and compare it with manual threading
//        See for alternative implementation of the argmax function
//        See for altnernative implementation of the transpose function
//        Think about how the loop could be unrolled to use all the registers available
// ------- TENSORS ALIAS -------


// Alias for the Tensor of two dimensions
template <std::size_t cols, std::size_t rows>
using Matrix = Tensor<cols, rows>;

// Alias for the Tensor of one dimension
template <std::size_t size>
using Vector = Tensor<size>;


// ------- TENSORS CONSTRUCTORS -------


// Initialize the tensor values
template<::std::size_t ...Dimensions>
inline void Tensor<Dimensions...>::init()
{
    _values = static_cast<float*>(_mm_malloc(_size * sizeof(float), PACKAGE_SIZE));
}


// Default constructor
template<std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor()
{
    _values = nullptr;
}


// Copy constructor
template<std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor(const Tensor<Dimensions...>& other)
{
    init();

    if(other._values)
        std::memcpy(this->_values, other._values, _size * sizeof(float));
}


// Rvalue assignement constructor
template <std::size_t... Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor(Tensor<Dimensions...>&& other) noexcept
{        
    this->_values = other._values;

    other._values = nullptr;
}


// Assignement constructor for a different dimensions Tensor of the same size
template <std::size_t... Dimensions>
template <std::size_t... OtherDimensions>
inline constexpr Tensor<Dimensions...>::Tensor(const Tensor<OtherDimensions...>& other)
{
    static_assert((1 * ... * Dimensions) == (1 * ... * OtherDimensions), "Error in Tensor move constructor: incorrect dimensions");

    init();

    std::memcpy(this->_values, other._values, _size * sizeof(float));
}


// Rvalue assignement constructor for a different dimensions Tensor of the same size
template <std::size_t... Dimensions>
template <std::size_t... OtherDimensions>
inline constexpr Tensor<Dimensions...>::Tensor(Tensor<OtherDimensions...>&& other) noexcept
{
    static_assert((1 * ... * Dimensions) == (1 * ... * OtherDimensions), "Error in Tensor move constructor: incorrect dimensions");

    this->_values = other._values;

    other._values = nullptr;
}


// Fill Tensors with a given value
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...>::Tensor(float value)
{
    // <!> TODO : Check if memset is not better
    init();

    const PACKAGE_FLOAT packedValue = _SET1_PS(value);

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        _STORE(_values + i, packedValue);
    }

    if constexpr (_offset)
        _MASKSTORE(_values + i, remainderMaskSI<_offset>(), packedValue);
}


// Fill the tensor from an initializer list
template<std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor(std::initializer_list<float> values)
{
    debug_assert(values.size() == _size && "Error in Tensor constructor : the given initalizer list does not have a valid size.");

    init();

    std::memcpy(_values, values.begin(), _size * sizeof(float));
}



// Fill the tensor from an initializer list
template<std::size_t ...Dimensions>
inline constexpr Tensor<Dimensions...>::Tensor(float values[_size])
{
    init();

    std::memcpy(_values, values, _size * sizeof(float));
}


// Fill Tensors with zeros
template <std::size_t... Dimensions>
Tensor<Dimensions...> zeros()
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_FLOAT packedValues = _SETZERO();

    constexpr uint16_t offset = output._offset;
    constexpr std::size_t size = output._size;

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        _STORE(output._values + i, packedValues);
    }

    if constexpr (offset)
        _MASKSTORE(output._values + i, remainderMaskSI<offset>(), packedValues);

    return output;
}


// Fill Tensors with ones
template <std::size_t... Dimensions>
inline Tensor<Dimensions...> ones()
{
    return Tensor<Dimensions...>(1.0f);
}


// Fill Tensors with random values from a normal distribution based on mean and the standard deviation
template <std::size_t... Dimensions>
Tensor<Dimensions...> normal(float mean, float std)
{
    Tensor<Dimensions...> output;

    output.init();

    constexpr uint16_t offset = output._offset;
    constexpr size_t size = output._size;

    auto seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();// To get differents epochs 
    std::default_random_engine generator(seed);// Create a generator of random numbers

    std::normal_distribution<float> distribution(mean, std);

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT randomValues = _RAND(generator, distribution);

        _STORE(output._values + i, randomValues);
    }

    if constexpr (offset)
    {
        const PACKAGE_FLOAT randomValues = _RAND(generator, distribution);
        
        _MASKSTORE(output._values + i, remainderMaskSI<offset>(), randomValues);
    }

    return output;
}


// Destructor
template<std::size_t ...Dimensions>
inline Tensor<Dimensions...>::~Tensor()
{
    if(_values)
        _mm_free(_values);  
}


// ------ TENSORS OPERATORS -------


// Copy operator with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
inline constexpr void Tensor<Dimensions...>::operator=(const Tensor<Dimensions...>& other)
{
    debug_assert(other._values && "Error in Tensor copy operator : an unitialized tensors is given as an argument.");

    if (this == &other)
		return;

    if(_values == nullptr)
        init();

    std::memcpy(this->_values, other._values, _size * sizeof(float));
}


// Copy operator with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
inline constexpr void Tensor<Dimensions...>::operator=(Tensor<Dimensions...>&& other) noexcept
{
    if(_values)
        _mm_free(_values);
   
    this->_values = other._values;

    other._values = nullptr;
}


// Element-wise addition with a float
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(float value) const &
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_FLOAT packedValues = _SET1_PS(value);

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _STORE(output._values + i, _ADD_PS(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _ADD_PS(packedValuesA, packedValues));

    }

    return output;
}


// Element-wise subtraction with a float
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(float value) const &
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_FLOAT packedValues = _SET1_PS(value);

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _STORE(output._values + i, _SUB_PS(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _SUB_PS(packedValuesA, packedValues));

    }

    return output;
}


// Element-wise multiplication with a float
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(float value) const &
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_FLOAT packedValues = _SET1_PS(value);

    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _STORE(output._values + i, _MUL(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValues)); // ------------------CHECK THIS------------------

    }

    return output;
}





// Element-wise division with a float
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(float value) const &
{
    Tensor<Dimensions...> output;

    output.init();

    const PACKAGE_FLOAT packedValues = _SET1_PS(value);

    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _STORE(output._values + i, _DIV(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
		const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

		_MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValues));

    }

    return output;
}


// Element-wise addition of a rvalue tensor with a float
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(float value) &&
{
    const PACKAGE_FLOAT packedValues = _SET1_PS(value);

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _STORE(this->_values + i, _ADD_PS(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _ADD_PS(packedValuesA, packedValues));

    }

    return *this;
}


// Element-wise subtraction of a rvalue tensor with a float
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(float value)&&
{
    const PACKAGE_FLOAT packedValues = _SET1_PS(value);

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _STORE(this->_values + i, _SUB_PS(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _SUB_PS(packedValuesA, packedValues));

    }

    return *this;
}


// Element-wise multiplication of a rvalue tensor with a float
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(float value)&&
{
    const PACKAGE_FLOAT packedValues = _SET1_PS(value);

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _STORE(this->_values + i, _MUL(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValues));

    }

    return *this;
}


// Element-wise division of a rvalue tensor with a float
template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(float value)&&
{
    const PACKAGE_FLOAT packedValues = _SET1_PS(value);

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _STORE(this->_values + i, _DIV(packedValuesA, packedValues));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValues));

    }

    return *this;
}

// Element-wise addition with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(const Tensor<Dimensions...>& tensor) const & 
{
    Tensor<Dimensions...> output;

    output.init();

    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(output._values + i, _ADD_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
		const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
		const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i );

		_MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _ADD_PS(packedValuesA, packedValuesB));

    }

    return output;
}


// Element-wise subtraction with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(const Tensor<Dimensions...>& tensor) const &
{
    Tensor<Dimensions...> output;

    output.init();

    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(output._values + i, _SUB_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _SUB_PS(packedValuesA, packedValuesB));

    }

    return output;
}


// Element-wise multiplication with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(const Tensor<Dimensions...>& tensor) const &
{
    Tensor<Dimensions...> output;

    output.init();

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(output._values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));

    }

    return output;
}


// Element-wise division with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(const Tensor<Dimensions...>& tensor) const &
{
    Tensor<Dimensions...> output;

    output.init();

    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(output._values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(output._values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));

    }

    return output;
}


// Element-wise addition with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(const Tensor<Dimensions...>& tensor) &&
{
    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _ADD_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _ADD_PS(packedValuesA, packedValuesB));

    }

    return *this;
}


// Element-wise subtraction with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(const Tensor<Dimensions...>& tensor) &&
{
    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _SUB_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _SUB_PS(packedValuesA, packedValuesB));

    }

    return *this;
}


// Element-wise multiplication with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(const Tensor<Dimensions...>& tensor) &&
{
    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));

    }

    return *this;
}


// Element-wise division with a Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(const Tensor<Dimensions...>& tensor) &&
{
    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));

    }

    return *this;
}


// Element-wise addition with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(Tensor<Dimensions...>&& tensor) const &
{
    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _ADD_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _ADD_PS(packedValuesA, packedValuesB));

    }

    return tensor;
}


// Element-wise subtraction with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(Tensor<Dimensions...>&& tensor) const &
{
    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _SUB_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _SUB_PS(packedValuesA, packedValuesB));

    }

    return tensor;
}


// Element-wise multiplication with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(Tensor<Dimensions...>&& tensor) const &
{
    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));

    }

    return tensor;
}


// Element-wise division with a rvalue Tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(Tensor<Dimensions...>&& tensor) const &
{
    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));

    }

    return tensor;
}


template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator+(Tensor<Dimensions...>&& tensor)&&
{
    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _ADD_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _ADD_PS(packedValuesA, packedValuesB));
    }

    return tensor;
}


template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator-(Tensor<Dimensions...>&& tensor)&&
{
    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _SUB_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _SUB_PS(packedValuesA, packedValuesB));

    }

    return tensor;
}


template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator*(Tensor<Dimensions...>&& tensor)&&
{
    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));

    }

    return tensor;
}


template<::std::size_t ...Dimensions>
constexpr Tensor<Dimensions...> Tensor<Dimensions...>::operator/(Tensor<Dimensions...>&& tensor)&&
{
    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(tensor._values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(tensor._values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));

    }

    return tensor;
}

// Element-wise addition of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator+=(const Tensor<Dimensions...>& tensor)
{
    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _ADD_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _ADD_PS(packedValuesA, packedValuesB));
    }
}


// Element-wise subtraction of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator-=(const Tensor<Dimensions...>& tensor)
{
    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _SUB_PS(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _SUB_PS(packedValuesA, packedValuesB));
    }
}


// Element-wise multiplication of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator*=(const Tensor<Dimensions...>& tensor)
{
    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _MUL(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _MUL(packedValuesA, packedValuesB));
    }
}


// Element-wise division of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
constexpr void Tensor<Dimensions...>::operator/=(const Tensor<Dimensions...>& tensor)
{
    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _STORE(this->_values + i, _DIV(packedValuesA, packedValuesB));
    }

    if constexpr (_offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + i);

        _MASKSTORE(this->_values + i, remainderMaskSI<_offset>(), _DIV(packedValuesA, packedValuesB));
    }
}

template<std::size_t ...Dimensions>
constexpr float& Tensor<Dimensions...>::operator()(std::size_t index) const
{
    debug_assert(index < _size && "Error in Tensor operator() : the given index is out of bounds.");

    return _values[index];

}

template<std::size_t ...Dimensions>
constexpr bool Tensor<Dimensions...>::operator==(const Tensor<Dimensions...>& tensor) const
{
    if (this == &tensor)
		return true;

	if (_size != tensor._size)
		return false;

    for (size_t i = 0; i < _size; ++i)
    {
		if (_values[i] != tensor._values[i])
			return false;
	}

	return true;
}


// ------ TENSORS BATCHES OPERATORS -------


// Element-wise addition with each tensors batches
template<std::size_t... Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator+(const Tensor<Dimensions..., batch_size>& tensor) const
{
    Tensor<Dimensions..., batch_size> output;

    output.init();

    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;
    
    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t i = 0;
        
        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(output._values + b * size + i, _ADD_PS(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(output._values + b * size + i, remainderMaskSI<offset>(), _ADD_PS(packedValuesA, packedValuesB));

        }
    }

    return output;
}


// Element-wise subtraction with each tensors batches
template<std::size_t... Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator-(const Tensor<Dimensions..., batch_size>& tensor) const
{
    Tensor<Dimensions..., batch_size> output;

    output.init();

    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t i = 0;

        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(output._values + b * size + i, _SUB_PS(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(output._values + b * size + i, remainderMaskSI<offset>(), _SUB_PS(packedValuesA, packedValuesB));

        }
    }

    return output;
}


// Element-wise multiplication with each tensors batches
template<std::size_t... Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator*(const Tensor<Dimensions..., batch_size>& tensor) const
{
    Tensor<Dimensions..., batch_size> output;

    output.init();

    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t i = 0;

        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(output._values + b * size + i, _MUL(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(output._values + b * size + i, remainderMaskSI<offset>(), _MUL(packedValuesA, packedValuesB));

        }
    }

    return output;
}


// Element-wise division with each tensors batches
template<std::size_t... Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator/(const Tensor<Dimensions..., batch_size>& tensor) const
{
    Tensor<Dimensions..., batch_size> output;

    output.init();

    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t i = 0;

        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(output._values + b * size + i, _DIV(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(output._values + b * size + i, remainderMaskSI<offset>(), _DIV(packedValuesA, packedValuesB));

        }
    }

    return output;
}


// Element-wise addition with each tensors batches rvalue Tensor
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator+(Tensor<Dimensions..., batch_size>&& tensor) const
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t i = 0;

        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(tensor._values + b * size + i, _ADD_PS(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(tensor._values + b * size + i, remainderMaskSI<offset>(), _ADD_PS(packedValuesA, packedValuesB));

        }
    }

    return tensor;
}


// Element-wise subtraction with each tensors batches rvalue Tensor
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator-(Tensor<Dimensions..., batch_size>&& tensor) const
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t i = 0;

        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(tensor._values + b * size + i, _SUB_PS(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(tensor._values + b * size + i, remainderMaskSI<offset>(), _SUB_PS(packedValuesA, packedValuesB));

        }
    }

    return tensor;
}


// Element-wise multiplication with each tensors batches rvalue Tensor
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator*(Tensor<Dimensions..., batch_size>&& tensor) const
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t i = 0;

        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(tensor._values + b * size + i, _MUL(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(tensor._values + b * size + i, remainderMaskSI<offset>(), _MUL(packedValuesA, packedValuesB));

        }
    }

    return tensor;
}


// Element-wise division with each tensors batches rvalue Tensor
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr Tensor<Dimensions..., batch_size> Tensor<Dimensions...>::operator/(Tensor<Dimensions..., batch_size>&& tensor) const
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t i = 0;
        
        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _STORE(tensor._values + b * size + i, _DIV(packedValuesA, packedValuesB));
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            _MASKSTORE(tensor._values + b * size + i, remainderMaskSI<offset>(), _DIV(packedValuesA, packedValuesB));

        }
    }

    return tensor;
}


// Element-wise addition of the tensor with another tensor of the same dimensions
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr void Tensor<Dimensions...>::operator+=(const Tensor<Dimensions..., batch_size>& tensor)
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        PACKAGE_FLOAT sum = packedValuesA;

        for (size_t b = 0; b < batch_size; ++b)
        {
			const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

			sum = _ADD_PS(sum, packedValuesB);
		}

        _STORE(this->_values + i, sum);
    }

    if constexpr (offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        PACKAGE_FLOAT sum = packedValuesA;

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            sum = _ADD_PS(sum, packedValuesB);
        }

        _MASKSTORE(this->_values + i, remainderMaskSI<offset>(), sum);
    }
}


// Element-wise subtraction of the tensor with each tensors batches
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr void Tensor<Dimensions...>::operator-=(const Tensor<Dimensions..., batch_size>& tensor)
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        PACKAGE_FLOAT sum = packedValuesA;

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            sum = _SUB_PS(sum, packedValuesB);
        }

        _STORE(this->_values + i, sum);
    }

    if constexpr (offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        PACKAGE_FLOAT sum = packedValuesA;

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            sum = _SUB_PS(sum, packedValuesB);
        }

        _MASKSTORE(this->_values + i, remainderMaskSI<offset>(), sum);
    }
}


// Sum of Element-wise multiplication of the tensor with each tensors batches
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr void Tensor<Dimensions...>::operator*=(const Tensor<Dimensions..., batch_size>& tensor)
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        PACKAGE_FLOAT sum = _SETZERO();

        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            const PACKAGE_FLOAT product = _MUL(packedValuesA, packedValuesB);

            sum = _ADD_PS(sum, product);
        }

        _STORE(this->_values + i, sum);
    }

    if constexpr (offset)
    {
        PACKAGE_FLOAT sum = _SETZERO();

        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            const PACKAGE_FLOAT product = _MUL(packedValuesA, packedValuesB);
            sum = _ADD_PS(sum, product);
        }

        _MASKSTORE(this->_values + i, remainderMaskSI<offset>(), sum);
    }
}


// Sum of Element - wise division of the tensor with each tensors batches
template<std::size_t ...Dimensions>
template<std::size_t batch_size>
constexpr void Tensor<Dimensions...>::operator/=(const Tensor<Dimensions..., batch_size>& tensor)
{
    constexpr size_t size = this->_size;
    constexpr uint16_t offset = this->_offset;

    size_t i = 0;

    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        PACKAGE_FLOAT sum = _SETZERO();

        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            const PACKAGE_FLOAT product = _DIV(packedValuesA, packedValuesB);

            sum = _ADD_PS(sum, product);
        }

        _STORE(this->_values + i, sum);
    }

    if constexpr (offset)
    {
        PACKAGE_FLOAT sum = _SETZERO();

        const PACKAGE_FLOAT packedValuesA = _LOAD(this->_values + i);

        for (size_t b = 0; b < batch_size; ++b)
        {
            const PACKAGE_FLOAT packedValuesB = _LOAD(tensor._values + b * size + i);

            const PACKAGE_FLOAT product = _DIV(packedValuesA, packedValuesB);
            sum = _ADD_PS(sum, product);
        }

        _MASKSTORE(this->_values + i, remainderMaskSI<offset>(), sum);
    }
}


// ------ TENSORS BASIC FUNCTIONS -------


// To print the tensor elements
template <std::size_t... Dimensions>
void print(const Tensor<Dimensions...>& tensor, float decimals)
{
    assert(tensor._values != nullptr && "Error: the given tensor is empty.");

    std::cout << std::fixed << std::setprecision(static_cast<int>(decimals));

    constexpr std::size_t dimensions[] = { Dimensions... };
    const size_t size = tensor._size;

    if constexpr (sizeof...(Dimensions) == 1) {
        // For 1D tensor (vector)
        for (std::size_t d = 0; d < size; ++d) {
            std::cout << tensor._values[d] << " ";
        }
    }
    else if constexpr (sizeof...(Dimensions) == 2) {
        // For 2D tensor (matrix)
        const size_t rows = dimensions[1];
        const size_t cols = dimensions[0];
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                std::cout << tensor._values[r * cols + c] << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        // For Higher-dimensional tensor
        const size_t rows = dimensions[1];
        const size_t cols = dimensions[0];
        for (std::size_t d = 0; d < size; ++d) {
            std::cout << tensor._values[d] << " ";

            if ((d + 1) % cols == 0) {
                std::cout << std::endl;
            }

            if ((d + 1) % (cols * rows) == 0 && d != size - 1) {
                std::cout << std::endl;
            }
        }
    }

    std::cout << std::defaultfloat;
}


// To show the shape of the tensor
template<std::size_t ...Dimensions>
void show_shape(const Tensor<Dimensions...>& tensor)
{
    ((std::cout << Dimensions << ' '), ...);
    std::cout << '\n';
}


// Compute the absolute value of each element in the tensor
template<std::size_t ...Dimensions>
Tensor<Dimensions...> abs(const Tensor<Dimensions...>& tensor)
{
    Tensor<Dimensions...> output;

    output.init();

    constexpr uint16_t offset = Tensor<Dimensions...>::_offset;
    constexpr uint16_t size = Tensor<Dimensions...>::_size;

    auto minus1 = _SET1_EPI32(-1);
    const PACKAGE_FLOAT absMask = _CASTSI_PS(_SRLI_EPI32(minus1, 1));

    size_t i = 0;
    
    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(tensor._values + i);

        _STORE(output._values + i, _AND(absMask, packedValues));// Clear the sign bit
    }

    if constexpr (offset)
    {
        const PACKAGE_FLOAT packedValues = _LOAD(tensor._values + i);

		_MASKSTORE(output._values + i, remainderMaskSI<offset>(), _AND(absMask, packedValues));
    }

    return output;
}


// Reshape the dimensions of the tensor to a compatible one
template<std::size_t ...Dimensions>
template<std::size_t ...newDimensions>
constexpr Tensor<newDimensions...> Tensor<Dimensions...>::reshape() const &
{
    static_assert(_size == (1 * ... * newDimensions), "Error in Tensor reshape : incorrect dimensions");

    return Tensor<newDimensions...>(*this);
}


// Reshape the dimensions of the tensor to a compatible one for rvalue tensor
template<::std::size_t ...otherDimensions>
template<std::size_t ...newDimensions>
constexpr Tensor<newDimensions...> Tensor<otherDimensions...>::reshape() &&
{
    static_assert(_size == (1 * ... * newDimensions), "Error in Tensor reshape : incorrect dimensions");

    Tensor<newDimensions...> output(std::move(*this));

    return output;
}


// Split the tensor into multiple tensors of the given dimensions
template<std::size_t ...Dimensions>
template<std::size_t ...splittedDimensions>
constexpr Tensor<splittedDimensions...>* Tensor<Dimensions...>::split()
{
    constexpr size_t splittedSize = Tensor<splittedDimensions...>::_size;
    constexpr size_t numSplits = this->_size / splittedSize;

    static_assert(this->_size % splittedSize == 0, "Error in Tensor split : incorrect dimensions");
    debug_assert(this->_values != nullptr && "Error in Tensor split : tensor not initialized");

    Tensor<splittedDimensions...>* splittedTensors = new Tensor<splittedDimensions...>[numSplits];
    
    for (size_t i = 0; i < numSplits; ++i)
    {
        splittedTensors[i]._values = this->_values + i * splittedSize;
    }

    return splittedTensors;
}

// Flatten the tensor to a one dimension tensor
template<std::size_t ...Dimensions>
constexpr Tensor<(1 * ... * Dimensions)> Tensor<Dimensions...>::flatten() const &
{
    if constexpr (sizeof...(Dimensions) == 1)
        return *this;

    return Tensor<_size>(*this);   
}

// Flatten the tensor to a one dimension tensor for rvalue tensor
template<std::size_t ...Dimensions>
constexpr Tensor<(1 * ... * Dimensions)> Tensor<Dimensions...>::flatten() &&
{
    if constexpr (sizeof...(Dimensions) == 1)
        return *this;

    return Tensor<_size>(std::move(*this));
}


// Compute the sum of each values in the tensor
template<std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::sum() const
{
    PACKAGE_FLOAT packedSum = _LOAD(_values);

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        size_t i = PACKAGE_LENGTH;

        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValues = _LOAD(_values + i);

            packedSum = _ADD_PS(packedValues, packedSum);
        }

        if constexpr (_offset)
        {
            const PACKAGE_FLOAT packedValues = _LOAD(_values + i);

            packedSum = _ADD_PS( _AND(packedValues, remainderMask<_offset>()), packedSum);
        }
    }
    else
    {
        packedSum = _AND(packedSum, remainderMask<_size>());
    }
       
    const float sum = _HSUM(packedSum);

    return sum;
}


// Find the index of the maximum value in the tensor
// ref : https://en.algorithmica.org/hpc/algorithms/argmin/
template<std::size_t ...Dimensions>
constexpr size_t Tensor<Dimensions...>::argmax() const
{
    size_t argmax = 0;
    float max = 0;

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        max = _HMAX(_LOAD(_values));
        
        PACKAGE_FLOAT maxValues = _SET1_PS(max);

        size_t i = PACKAGE_LENGTH;

        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValues = _LOAD(_values + i);

            maxValues = _MAX_PS(packedValues, maxValues);

            auto compMask = _CMPGT_EPI32(_CASTPS_SI(maxValues), _CASTPS_SI(packedValues));

            if (!_TESTZSI(compMask, compMask)) [[unlikely]]
            {
                max = _HMAX(maxValues);
                argmax = i;

                maxValues = _SET1_PS(max);
            }
				
        }

        if constexpr (_offset)
        {

            for (; i < _size; ++i)
            {
                if (_values[i] > max)
                {
                    max = _values[i];
                    argmax = i;

                    for (size_t r = i+1; r < _size; ++r)
                    {
                        argmax = (_values[r] > _values[argmax]) ? r : argmax;
                    }

                    return argmax;
                }
            }
        }
    }
    else
	{
        max = _values[0];
         
        for (uint16_t i = 1; i < _size; ++i)
        {
            if (_values[i] > max)
            {
                max = _values[i];
                argmax = i;
            }
        }

        return argmax;
	}

    
    for (size_t i = argmax; i < argmax + PACKAGE_LENGTH; ++i)
    {
        if (_values[i] == max)
            return i;
	}

	return -1;
}


// Find the maximum value in the tensor
template<std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::max() const
{
    PACKAGE_FLOAT maxValues = _LOAD(_values);

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        size_t i = PACKAGE_LENGTH;

        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValues = _LOAD(_values + i);

            maxValues = _MAX_PS(packedValues, maxValues);
        }

        if constexpr (_offset)
        {
            PACKAGE_FLOAT minimum = _SET1_PS(-FLT_MAX);
            constexpr int mask = (1 << _offset) - 1;

            const PACKAGE_FLOAT packedValues = _LOAD(_values + i);

            maxValues = _MAX_PS(_BLEND(minimum, packedValues, mask), maxValues);
        }
    }
    else
    {
        PACKAGE_FLOAT minimum = _SET1_PS(-FLT_MAX);
        constexpr int mask = (1 << _size) - 1;

        maxValues = _BLEND(minimum, maxValues, mask);
    }
       

    float max = _HMAX(maxValues);

    return max;
}


// Compute the mean of the values in the tensor
template<std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::mean() const
{
    const float mean = sum() / (float)_size;

    return mean;
}


// Compute the variance of the values in the tensor based on a given mean
template<std::size_t ...Dimensions>
constexpr float Tensor<Dimensions...>::variance(float mean) const
{
    const PACKAGE_FLOAT packedMean = _SET1_PS(mean);
    
    PACKAGE_FLOAT deviation = _SUB_PS(_LOAD(_values), packedMean);

    if constexpr (_size >= PACKAGE_LENGTH)
    {
        size_t i = PACKAGE_LENGTH;
        
        UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= _size; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packedValues = _LOAD(_values + i);

            deviation = _ADD_PS( _SUB_PS(packedValues, packedMean), deviation);
        }

        if constexpr (_offset)
        {
            const PACKAGE_FLOAT packedValues = _SUB_PS( _LOAD(_values + i), packedMean);

            const PACKAGE_FLOAT maskedValues = _AND(packedValues, remainderMask<_offset>());

            deviation = _ADD_PS( maskedValues, deviation);
        }
    }
    else
    {
        deviation = _AND( deviation, remainderMask<_size>());
    }
        

    const float variance = _HSUM( _MUL(deviation, deviation)) / (float) _size;

    return variance;
}


// ------ TENSORS MATRIX OPERATIONS -------

#ifdef __FMA__
// Element-wise addition between tensorC and the element-wise multiplication of tensorA and tensorB
// @return tensorA * tensorB + tensorC
template<std::size_t ...Dimensions>
Tensor<Dimensions...> multiply_and_add(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC)
{
    Tensor<Dimensions...> output;
    output.init();

    constexpr size_t size = output._size;
    constexpr uint16_t offset = output._offset;

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(tensorA._values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensorB._values + i);
        const PACKAGE_FLOAT packedValuesC = _LOAD(tensorC._values + i);

        const PACKAGE_FLOAT result = _FMADD(packedValuesA, packedValuesB, packedValuesC);

        _STORE(output._values + i, result);
    }

    if constexpr (offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(tensorA._values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensorB._values + i);
        const PACKAGE_FLOAT packedValuesC = _LOAD(tensorC._values + i);

        const PACKAGE_FLOAT result = _FMADD(packedValuesA, packedValuesB, packedValuesC);

        _MASKSTORE(output._values + i, remainderMaskSI<offset>(), result);
    }

    return output;
}


// Element-wise subtraction between tensorC and the element-wise multiplication of tensorA and tensorB
// @return tensorA * tensorB - tensorC
template<std::size_t ...Dimensions>
Tensor<Dimensions...> multiply_and_sub(const Tensor<Dimensions...>& tensorA, const Tensor<Dimensions...>& tensorB, const Tensor<Dimensions...>& tensorC)
{
    Tensor<Dimensions...> output;
    output.init();

    constexpr size_t size = output._size;
    constexpr uint16_t offset = output._offset;

    size_t i = 0;

    UNROLL_LOOP(Tensor<Dimensions...>::UNROLL_FACTOR)
    for (; i + PACKAGE_LENGTH <= size; i += PACKAGE_LENGTH)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(tensorA._values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensorB._values + i);
        const PACKAGE_FLOAT packedValuesC = _LOAD(tensorC._values + i);

        const PACKAGE_FLOAT result = _FMSUB(packedValuesA, packedValuesB, packedValuesC);

        _STORE(output._values + i, result);
    }

    if constexpr (offset)
    {
        const PACKAGE_FLOAT packedValuesA = _LOAD(tensorA._values + i);
        const PACKAGE_FLOAT packedValuesB = _LOAD(tensorB._values + i);
        const PACKAGE_FLOAT packedValuesC = _LOAD(tensorC._values + i);

        const PACKAGE_FLOAT result = _FMSUB(packedValuesA, packedValuesB, packedValuesC);

        _MASKSTORE(output._values + i, remainderMaskSI<offset>(), result);
    }

    return output;
}

#endif


// Transpose the tensor
template <std::size_t cols, std::size_t rows, std::size_t... rest>
Tensor<rows, cols, rest...> transpose(const Tensor<cols, rows, rest...>& tensor)
{
    Tensor<rows, cols, rest...> output;
    output.init();

    constexpr std::size_t colsA = cols;
    constexpr std::size_t rowsA = rows;

    constexpr std::size_t colsB = rows;
    constexpr std::size_t rowsB = cols;

    constexpr std::uint16_t offColsA = colsA % PACKAGE_LENGTH;
    constexpr std::uint16_t offRowsA = rowsA % PACKAGE_LENGTH;

    float* valuesA = tensor._values;
    float* valuesB = output._values;

    std::size_t r = 0;

    
    for (; r + PACKAGE_LENGTH <= rowsA; r += PACKAGE_LENGTH)
    {
        std::size_t c = 0;

        for (; c + PACKAGE_LENGTH <= colsA; c += PACKAGE_LENGTH)
        {
            PACKAGE_FLOAT rowArray[PACKAGE_LENGTH];

            for (std::uint16_t i = 0; i < PACKAGE_LENGTH; ++i)
            {
				rowArray[i] = _LOAD(tensor._values + (r + i) * colsA + c);

            }

            _TRANSPOSE(rowArray);

            for(std::uint16_t i = 0; i < PACKAGE_LENGTH; ++i)
			{
                _STORE(output._values + (c + i) * colsB + r, rowArray[i]);
			}
        }

        if constexpr (offColsA)
        {
            PACKAGE_FLOAT rowArray[PACKAGE_LENGTH];

            for (std::uint16_t i = 0; i < PACKAGE_LENGTH; ++i)
            {
                rowArray[i] = _LOAD(tensor._values + (r + i) * colsA + c);
            }

            _TRANSPOSE(rowArray);

            for (std::uint16_t i = 0; i < offColsA; ++i)
            {
                _STORE(output._values + (c + i) * colsB + r, rowArray[i]);
            }
        }
    }

    if constexpr (offRowsA)
    {
        std::size_t c = 0;

        
        for (; c + PACKAGE_LENGTH <= colsA; c += PACKAGE_LENGTH)
        {
            PACKAGE_FLOAT rowArray[PACKAGE_LENGTH];

            for (std::uint16_t i = 0; i < offRowsA; ++i)
            {
                rowArray[i] = _LOAD(tensor._values + (r + i) * colsA + c);
            }


            _TRANSPOSE(rowArray);

            for (std::uint16_t i = 0; i < PACKAGE_LENGTH; ++i)
            {
                _MASKSTORE(output._values + (c + i) * colsB + r, remainderMaskSI<offRowsA>(), rowArray[i]);
            }
        }

        if constexpr (offColsA)
        {
            PACKAGE_FLOAT rowArray[PACKAGE_LENGTH];

            for (std::uint16_t i = 0; i < offRowsA; ++i)
            {
                rowArray[i] = _LOAD(tensor._values + (r + i) * colsA + c);
            }

            _TRANSPOSE(rowArray);

            for (std::uint16_t i = 0; i < offColsA; ++i)
            {
                _MASKSTORE(output._values + (c + i) * colsB + r, remainderMaskSI<offRowsA>(), rowArray[i]);
            }
        }
    }
   
    return output;
}


// ------ TENSORS MATRIX MULTIPLICATION -------


// Matrix multiplication between tensorA and tensorB
template<std::size_t colsA, std::size_t rowsA, std::size_t colsB, std::size_t... rest>
Tensor<colsB, rowsA, rest...> mul(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsB, colsA, rest...>& tensorB)
{
    constexpr std::size_t COLS_A_UNROLL_FACTOR = (colsA >= 16) ? 16 :
                                                 (colsA >= 8) ? 8 :
                                                 (colsA >= 4) ? 4 : 1;
    constexpr auto offCols = colsB % PACKAGE_LENGTH;
    constexpr std::size_t outerLength = (1 * ... * rest);

    const auto mask = remainderMaskSI<offCols>();

    Tensor<colsB, rowsA, rest...> output;

    output.init();
    
    const float* iterA = tensorA._values;
    const float* iterB;

    float* iterO = output._values;

    
    for (std::size_t l = 0; l < outerLength; ++l)
    {
        for (std::size_t r = 0; r < rowsA; ++r)
        {
            std::size_t c = 0;

            for (; c + PACKAGE_LENGTH <= colsB; c += PACKAGE_LENGTH)
            {
                iterB = tensorB._values + c + l * colsB * colsA;

                PACKAGE_FLOAT sum = _MUL(_LOAD1(iterA), _LOAD(iterB));

                UNROLL_LOOP(COLS_A_UNROLL_FACTOR)
                for (std::size_t i = 1; i < colsA; ++i)
                {
                    const PACKAGE_FLOAT packageA = _LOAD1(iterA + i);
                    const PACKAGE_FLOAT packageB = _LOAD(iterB + i * colsB);
                    
                    #ifdef __FMA__
                        sum = _FMADD(packageA, packageB, sum);
                    #else
                        sum = _ADD_PS(_MUL(packageA, packageB), sum);
                    #endif
                }

                _STORE(iterO + c, sum);
            }

            if constexpr (offCols)
            {
                iterB = tensorB._values + c + l * colsB * colsA;

                PACKAGE_FLOAT sum = _MUL(_LOAD1(iterA), _LOAD(iterB));

                UNROLL_LOOP(COLS_A_UNROLL_FACTOR)
                for (std::size_t i = 1; i < colsA; ++i)
                {
                    const PACKAGE_FLOAT packageA = _LOAD1(iterA + i);
                    const PACKAGE_FLOAT packageB = _LOAD(iterB + i * colsB);

                    #ifdef __FMA__
                        sum = _FMADD(packageA, packageB, sum);
                    #else
                        sum = _ADD_PS(_MUL(packageA, packageB), sum);
                    #endif
                }

                _MASKSTORE(iterO + c, mask, sum);
            }

            iterA += colsA;
            iterO += colsB;
        }
    }

    return output;
}


// Matrix multiplication between tensorA and the transpose of tensorB
template<std::size_t colsA, std::size_t rowsA, std::size_t colsB, std::size_t... rest>
Tensor<colsB, rowsA, rest...> mul_transposed(const Tensor<colsA, rowsA, rest...>& tensorA, const Tensor<colsA, colsB, rest...>& tensorB)
{
    constexpr std::size_t COLS_A_UNROLL_FACTOR = (colsA >= 17 * PACKAGE_LENGTH) ? 16 :
                                                 (colsA >= 9 * PACKAGE_LENGTH) ? 8 :
                                                 (colsA >= 5 * PACKAGE_LENGTH) ? 4 : 1;

    constexpr auto offcols = colsA % PACKAGE_LENGTH;
    constexpr std::size_t outerLenght = (1 * ... * rest);

    const auto mask = remainderMaskSI<offcols>();

    Tensor<colsB, rowsA, rest...> output;

    output.init();

    float* iterA;
    float* iterB;

    float* iterO = output._values;

    
    for (std::size_t l = 0; l < outerLenght; ++l)
    {
        for (std::size_t r = 0; r < rowsA; ++r)
        {
            for (std::size_t c = 0; c < colsB; ++c)
            {
                iterA = tensorA._values + colsA * r + colsA * rowsA * l;
                iterB = tensorB._values + colsA * c + colsA * colsB * l;

                PACKAGE_FLOAT sum;

                if constexpr (colsA >= PACKAGE_LENGTH)
                {
                    sum = _MUL(_LOAD(iterA), _LOAD(iterB));

                    std::size_t i = PACKAGE_LENGTH;

                    UNROLL_LOOP(COLS_A_UNROLL_FACTOR)
                    for (; i + PACKAGE_LENGTH <= colsA; i += PACKAGE_LENGTH)
                    {
                        const PACKAGE_FLOAT packageA = _LOAD(iterA + i);
                        const PACKAGE_FLOAT packageB = _LOAD(iterB + i);

                        #ifdef __FMA__
                            sum = _FMADD(packageA, packageB, sum);
                        #else
                            sum = _ADD_PS(_MUL(packageA, packageB), sum);
                        #endif
                    }

                    if constexpr (offcols)
                    {
                        const PACKAGE_FLOAT packageA = _MASKLOAD(iterA + i, mask);
                        const PACKAGE_FLOAT packageB = _MASKLOAD(iterB + i, mask);

                        #ifdef __FMA__
                            sum = _FMADD(packageA, packageB, sum);
                        #else
                            sum = _ADD_PS(_MUL(packageA, packageB), sum);
                        #endif
                    }
                }
                else
                {
                    const PACKAGE_FLOAT packageA = _LOAD(iterA);
                    const PACKAGE_FLOAT packageB = _LOAD(iterB);

                    sum = _AND(_MUL(packageA, packageB), remainderMask<colsA>());
                }

                *iterO = _HSUM(sum);

                ++iterO;
            }
        }
    }

    return output;
}


// Matrix multiplication between tensorA and the transpose of tensorB as a scalar
template<std::size_t colsA, std::size_t rowsA>
Tensor<rowsA> mul_b_transposed_scalar(const Tensor<colsA, rowsA>& tensorA, const Tensor<colsA>& tensorB)
{
    constexpr std::size_t COLS_A_UNROLL_FACTOR = (colsA >= 17 * PACKAGE_LENGTH) ? 16 :
                                                 (colsA >= 9 * PACKAGE_LENGTH) ? 8 :
                                                 (colsA >= 5 * PACKAGE_LENGTH) ? 4 : 1;

    constexpr auto offCols = colsA % PACKAGE_LENGTH;

    const auto mask = remainderMaskSI<offCols>();

    Tensor<rowsA> output;

    output.init();

    float* iterA;
    float* iterB = tensorB._values;

    float* iterO = output._values;

    
    for (std::size_t r = 0; r < rowsA; ++r)
    {
        iterA = tensorA._values + r * colsA;

        PACKAGE_FLOAT sum;

        if constexpr (colsA >= PACKAGE_LENGTH)
        {
            sum = _MUL( _LOAD(iterA), _LOAD(iterB));

            std::size_t i = PACKAGE_LENGTH;

            UNROLL_LOOP(COLS_A_UNROLL_FACTOR)
            for (; i + PACKAGE_LENGTH <= colsA; i += PACKAGE_LENGTH)
            {
                const PACKAGE_FLOAT packageA = _LOAD(iterA + i);
                const PACKAGE_FLOAT packageB = _LOAD(iterB + i);

                #ifdef __FMA__
                    sum = _FMADD(packageA, packageB, sum);
                #else
                    sum = _ADD_PS(_MUL(packageA, packageB), sum);
                #endif
            }

            if constexpr (offCols)
            {
                const PACKAGE_FLOAT packageA = _MASKLOAD(iterA + i, mask);
                const PACKAGE_FLOAT packageB = _MASKLOAD(iterB + i, mask);

                #ifdef __FMA__
                    sum = _FMADD(packageA, packageB, sum);
                #else
                    sum = _ADD_PS(_MUL(packageA, packageB), sum);
                #endif
            }
        }
        else
        {
            const PACKAGE_FLOAT packageA = _LOAD(iterA);
            const PACKAGE_FLOAT packageB = _LOAD(iterB);

            sum = _AND( _MUL(packageA, packageB), remainderMask<colsA>());
        }

        *iterO = _HSUM(sum);

        ++iterO;
    }

    return output;
}


// Matrix multiplication between the transpose of tensorA and tensorB both as a scalar
template<std::size_t colsA, std::size_t colsB>
Tensor<colsB, colsA> mul_transposed_scalar(const Tensor<colsA>& tensorA, const Tensor<colsB>& tensorB)
{
    constexpr std::size_t COLS_B_UNROLL_FACTOR = (colsB >= 16 * PACKAGE_LENGTH) ? 16 :
                                                 (colsB >= 8 * PACKAGE_LENGTH) ? 8 :
                                                 (colsB >= 4 * PACKAGE_LENGTH) ? 4 : 1;

    constexpr auto offset = colsB % PACKAGE_LENGTH;

    const auto mask = remainderMaskSI<offset>();

    Tensor<colsB, colsA> output;

    output.init();

    
    for (std::size_t c = 0; c < colsA; ++c)
    {
        PACKAGE_FLOAT packageA = _LOAD1(tensorA._values + c);

        std::size_t i = 0;

        UNROLL_LOOP(COLS_B_UNROLL_FACTOR)
        for (; i + PACKAGE_LENGTH <= colsB; i += PACKAGE_LENGTH)
        {
            const PACKAGE_FLOAT packageB = _LOAD(tensorB._values + i);

            const PACKAGE_FLOAT result = _MUL(packageA, packageB);

            _STORE(output._values + i + c * colsB, result);
        }

        if constexpr (offset)
        {
            const PACKAGE_FLOAT packageB = _LOAD(tensorB._values + i);

            const PACKAGE_FLOAT result = _MUL(packageA, packageB);

            _MASKSTORE(output._values + i + c * colsB, mask, result);
        }
    }

    return output;

}
