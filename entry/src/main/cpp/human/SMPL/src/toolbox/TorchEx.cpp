#include "toolbox/TorchEx.hpp"

namespace smpl {


TorchEx::TorchEx() noexcept(false)
{
    throw smpl_error("TorchEx", "Cannot call the constructor!");
}

TorchEx::TorchEx(const TorchEx &TorchEx) noexcept(false)
{
    throw smpl_error("TorchEx", "Cannot call the copy constructor!");
}

TorchEx::~TorchEx() noexcept(false)
{
    throw smpl_error("TorchEx", "Cannot call the Destructor!");
}

TorchEx &TorchEx::operator=(const TorchEx &TorchEx) noexcept(false)
{
    throw smpl_error("TorchEx", "Cannot not call the assignment operator!");
}

}