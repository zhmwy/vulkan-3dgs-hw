/* ========================================================================= *
 *                                                                           *
 *                                 SMPL++                                    *
 *                    Copyright (c) 2018, Chongyi Zheng.                     *
 *                          All Rights reserved.                             *
 *                                                                           *
 * ------------------------------------------------------------------------- *
 *                                                                           *
 * This software implements a 3D human skinning model - SMPL: A Skinned      *
 * Multi-Person Linear Model with C++.                                       *
 *                                                                           *
 * For more detail, see the paper published by Max Planck Institute for      *
 * Intelligent Systems on SIGGRAPH ASIA 2015.                                *
 *                                                                           *
 * We provide this software for research purposes only.                      *
 * The original SMPL model is available at http://smpl.is.tue.mpg.           *
 *                                                                           *
 * ========================================================================= */

//=============================================================================
//
//  CLASS TorchEx DECLARATIONS AND IMPLEMENTATIONS
//
//=============================================================================

#ifndef TORCH_EXTRA_H
#define TORCH_EXTRA_H


#include <Eigen/Dense>
//----------
#include "toolbox/Exception.h"

namespace smpl {


class TorchEx final
{

private: // PRIVATE ATTRIBUTES

    // %% Constructor and Destructor %%
    TorchEx() noexcept(false) ;
    TorchEx(const TorchEx &other) noexcept(false);
    ~TorchEx() noexcept(false);


    // %% Operators %%
    TorchEx &operator=(const TorchEx &TorchEx) noexcept(false);

    // %% Extended Libraries %%
    template <class Array, class ... Rest>
    static
    Eigen::MatrixXd indexing_impl(Eigen::MatrixXd &self, 
        int64_t layer, Array index, Rest ... indices) noexcept(false);

    template <class Array>
    static
    Eigen::MatrixXd indexing_impl(Eigen::MatrixXd &self, 
        int64_t layer, Array index) noexcept(false);

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES

    // %% Extended Libraries %%
    template <class ... Arrays>
    static
    Eigen::MatrixXd indexing(Eigen::MatrixXd &self,
        Arrays ... indices) noexcept(false);
};

template <class ... Arrays>
Eigen::MatrixXd TorchEx::indexing(Eigen::MatrixXd &self, 
    Arrays ... indices) noexcept(false)
{
    Eigen::MatrixXd out;
    try {
        out = indexing_impl(self, 0, indices ...);
    }
    catch(std::exception &e) {
        throw;
    }

    return out;
}


template <class Array, class ... Rest>
Eigen::MatrixXd TorchEx::indexing_impl(Eigen::MatrixXd &self, 
    int64_t layer, Array index, Rest ... indices) noexcept(false)
{
    if (std::is_same<Array, std::vector<int64_t>>::value) {
        Eigen::MatrixXd slice;
        Eigen::MatrixXd out;
        
        // 根据 index 的大小来判断如何切片
        switch (index.size()) {
            case 0:
                slice = self; // layer 维度可以调整这里
                out = indexing_impl(slice, layer + 1, indices ...);
                break;
            case 1:
                slice = self.block(0, *(index.begin()), self.rows(), 1); // 切取某一列
                out = indexing_impl(slice, layer, indices ...);
                break;
            case 2:
                slice = self.block(*(index.begin()), 0, *(index.begin() + 1) - *(index.begin()), self.cols()); // 切取某一行
                out = indexing_impl(slice, layer + 1, indices ...);
                break;
            case 3:
                slice = self.block(*(index.begin()), *(index.begin() + 1), *(index.begin() + 2), self.cols()); // 取三维子块
                out = indexing_impl(slice, layer + 1, indices ...);
                break;
            default:
                throw std::runtime_error("Invalid integer list for recursive indexing!");
                break;
        }
        return out;
    }
    else {
        throw std::runtime_error("Integer list type mismatch in recursion!");
    }

    return Eigen::MatrixXd(); // 递归结束后返回的空矩阵
}

template <class Array>
Eigen::MatrixXd TorchEx::indexing_impl(Eigen::MatrixXd &self, 
    int64_t layer, Array index) noexcept(false)
{
    if (std::is_same<Array, std::vector<int64_t>>::value) {
        Eigen::MatrixXd out;

        // 根据 index 的大小来判断如何切片
        switch (index.size()) {
            case 0:
                out = self; // 默认返回整个矩阵
                break;
            case 1:
                // 切取某一列
                out = self.block(0, *(index.begin()), self.rows(), 1); 
                break;
            case 2:
                // 切取某一行
                out = self.block(*(index.begin()), 0, 1, self.cols()); 
                break;
            case 3:
                // 切取三维子块，假设是二维矩阵的子集
                out = self.block(*(index.begin()), *(index.begin() + 1), 1, self.cols());
                break;
            default:
                throw std::runtime_error("Invalid integer list for basic indexing!");
                break;
        }
        return out;
    }
    else {
        throw std::runtime_error("Integer list type mismatch in base!");
    }

    return Eigen::MatrixXd(); // 如果类型不匹配，返回空矩阵
}


//=============================================================================
}
//=============================================================================
#endif // TORCH_EXTRA_H
//=============================================================================
