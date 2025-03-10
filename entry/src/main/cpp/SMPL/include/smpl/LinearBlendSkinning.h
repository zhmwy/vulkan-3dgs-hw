
#ifndef LINEAR_BLEND_SKINNING_H
#define LINEAR_BLEND_SKINNING_H

#include <Eigen/Dense>

#include "definition/def.h"
#include "toolbox/Exception.h"
#include "Eigen/Dense"
#include <iostream>  

namespace smpl {
class LinearBlendSkinning final
{
private: // PRIVATE ATTRIBUTES

    Eigen::MatrixXd m__restShape;
    Eigen::MatrixXd m__transformation;
    
    Eigen::MatrixXd m__posedVert;

protected: // PROTECTED ATTRIBUTES

public: // PUBLIC ATTRIBUTES
Eigen::MatrixXd m__weights;

private: // PRIVATE METHODS

    // %% Linear Blend Skinning %%
    Eigen::MatrixXd cart2homo(Eigen::MatrixXd &cart) noexcept(false);
    Eigen::MatrixXd homo2cart(Eigen::MatrixXd &homo) noexcept(false);

protected: // PROTECTED METHODS

public: // PUBLIC METHODS

    // %% Constructor and Destructor %%
    LinearBlendSkinning() noexcept(true);
    LinearBlendSkinning(Eigen::MatrixXd &weights) noexcept(false); // 去掉了 torch::Device
    LinearBlendSkinning(const LinearBlendSkinning &linearBlendSkinning) noexcept(false);
    ~LinearBlendSkinning() noexcept(true);

    // %% Operator %%
    LinearBlendSkinning &operator=(const LinearBlendSkinning &linearBlendSkinning) noexcept(false);

    // %% Setter and Getter %%
    void setWeight(const Eigen::MatrixXd &weights) noexcept(false);
    void setRestShape(const Eigen::MatrixXd &restShape) noexcept(false);
    void setTransformation(const Eigen::MatrixXd &transformation) noexcept(false);

    Eigen::MatrixXd getVertex() noexcept(false);

    // %% Linear Blend Skinning %%
    void skinning() noexcept(false);
};


//=============================================================================
} // namespace smpl
//=============================================================================
#endif // LINEAR_BLEND_SKINNING_H
//=============================================================================
