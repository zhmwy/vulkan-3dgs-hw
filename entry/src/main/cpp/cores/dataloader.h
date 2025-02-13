#pragma once
#include <iostream>
#include <vector>
#include "tinyply.h"
#include <glm/glm.hpp>
#include "vulkanexamplebase.h"
#define VULKAN_HW
// #define VULKAN_PC

using namespace tinyply;



// DataLoader Tool Class
class DataLoader
{
public:
    // PlyDataLoader Test
    struct PlyDataStruct {
        std::vector<glm::vec3> xyz;
        std::vector<glm::vec3> scale;
        std::vector<glm::vec4> rotate;
        void print()const
        {
            std::cout << "Loaded " << xyz.size() << " vertices." << std::endl;

            // Output the data of the first vertex
            if (!xyz.empty()) {
                std::cout << "First vertex position: (" << xyz[0].x << ", " << xyz[0].y << ", " << xyz[0].z << ")" << std::endl;
                std::cout << "First vertex scale: (" << scale[0].x << ", " << scale[0].y << ", " << scale[0].z << ")" << std::endl;
                std::cout << "First vertex rotation: (" << rotate[0].x << ", " << rotate[0].y << ", " << rotate[0].z << ", " << rotate[0].w << ")" << std::endl;
            }
        }
    };

    struct InputGaussianStruct {
        std::vector<glm::vec3> positions;      // 3D positions (x, y, z)
        std::vector<glm::vec3> scales;         // Scaling factors (scale_x, scale_y, scale_z)
        std::vector<glm::vec4> rotations;      // Rotation (rotate_0, rotate_1, rotate_2, rotate_3)
        std::vector<float> opacities;          // Opacity
        std::vector<glm::vec3> colors;         // RGB colors (r, g, b)

        // Print the first vertex's data for verification
        void print() const {
            std::cout << "Loaded " << positions.size() << " vertices." << std::endl;
            if (!positions.empty()) {
                std::cout << "\nFirst vertex data:" << std::endl;

                // Print position (x, y, z)
                std::cout << "Position (x, y, z): (" << positions[0].x << ", " << positions[0].y << ", " << positions[0].z << ")" << std::endl;

                // Print scale (scale_x, scale_y, scale_z)
                std::cout << "Scale (x, y, z): (" << scales[0].x << ", " << scales[0].y << ", " << scales[0].z << ")" << std::endl;

                // Print rotation (as quaternion)
                std::cout << "Rotation (quaternion): (" << rotations[0].x << ", " << rotations[0].y << ", " << rotations[0].z << ", " << rotations[0].w << ")" << std::endl;

                // Print opacity
                std::cout << "Opacity: " << opacities[0] << std::endl;

                // Print color (r, g, b)
                std::cout << "Color (r, g, b): (" << colors[0].r << ", " << colors[0].g << ", " << colors[0].b << ")" << std::endl;
            }
        }
    };


    // Input and output of ProjectEWAForward
    struct ProjectEWAForwardStruct {
        std::vector<glm::vec3> xyz;
        std::vector<glm::vec3> scale;
        std::vector<glm::vec4> rotate;
        std::vector<bool> vis_input;
        std::vector<glm::vec3> uvd;
        std::vector<glm::vec3> cov2d;
        std::vector<glm::vec3> cov3d; //double size
        std::vector<int> radius;
        std::vector<int> tiles_torched;
        std::vector<glm::vec4> tiles_rect;
        std::vector<bool> vis_output;

        void print() const {
            std::cout << "Loaded " << xyz.size() << " vertices." << std::endl;

            // �����һ���������������
            if (!xyz.empty()) {
                std::cout << "\nFirst vertex data:" << std::endl;

                std::cout << "Position (xyz): (" << xyz[0].x << ", " << xyz[0].y << ", " << xyz[0].z << ")" << std::endl;
                std::cout << "Scale (scale): (" << scale[0].x << ", " << scale[0].y << ", " << scale[0].z << ")" << std::endl;
                std::cout << "Rotation (rotate): (" << rotate[0].x << ", " << rotate[0].y << ", " << rotate[0].z << ", " << rotate[0].w << ")" << std::endl;

                std::cout << "Visibility input (vis_input): " << vis_input[0] << std::endl;

                std::cout << "UV coordinates and depth (uvd): (" << uvd[0].x << ", " << uvd[0].y << ", " << uvd[0].z << ")" << std::endl;

                std::cout << "2D covariance (cov2d): (" << cov2d[0].x << ", " << cov2d[0].y << ", " << cov2d[0].z << ")" << std::endl;

                std::cout << "3D covariance (cov3d): ("
                    << cov3d[0].x << ", " << cov3d[0].y << ", " << cov3d[0].z << ", "
                    << cov3d[1].x << ", " << cov3d[1].y << ", " << cov3d[1].z << ")" << std::endl;

                std::cout << "Radius (radius): " << radius[0] << std::endl;

                std::cout << "Tiles torched (tiles_torched): " << tiles_torched[0] << std::endl;

                std::cout << "Tiles rectangle (tiles_rect): ("
                    << tiles_rect[0].x << ", " << tiles_rect[0].y << ", "
                    << tiles_rect[0].z << ", " << tiles_rect[0].w << ")" << std::endl;

                std::cout << "Visibility output (vis_output): " << vis_output[0] << std::endl;
            }
        }

    };

    struct ProjectEWABackwardStruct {
        std::vector<glm::vec3> xyz;
        std::vector<glm::vec3> scale;
        std::vector<glm::vec4> rotate;
        std::vector<bool> visible;
        std::vector<glm::vec3> cov3d;       // Double size as it represents 6 floats
        std::vector<glm::vec3> dL_duv;      // For dL_du, dL_dv, dL_dd
        std::vector<glm::vec3> dL_dcov2d;
        std::vector<glm::vec3> dL_dxyz;
        std::vector<glm::vec3> dL_dscale;
        std::vector<glm::vec4> dL_drotate;

        void print() const {
            std::cout << "Loaded " << xyz.size() << " vertices." << std::endl;
            if (!xyz.empty()) {
                std::cout << "\nFirst vertex data:" << std::endl;
                std::cout << "Position (xyz): (" << xyz[0].x << ", " << xyz[0].y << ", " << xyz[0].z << ")" << std::endl;
                std::cout << "Scale (scale): (" << scale[0].x << ", " << scale[0].y << ", " << scale[0].z << ")" << std::endl;
                std::cout << "Rotation (rotate): (" << rotate[0].x << ", " << rotate[0].y << ", " << rotate[0].z << ", " << rotate[0].w << ")" << std::endl;
                std::cout << "Visible (visible): " << visible[0] << std::endl;
                std::cout << "3D Covariance (cov3d): ("
                    << cov3d[0].x << ", " << cov3d[0].y << ", " << cov3d[0].z << ", "
                    << cov3d[1].x << ", " << cov3d[1].y << ", " << cov3d[1].z << ")" << std::endl;
                std::cout << "dL/duv (dL_du, dL_dv, dL_dd): ("
                    << dL_duv[0].x << ", " << dL_duv[0].y << ", " << dL_duv[0].z << ")" << std::endl;
                std::cout << "dL/dcov2d (dL_dcov2d_0, dL_dcov2d_1, dL_dcov2d_2): ("
                    << dL_dcov2d[0].x << ", " << dL_dcov2d[0].y << ", " << dL_dcov2d[0].z << ")" << std::endl;
                std::cout << "dL/dxyz (dL_dx, dL_dy, dL_dz): ("
                    << dL_dxyz[0].x << ", " << dL_dxyz[0].y << ", " << dL_dxyz[0].z << ")" << std::endl;
                std::cout << "dL/dscale (dL_dscale_x, dL_dscale_y, dL_dscale_z): ("
                    << dL_dscale[0].x << ", " << dL_dscale[0].y << ", " << dL_dscale[0].z << ")" << std::endl;
                std::cout << "dL/drotate (dL_drotate_0, dL_drotate_1, dL_drotate_2, dL_drotate_3): ("
                    << dL_drotate[0].x << ", " << dL_drotate[0].y << ", " << dL_drotate[0].z << ", " << dL_drotate[0].w << ")" << std::endl;
            }
        }
    };

    struct EvalSHForwardStruct {
        std::vector<std::vector<std::vector<float>>> sh_coefficients;  // Dimensions: P * 3 * D
        std::vector<glm::vec3> v;          // Velocity (vx, vy, vz)
        std::vector<int> visible;          // Visibility flag
        std::vector<glm::vec3> color;      // RGB color (r, g, b)

        void print() const {
            std::cout << "Loaded " << sh_coefficients.size() << " vertices." << std::endl;
            if (!sh_coefficients.empty()) {
                std::cout << "\nFirst vertex data:" << std::endl;
                std::cout << "SH Coefficients (First vertex, D = 16): ";
                for (int j = 0; j < 3; ++j) {
                    for (int i = 0; i < 16; ++i) {
                        std::cout << sh_coefficients[0][j][i] << " ";
                    }
                }
                std::cout << std::endl;

                std::cout << "Velocity (v): (" << v[0].x << ", " << v[0].y << ", " << v[0].z << ")" << std::endl;
                std::cout << "Visible: " << visible[0] << std::endl;
                std::cout << "Color (rgb): (" << color[0].x << ", " << color[0].y << ", " << color[0].z << ")" << std::endl;
            }
        }
    };

    struct EvalSHBackwardStruct {
        std::vector<std::vector<std::vector<float>>> sh_coefficients;   // SH coefficients (P * 3 * D)
        std::vector<glm::vec3> v;                                       // Velocity (vx, vy, vz)
        std::vector<int> visible;                                       // Visibility flag
        std::vector<glm::vec3> dL_dcolor;                               // dL/dr, dL/dg, dL/db
        std::vector<std::vector<std::vector<float>>> dL_dsh;            // Derivative SH coefficients (P * 3 * D)
        std::vector<glm::vec3> dL_dv;                                   // Derivative of velocity (dL_dvx, dL_dvy, dL_dvz)

        void print() const {
            std::cout << "Loaded " << sh_coefficients.size() << " vertices." << std::endl;
            if (!sh_coefficients.empty()) {
                std::cout << "\nFirst vertex data:" << std::endl;

                std::cout << "SH Coefficients (First vertex, D = 16): ";
                for (int j = 0; j < 3; ++j) {
                    for (int i = 0; i < 16; ++i) {
                        std::cout << sh_coefficients[0][j][i] << " ";
                    }
                }
                std::cout << std::endl;

                std::cout << "Velocity (v): (" << v[0].x << ", " << v[0].y << ", " << v[0].z << ")" << std::endl;
                std::cout << "Visible: " << visible[0] << std::endl;

                std::cout << "dL/dColor (r, g, b): (" << dL_dcolor[0].x << ", " << dL_dcolor[0].y << ", " << dL_dcolor[0].z << ")" << std::endl;

                std::cout << "SH Coefficients (First vertex, D = 16): ";
                for (int j = 0; j < 3; ++j) {
                    for (int i = 0; i < 16; ++i) {
                        std::cout << dL_dsh[0][j][i] << " ";
                    }
                }
                std::cout << std::endl;

                std::cout << "dL/dv (vx, vy, vz): (" << dL_dv[0].x << ", " << dL_dv[0].y << ", " << dL_dv[0].z << ")" << std::endl;
            }
        }
    };

    struct InputLegoInitStruct {
        std::vector<glm::vec3> positions;      // 3D positions (x, y, z)
        std::vector<glm::vec3> scales;         // Scaling factors (scale_x, scale_y, scale_z)
        std::vector<glm::vec4> rotations;      // Rotation (rotate_0, rotate_1, rotate_2, rotate_3)
        std::vector<float> opacities;          // Opacity

        struct Data {
            float values[48];  // 48维浮点数组
        };
        std::vector<Data> colors;         // RGB colors (r, g, b)
        std::vector<vks::Texture2D> images;   // Images

        std::vector<std::string> img_list;
        std::vector<std::vector<float>> extr_tmp;  
        std::vector<std::vector<float>> extr_tmp_test; 
        // std::vector<std::vector<float>> intr_tmp;

        // Print the first vertex's data for verification
        void print() const {
            std::cout << "Loaded " << positions.size() << " vertices." << std::endl;
            if (!positions.empty()) {
#ifdef VULKAN_PC
                std::cout << "\nFirst vertex data:" << std::endl;

                // Print position (x, y, z)
                std::cout << "Position (x, y, z): (" << positions[0].x << ", " << positions[0].y << ", " << positions[0].z << ")" << std::endl;

                // Print scale (scale_x, scale_y, scale_z)
                std::cout << "Scale (x, y, z): (" << scales[0].x << ", " << scales[0].y << ", " << scales[0].z << ")" << std::endl;


                // // Print rotation (as quaternion)
                // std::cout << "Rotation (quaternion): (" << rotations[0].x << ", " << rotations[0].y << ", " << rotations[0].z << ", " << rotations[0].w << ")" << std::endl;

                // // Print opacity
                // std::cout << "Opacity: " << opacities[0] << std::endl;

                // Print color (r, g, b)
                std::cout << "Color (r, g, b): (" << colors[0].values[0] << ", " << colors[0].values[1] << ", " << colors[0].values[2] << ")" << std::endl;

                // Print image list
                std::cout << "Image list: " << img_list[0] << std::endl;

                // Print extrinsic parameters
                std::cout << "Extrinsic parameters: ";
                for (int i = 0; i < 12; ++i) {
                    std::cout << extr_tmp[0][i] << " ";
                }
#endif
#ifdef VULKAN_HW
                LOGI("First vertex data:");
                LOGI("Position (x, y, z): (%{public}f, %{public}f, %{public}f)", positions[0].x, positions[0].y, positions[0].z);
                LOGI("Scale (x, y, z): (%{public}f, %{public}f, %{public}f)", scales[0].x, scales[0].y, scales[0].z);
                LOGI("Color (r, g, b): (%{public}f, %{public}f, %{public}f)", colors[0].values[0], colors[0].values[1], colors[0].values[2]);
                LOGI("Image list: : %{public}s", img_list[0].c_str());
                LOGI("Extrinsic parameters: ");
                std::ostringstream oss;
                for (int i = 0; i < 12; ++i) {
                    oss << extr_tmp[0][i];
                    if (i < 12 - 1) {
                        oss << " ";
                    }
                }
                std::string result = oss.str();
                LOGI("%{public}s", result.c_str());
                LOGI("Extrinsic parameters for render: ");
                oss.str("");
                oss.clear();
                for (int i = 0; i < 12; ++i) {
                    oss << extr_tmp_test[0][i];
                    if (i < 12 - 1) {
                        oss << " ";
                    }
                }
                result = oss.str();
                LOGI("%{public}s", result.c_str());
#endif
            }
        }
    };

    static PlyDataStruct load_input_data(const std::string& filepath);

    static ProjectEWAForwardStruct loadEWAForwardData();
    static ProjectEWABackwardStruct loadEWABackwardData();
    static EvalSHForwardStruct loadEvalSHForwardData();
    static EvalSHBackwardStruct loadEvalSHBackwardData();
    static InputGaussianStruct loadInputGaussianData();
    static InputLegoInitStruct loadInputLegoInitData();
    static InputLegoInitStruct loadAREngineInitData();

    static void printPlyHeader(const std::string& filepath);
    
};
