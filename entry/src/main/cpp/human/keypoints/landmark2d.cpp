#include "keypoints.h"
#include "bfm_data.h"
#include "../../../base/math/vector.h"


// ----------------------- 2D landmark检测代码 -----------------------
struct ParamData {
    float R[3][4];                // 旋转矩阵 (3x4)
    Base::Math::Vec3 offset;      // 平移向量
    std::vector<float> alpha_shp; // 形状系数
    std::vector<float> alpha_exp; // 表情系数
};

Base::Math::Vec4 parse_roi_box_from_bbox(const cv::Rect2d& bbox) {
    float left   = static_cast<float>(bbox.x);
    float top    = static_cast<float>(bbox.y);
    float right  = static_cast<float>(bbox.x + bbox.width);
    float bottom = static_cast<float>(bbox.y + bbox.height);

    float old_size = (right - left + bottom - top) / 2.0f;
    float center_x = right - (right - left) / 2.0f;
    float center_y = bottom - (bottom - top) / 2.0f + old_size * 0.14f;

    float size = old_size * 1.58f;

    Base::Math::Vec4 roi_box;
    roi_box.x = center_x - size / 2.0f; // x1
    roi_box.y = center_y - size / 2.0f; // y1
    roi_box.z = roi_box.x + size;       // x2
    roi_box.w = roi_box.y + size;       // y2

    return roi_box;
}

ParamData parse_param(const std::vector<float>& param) {
    int n = param.size();
    int trans_dim = 12, shape_dim = 0, exp_dim = 0;

    if (n == 62) {
        shape_dim = 40;
        exp_dim = 10;
    } else if (n == 72) {
        shape_dim = 40;
        exp_dim = 20;
    } else if (n == 141) {
        shape_dim = 100;
        exp_dim = 29;
    } else {
        throw std::invalid_argument("Invalid param size: " + std::to_string(n));
    }

    ParamData data;

    // 填充 R 矩阵 (3x4)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            data.R[i][j] = param[i * 4 + j];

    // offset 是第4列
    data.offset = Base::Math::Vec3(data.R[0][3], data.R[1][3], data.R[2][3]);

    // 提取 alpha_shp
    data.alpha_shp.assign(param.begin() + trans_dim,
                          param.begin() + trans_dim + shape_dim);

    // 提取 alpha_exp
    data.alpha_exp.assign(param.begin() + trans_dim + shape_dim, param.end());

    return data;
}

std::vector<Base::Math::Vec3> compute_pts3d(const ParamData& data) {
    const int dim = 204;
    std::vector<float> shape_basis(dim, 0.0f);
    std::vector<float> exp_basis(dim, 0.0f);

    // w_shp_base @ alpha_shp
    for (int i = 0; i < dim; ++i) {
        for (size_t j = 0; j < data.alpha_shp.size(); ++j) {
            shape_basis[i] += w_shp_base[i * data.alpha_shp.size() + j] * data.alpha_shp[j];
        }
    }

    // w_exp_base @ alpha_exp
    for (int i = 0; i < dim; ++i) {
        for (size_t j = 0; j < data.alpha_exp.size(); ++j) {
            exp_basis[i] += w_exp_base[i * data.alpha_exp.size() + j] * data.alpha_exp[j];
        }
    }

    std::vector<Base::Math::Vec3> pts3d(dim / 3);  // 每个点 3 维，共 68 个点

    for (int i = 0; i < dim; i += 3) {
        Base::Math::Vec3 point(
            u_base[i] + shape_basis[i] + exp_basis[i],
            u_base[i + 1] + shape_basis[i + 1] + exp_basis[i + 1],
            u_base[i + 2] + shape_basis[i + 2] + exp_basis[i + 2]
        );

        // R_3x3 * point + offset
        float x = data.R[0][0] * point.x + data.R[0][1] * point.y + data.R[0][2] * point.z + data.offset.x;
        float y = data.R[1][0] * point.x + data.R[1][1] * point.y + data.R[1][2] * point.z + data.offset.y;
        float z = data.R[2][0] * point.x + data.R[2][1] * point.y + data.R[2][2] * point.z + data.offset.z;

        pts3d[i / 3] = Base::Math::Vec3(x, y, z);
    }

    return pts3d;
}

void similar_transform(std::vector<Base::Math::Vec3>& pts3d, const Base::Math::Vec4& roi_box, int size) {
    float sx = roi_box.x;
    float sy = roi_box.y;
    float ex = roi_box.z;
    float ey = roi_box.w;

    float scale_x = (ex - sx) / size;
    float scale_y = (ey - sy) / size;
    float s = (scale_x + scale_y) / 2.0f;

    // 找到最小的 z 值
    float min_z = std::numeric_limits<float>::max();
    for (const auto& pt : pts3d) {
        min_z = std::min(min_z, pt.z);
    }

    // 变换坐标
    for (auto& pt : pts3d) {
        pt.x = (pt.x - 1.0f) * scale_x + sx;
        pt.y = (size - (pt.y - 1.0f)) * scale_y + sy;
        pt.z = (pt.z - min_z) * s;
    }
}

void Output2Landmarks(std::vector<float>& param, cv::Rect2d& face_box, std::string& savePath){
    // 1. 解析3D人脸参数和检测框
    ParamData data = parse_param(param);
    Base::Math::Vec4 roi_box = parse_roi_box_from_bbox(face_box);

    // 2. 构建 3D 点
    std::vector<Base::Math::Vec3> pts3d = compute_pts3d(data);

    // 3. 坐标变换
    similar_transform(pts3d, roi_box, 120);

    // 4. 提取前16个关键点的 x, y
    vector<string> keypoints_outputs;
    for (int i = 0; i < 16; ++i) {
        const auto& pt = pts3d[i];
        ostringstream oss;
        oss << fixed << setprecision(6) << pt.x << " " << pt.y << "\n";
        keypoints_outputs.push_back(oss.str());
    }

    // 5. 保存到文件
    ofstream ofs(savePath, ios::out);
    if (!ofs.is_open()) {
        cerr << "Failed to open file for writing: " << savePath << endl;
        return;
    }

    for (const auto& line : keypoints_outputs) {
        ofs << line;
    }
    ofs.close();

    // 打印日志
    cout << "Saved 2D landmark to: " << savePath << endl;
}
