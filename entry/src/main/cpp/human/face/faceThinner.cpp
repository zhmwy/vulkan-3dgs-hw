#include "face.h"

// Function to read keypoints from a PLY file with validity check, code from SMPLX/src/fit_smplv2.cpp
std::vector<Base::Math::Vec3> readLandmark3D(const std::string& filename) {
    // Open the PLY file
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Cannot open the file: " << filename << std::endl;
        return {};
    }

    std::string line;
    int num_vertices = 0;
    bool header_ended = false;

    // Read the header to determine the number of vertices
    while (std::getline(infile, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line == "end_header") {
            header_ended = true;
            break;
        }

        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element") {
            std::string element_type;
            iss >> element_type;
            if (element_type == "vertex") {
                iss >> num_vertices;
            }
        }
    }

    if (!header_ended || num_vertices <= 0) {
        std::cerr << "Invalid PLY file format." << std::endl;
        return {};
    }

    std::vector<Base::Math::Vec3> keypoints;
    keypoints.reserve(num_vertices);  // 预分配空间，提升效率

    for (int i = 0; i < num_vertices; ++i) {
        if (!std::getline(infile, line)) {
            std::cerr << "Unexpected end of file when reading vertex data." << std::endl;
            return {};
        }

        std::istringstream iss(line);
        float x, y, z;
        int r, g, b;
        if (!(iss >> x >> y >> z >> r >> g >> b)) {
            std::cerr << "Error parsing vertex data at vertex index " << i << std::endl;
            return {};
        }

        keypoints.emplace_back(x, y, z);
    }

    infile.close();
    return keypoints;
}

using Vec3 = Base::Math::Vec3;
using PointCloud = std::vector<Vec3>;

// 对应np.linspace()
auto linspace = [](float start, float end, int count) {
    std::vector<float> values(count);
    for (int i = 0; i < count; ++i) {
        float t = static_cast<float>(i) / (count - 1);
        values[i] = start + t * (end - start);
    }
    return values;
};

// 计算先升后降的收缩曲线
std::vector<float> generate_shrink_profile(int num_points, float max_value) {
    std::vector<float> result;
    result.reserve(num_points);

    if (num_points == 1) {
        result.push_back(max_value);
        return result;
    }

    int mid = num_points / 2;
    auto up = linspace(0.0f, max_value, mid + 1);         // 0 -> max
    auto down = linspace(max_value, 0.0f, num_points - mid); // max -> 0

    result.insert(result.end(), up.begin(), up.end());
    result.insert(result.end(), down.begin() + 1, down.end()); // 去重连接点
    return result;
}

// 二范数
float length2(const Vec3& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// 归一化
Vec3 normalized(const Vec3& v) {
    float len = std::sqrt(length2(v));
    if (len == 0.0f) return Vec3(0.0f, 0.0f, 0.0f);
    float inv_len = 1.0f / len;
    return Vec3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}


// 暴力最近邻计算
void query_knn(
    const PointCloud& points,
    const PointCloud& queries,
    std::vector<int>& indices,
    std::vector<float>& distances)
{
    indices.resize(queries.size());
    distances.resize(queries.size());

    for (size_t qi = 0; qi < queries.size(); ++qi) {
        float min_dist = std::numeric_limits<float>::max();
        int min_idx = 0;
        for (size_t pi = 0; pi < points.size(); ++pi) {
            Vec3 diff = queries[qi] - points[pi];
            float dist = length2(diff);
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = static_cast<int>(pi);
            }
        }
        indices[qi] = min_idx;
        distances[qi] = std::sqrt(min_dist);  // real distance
    }
}

// 对头部高斯点云进行收缩
PointCloud shrink_head(
    const PointCloud& pcd_3d,
    const PointCloud& landmarks_3d,
    float max_shrink_value = 0.008f,
    float decay_rate = 2.0f,
    const std::string& decay_type = "linear")
{
    if (landmarks_3d.empty()) {
        throw std::invalid_argument("landmarks_3d is empty");
    }

    // 1. 计算头部中心（用于收缩方向）
    Vec3 head_center = (landmarks_3d.front() + landmarks_3d.back()) * 0.5f;

    // 2. 生成半边的收缩强度曲线（从下巴到鬓角）
    int N = landmarks_3d.size(); // 17
    int half_N = (N + 1) / 2;    // 9
    auto shrink_curve = generate_shrink_profile(half_N, max_shrink_value);

    // 3. 合并为完整的17个点的收缩值，下巴由于会重复计算一次，因此去掉一个中间点
    std::vector<float> shrink_values;
    shrink_values.insert(shrink_values.end(), shrink_curve.begin(), shrink_curve.end() - 1);
    shrink_values.insert(shrink_values.end(), shrink_curve.begin(), shrink_curve.end());
    
    // 4. 计算每个关键点的收缩向量（向head_center收缩）
    std::vector<Vec3> delta_lmk_xyz;
    for (size_t i = 0; i < landmarks_3d.size(); ++i) {
        Vec3 dir = head_center - landmarks_3d[i];
        delta_lmk_xyz.push_back(dir * shrink_values[i]);
    }
    
    // 5. KNN匹配
    std::vector<int> knnIndices;
    std::vector<float> knnDists;
    query_knn(landmarks_3d, pcd_3d, knnIndices, knnDists);
    
    // 6. 计算每个高斯点的位移,离关键点越近的高斯点受到的影响越大
    std::vector<Vec3> delta_xyz(pcd_3d.size());
    
    float max_dist = (decay_type == "linear") ? (1.0f / decay_rate) : 0.0f;
    for (size_t i = 0; i < pcd_3d.size(); ++i) {
        int lmk_idx = knnIndices[i];
        float dist = knnDists[i];
        float weight = 1.0f;

        if (decay_type == "exp") {
            weight = std::exp(-decay_rate * dist);
        } else if (decay_type == "linear") {
            weight = std::clamp(1.0f - dist / max_dist, 0.0f, 1.0f);  // weight: f(x) = 1 - 0.5x
        } else {
            throw std::invalid_argument("decay_type must be 'exp' or 'linear'");
        }
        delta_xyz[i] = delta_lmk_xyz[lmk_idx] * weight;
    }
    
    // 7. 应用位移
    PointCloud new_pcd3d = pcd_3d;
    for (size_t i = 0; i < new_pcd3d.size(); ++i) {
        new_pcd3d[i] = new_pcd3d[i] + delta_xyz[i];
    }
    return new_pcd3d;
}

// Helper function to check if a point is within the bounding box
bool isPointInBBox(const Vec3& point, const std::vector<float>& bbox) {
    if (bbox.size() != 6) {
        return false;
    }
    return point.x >= bbox[0] && point.x <= bbox[1] &&
           point.y >= bbox[2] && point.y <= bbox[3] &&
           point.z >= bbox[4] && point.z <= bbox[5];
}

void FaceTune(DataLoader::InputInitStruct& inputGaussian, std::string scenePath)
{
    std::string landmarkFile = scenePath + "/landmarks.ply";
    auto landmarks_3d = readLandmark3D(landmarkFile);
    
    // 下颌线关键点应该有17个
    if (landmarks_3d.size() != 17) {
        std::cerr << "Warning: Expected 17 landmarks, got " << landmarks_3d.size() << std::endl;
    }
    
    // Check if body_bbox is valid
    if (inputGaussian.body_bboxes.size() != 6) {
        std::cerr << "Warning: Invalid body_bbox, processing all points" << std::endl;
        // Fall back to original behavior if bbox is invalid
        auto new_cloud = shrink_head(inputGaussian.positions, landmarks_3d, 0.015f, 1.0f);
        
        inputGaussian.positions.clear();
        inputGaussian.positions.reserve(new_cloud.size());
        for (const auto& pt : new_cloud) {
            inputGaussian.positions.emplace_back(pt.x, pt.y, pt.z);
        }
    } else {
        // Filter points within the bounding box
        PointCloud filtered_points;
        std::vector<size_t> indices_in_bbox;
        
        for (size_t i = 0; i < inputGaussian.positions.size(); ++i) {
            if (isPointInBBox(inputGaussian.positions[i], inputGaussian.body_bboxes)) {
                filtered_points.push_back(inputGaussian.positions[i]);
                indices_in_bbox.push_back(i);
            }
        }
        
        // Only process if there are points in the bbox
        if (!filtered_points.empty()) {
            // max_shrink_value 和 decay_rate可能需要调整，分别代表了收缩幅度、高斯点受关键点影响随距离的衰减率
            // max_shrink_value越大，收缩效果越明显；decay_rate越大，对场景的影响范围越小。
            auto new_cloud = shrink_head(filtered_points, landmarks_3d, 0.015f, 1.0f);
            
            // Update only the points that were within the bbox
            for (size_t i = 0; i < indices_in_bbox.size(); ++i) {
                inputGaussian.positions[indices_in_bbox[i]] = new_cloud[i];
            }
        } else {
            std::cerr << "Warning: No points found within body_bbox" << std::endl;
        }
    }
    
    // 保存
    std::filesystem::create_directories(scenePath + "/processed_gaussians");
    std::string saveProcessedGSFile = scenePath + "/processed_gaussians";
    DataLoader::saveToPLY_tinyply(saveProcessedGSFile, inputGaussian);
}