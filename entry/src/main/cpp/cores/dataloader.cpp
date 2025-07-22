#include "dataloader.h"
#include "base/math/matrix.h"
#include "base/math/matrix_util.h"
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <dirent.h>


int extractNumberKtx(const std::string& str) {
    std::regex re("r_(\\d+)\\.ktx");
    std::smatch match;
    if (std::regex_search(str, match, re)) {
        return std::stoi(match[1].str()); // Extract and convert the number
    }
    return -1; // Return -1 if no number is found (optional, depends on data guarantee)
}

int extractNumberTxt(const std::string& str) {
    std::regex re("r_(\\d+)\\.txt");
    std::smatch match;
    if (std::regex_search(str, match, re)) {
        return std::stoi(match[1].str()); // Extract and convert the number
    }
    return -1; // Return -1 if no number is found (optional, depends on data guarantee)
}

DataLoader::PlyDataStruct DataLoader::load_input_data(const std::string& filepath){
    DataLoader::PlyDataStruct data;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open PLY file.");

    PlyFile plyFile;
    plyFile.parse_header(file);

    std::shared_ptr<PlyData> x, y, z, scale_x, scale_y, scale_z, rotate_0, rotate_1, rotate_2, rotate_3;

    try {
        x = plyFile.request_properties_from_element("vertex", { "x" });
        y = plyFile.request_properties_from_element("vertex", { "y" });
        z = plyFile.request_properties_from_element("vertex", { "z" });
        scale_x = plyFile.request_properties_from_element("vertex", { "scale_x" });
        scale_y = plyFile.request_properties_from_element("vertex", { "scale_y" });
        scale_z = plyFile.request_properties_from_element("vertex", { "scale_z" });
        rotate_0 = plyFile.request_properties_from_element("vertex", { "rotate_0" });
        rotate_1 = plyFile.request_properties_from_element("vertex", { "rotate_1" });
        rotate_2 = plyFile.request_properties_from_element("vertex", { "rotate_2" });
        rotate_3 = plyFile.request_properties_from_element("vertex", { "rotate_3" });
    }
    catch (const std::exception& e) {
        std::cout << "Error reading properties: " << e.what() << std::endl;
    }

    plyFile.read(file);

    // To GLM
    size_t numVertices = x->count;

    for (size_t i = 0; i < numVertices; ++i) {
        data.xyz.emplace_back(
            reinterpret_cast<float*>(x->buffer.get())[i],
            reinterpret_cast<float*>(y->buffer.get())[i],
            reinterpret_cast<float*>(z->buffer.get())[i]
        );

        data.scale.emplace_back(
            reinterpret_cast<float*>(scale_x->buffer.get())[i],
            reinterpret_cast<float*>(scale_y->buffer.get())[i],
            reinterpret_cast<float*>(scale_z->buffer.get())[i]
        );

        data.rotate.emplace_back(
            reinterpret_cast<float*>(rotate_0->buffer.get())[i],
            reinterpret_cast<float*>(rotate_1->buffer.get())[i],
            reinterpret_cast<float*>(rotate_2->buffer.get())[i],
            reinterpret_cast<float*>(rotate_3->buffer.get())[i]
        );
    }

    return data;
}

void DataLoader::printPlyHeader(const std::string& filepath)
{
    DataLoader::PlyDataStruct data;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open PLY file.");

    PlyFile plyFile;
    plyFile.parse_header(file);
    std::cout << "\t[ply_header] Type: " << (plyFile.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto& c : plyFile.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto& c : plyFile.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto& e : plyFile.get_elements())
    {
        std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
        for (const auto& p : e.properties)
        {
            std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
            if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
            std::cout << std::endl;
        }
    }
}

void extractIntrinsics(const std::vector<std::string>& file_list, std::vector<std::vector<float>>& intrinsics) {
    for (const std::string& filename : file_list) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open intrinsic file: " << filename << std::endl;
            continue;
        }

        std::vector<float> intr(4, 0.0f);  // 预留 4 个浮点数
        for (int i = 0; i < 4; ++i) {
            file >> intr[i];
        }
        file.close();

        if (intr.size() == 4) {
            intrinsics.push_back(intr);
        }
        else {
            std::cerr << "Invalid intrinsic file format in: " << filename << std::endl;
        }
    }
}

void extractExtrinsics(const std::vector<std::string>& files, std::vector<std::vector<float>>& extrinsics) {
    for (const auto& entry : files) {
        if (entry.find("cameras.txt") == std::string::npos) {
            std::ifstream file(entry);
            if (!file.is_open()) throw std::runtime_error("Failed to open file.");

            std::string line;
            std::vector<float> extrinsicValues;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                float value;

                // Read values from the line
                while (iss >> value) {
                    extrinsicValues.push_back(value);
                }
            }
            extrinsics.push_back(extrinsicValues);
        }
    }
}

long long extractNumber(const std::string& str) {
    std::string imgName = std::filesystem::path(str).stem().string();
    
    std::string numberStr;

    // 遍历文件名，提取数字部分
    for (char ch : imgName) {
        if (std::isdigit(ch)) {
            numberStr += ch;
        }
    }
    
    // 如果提取到数字，则返回转换后的数字
    if (!numberStr.empty()) {
        try {
            return std::stoll(numberStr); // 使用 stoll 转换为 long long
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Invalid number in string: " << str << std::endl;
            throw;
        }
    }

    // 如果没有找到数字，返回默认值（例如：0）
    return 0;
}

void getFilesInDirectory(std::string path, std::string ext, std::vector<std::string>& outFiles){
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        std::cerr << "Error: Could not open directory " << path << std::endl;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;

        // Skip "." and ".." entries
        if (filename == "." || filename == "..") {
            continue;
        }

        // Check if the filename matches the pattern "r_*.ktx"
        if (filename.size() > ext.size() && filename.substr(filename.size() - ext.size(), ext.size()) == ext){
            outFiles.push_back(path + "/" + filename);
        }
    }
    closedir(dir);
}

struct Vec3Hash {
    size_t operator()(const Base::Math::Vec3& v) const {
        std::hash<float> floatHasher;
        size_t h1 = floatHasher(v.x);
        size_t h2 = floatHasher(v.y);
        size_t h3 = floatHasher(v.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);  // 组合哈希值
    }
};

struct Vec3Equal {
    bool operator()(const Base::Math::Vec3& v1, const Base::Math::Vec3& v2) const {
        return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
    }
};

Base::Math::Mat4X4 quat2matrix(DataLoader::InputInitStruct::DataExtr& extr) {
    Base::Math::Mat4X4 W2C(1.0f);
    double w = extr.values[0], x = extr.values[1], y = extr.values[2], z = extr.values[3];
    double tx = extr.values[4], ty = extr.values[5], tz = extr.values[6];
    
    W2C[0][0] = 1 - 2 * (y * y + z * z);
    W2C[0][1] = 2 * (x * y - w * z);
    W2C[0][2] = 2 * (x * z + w * y);
    W2C[0][3] = tx;

    W2C[1][0] = 2 * (x * y + w * z);
    W2C[1][1] = 1 - 2 * (x * x + z * z);
    W2C[1][2] = 2 * (y * z - w * x);
    W2C[1][3] = ty;

    W2C[2][0] = 2 * (x * z - w * y);
    W2C[2][1] = 2 * (y * z + w * x);
    W2C[2][2] = 1 - 2 * (x * x + y * y);
    W2C[2][3] = tz;

    return W2C;
}

/*
void DataLoader::InputInitStruct::normalizeData() {
    if (positions.empty()) {
        std::cerr << "Error: No data available for normalization!" << std::endl;
        return;
    }

    float stddev = glm::sqrt(variance) + 1e-6f;  // **保证所有维度的缩放一致**

    // **标准化 positions**
    for (auto& pos : positions) {
        pos = (pos - mean) / stddev;
    }
    for (auto& scale : scales) {
        scale = scale - glm::log(stddev);
    }
    
    for (auto& extr : extr_tmp) {

        glm::mat4 W2C = quat2matrix(extr);
        glm::mat4 C2W = glm::inverse(W2C);
        glm::vec3 cam_center(C2W[0][3], C2W[1][3], C2W[2][3]);
        
        glm::vec3 normalized_center = (cam_center - mean) / stddev;
        C2W[0][3] = normalized_center.x;
        C2W[1][3] = normalized_center.y;
        C2W[2][3] = normalized_center.z;

        W2C = glm::inverse(C2W);

        extr.values[4] = W2C[0][3];
        extr.values[5] = W2C[1][3];
        extr.values[6] = W2C[2][3];
    }
    
    for (auto& extr : extr_tmp_test) {

        glm::mat4 W2C = quat2matrix(extr);
        glm::mat4 C2W = glm::inverse(W2C);
        glm::vec3 cam_center(C2W[0][3], C2W[1][3], C2W[2][3]);

        glm::vec3 normalized_center = (cam_center - mean) / stddev;
        C2W[0][3] = normalized_center.x;
        C2W[1][3] = normalized_center.y;
        C2W[2][3] = normalized_center.z;

        W2C = glm::inverse(C2W);

        extr.values[4] = W2C[0][3];
        extr.values[5] = W2C[1][3];
        extr.values[6] = W2C[2][3];
    }
}*/

void DataLoader::InputInitStruct::denormalizeExtr() {
    float stddev = sqrt(variance) + 1e-6f;
    for (auto& extr: extr_tmp) {
        Base::Math::Mat4X4 W2C = quat2matrix(extr);
        Base::Math::Mat4X4 C2W = Base::Math::Inverse(W2C);
        Base::Math::Vec3 cam_center(C2W[0][3], C2W[1][3], C2W[2][3]);
        
        Base::Math::Vec3 normalized_center = cam_center * stddev + Base::Math::Vec3(mean[0], mean[1], mean[2]);
        C2W[0][3] = normalized_center.x;
        C2W[1][3] = normalized_center.y;
        C2W[2][3] = normalized_center.z;
        W2C = Base::Math::Inverse(C2W);
        
        extr.values[4] = W2C[0][3];
        extr.values[5] = W2C[1][3];
        extr.values[6] = W2C[2][3];
    }
}

void DataLoader::InputInitStruct::denormalizePoints() {
    if (positions.empty()) {
        return;
    }
    float stddev = sqrt(variance) + 1e-6f;
    
    for (auto& pos : positions) {
        pos = pos * stddev + Base::Math::Vec3(mean[0], mean[1], mean[2]);
    }
    
    float logdev = log(stddev);
    for (auto& scale : scales) {
        scale = scale + Base::Math::Vec3(logdev, logdev, logdev);
    }
}
/*
void DataLoader::InputInitStruct::computePositionStats() {
    if (positions.empty()) {
        std::cerr << "Error: No position data available!" << std::endl;
        return;
    }
    
    mean = glm::vec3(0.0, 0.0, 0.0);
    for (const auto& pos : positions) {
        mean += pos;
    }
    mean /= static_cast<float>(positions.size());
    
    double sum = 0;
    for (const auto& pos : positions) {
        float dist = glm::length(pos - mean);  // **计算欧几里得距离**
        sum += dist * dist;  // **累加平方距离**
    }
    variance = sum / static_cast<double>(positions.size());  // **求均值**
    
    
    std::cout << "Position Mean: (" << mean.x << ", " << mean.y << ", " << mean.z << ")" << std::endl;
    std::cout << "Position Variance: " << variance << std::endl;  // **variance 现在是单个 float**
}*/

void DataLoader::InputInitStruct::getNerfppNorm() {
    this->radius = -1.0;
    Base::Math::Vec3 cam_centers = Base::Math::Vec3(0, 0, 0);
    for (auto& extr : extr_tmp) {
        Base::Math::Mat4X4 W2C = quat2matrix(extr);
        Base::Math::Mat4X4 C2W = Base::Math::Inverse(W2C);
        Base::Math::Vec3 cam_center(C2W[0][3], C2W[1][3], C2W[2][3]);
        cam_centers += cam_center;
    }
    cam_centers /= extr_tmp.size();
    
    for (auto& extr : extr_tmp) {
    	Base::Math::Mat4X4 W2C = quat2matrix(extr);
        Base::Math::Mat4X4 C2W = Base::Math::Inverse(W2C);
        Base::Math::Vec3 cam_center(C2W[0][3], C2W[1][3], C2W[2][3]);
        Base::Math::Vec3 dist = cam_centers - cam_center;
        this->radius = std::max(this->radius, sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]));
    }
    
    this->radius *= 1.1;
}

void DataLoader::loadInputGaussianData(std::string& gaussianPath, DataLoader::InputInitStruct& data) {
    std::ifstream file(gaussianPath, std::ios::binary);
    if (!file.is_open()){
        LOGE("Failed to open PLY file from %{public}s.", gaussianPath.c_str());
        return;
    }

    PlyFile plyFile;
    plyFile.parse_header(file);

    // Declare shared pointers to request each property from the PLY file
    std::shared_ptr<PlyData> x, y, z;
    std::shared_ptr<PlyData> scale_x, scale_y, scale_z;
    std::shared_ptr<PlyData> rotate_0, rotate_1, rotate_2, rotate_3;
    std::shared_ptr<PlyData> opacity;
    std::shared_ptr<PlyData> r, g, b;
    std::vector<std::shared_ptr<PlyData>> colorProps;
    
    try {
        // Request each property from the "vertex" element
        x = plyFile.request_properties_from_element("vertex", { "x" });
        y = plyFile.request_properties_from_element("vertex", { "y" });
        z = plyFile.request_properties_from_element("vertex", { "z" });
        scale_x = plyFile.request_properties_from_element("vertex", { "scale_0" });
        scale_y = plyFile.request_properties_from_element("vertex", { "scale_1" });
        scale_z = plyFile.request_properties_from_element("vertex", { "scale_2" });
        rotate_0 = plyFile.request_properties_from_element("vertex", { "rot_0" });
        rotate_1 = plyFile.request_properties_from_element("vertex", { "rot_1" });
        rotate_2 = plyFile.request_properties_from_element("vertex", { "rot_2" });
        rotate_3 = plyFile.request_properties_from_element("vertex", { "rot_3" });
        opacity = plyFile.request_properties_from_element("vertex", { "opacity" });
        
        for (int i = 0; i < 3; ++i) {
            std::string propName = "f_dc_" + std::to_string(i);
            colorProps.push_back(plyFile.request_properties_from_element("vertex", { propName }));
        }

        for (int i = 0; i < 45; ++i) {
            std::string propName = "f_rest_" + std::to_string(i);
            colorProps.push_back(plyFile.request_properties_from_element("vertex", { propName }));
        }
        
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading properties: " << e.what() << std::endl;
    }
    
    plyFile.read(file);
    size_t numVertices = x->count;
    
    data.positions.clear();
    data.positions.reserve(numVertices);
    data.scales.clear();
    data.scales.reserve(numVertices);
    data.rotations.clear();
    data.rotations.reserve(numVertices);
    data.opacities.clear();
    data.opacities.reserve(numVertices);
    data.colors.clear();
    data.colors.reserve(numVertices);
    
    for (size_t i = 0; i < numVertices; ++i) {
        Base::Math::Vec3 pos(
            reinterpret_cast<float*>(x->buffer.get())[i],
            reinterpret_cast<float*>(y->buffer.get())[i],
            reinterpret_cast<float*>(z->buffer.get())[i]
        );

        // 继续存储其他属性
        data.positions.push_back(pos);

        data.scales.emplace_back(
            reinterpret_cast<float*>(scale_x->buffer.get())[i],
            reinterpret_cast<float*>(scale_y->buffer.get())[i],
            reinterpret_cast<float*>(scale_z->buffer.get())[i]
        );

        data.rotations.emplace_back(
            reinterpret_cast<float*>(rotate_0->buffer.get())[i],
            reinterpret_cast<float*>(rotate_1->buffer.get())[i],
            reinterpret_cast<float*>(rotate_2->buffer.get())[i],
            reinterpret_cast<float*>(rotate_3->buffer.get())[i]
        );

        data.opacities.push_back(reinterpret_cast<float*>(opacity->buffer.get())[i]);

        // 处理颜色数据
        DataLoader::InputInitStruct::Data colorData = {};
        for (size_t j = 0; j < colorProps.size(); ++j) {
            float value = reinterpret_cast<float*>(colorProps[j]->buffer.get())[i];
            if(j < 3){
                colorData.values[j / 3 + (j % 3) * 16] = value;
            }
            else if(j < 18){
                colorData.values[j-2] = value;
            }
            else if(j < 33){
                colorData.values[j-1] = value;
            }
            else{
                colorData.values[j] = value;
            }
        }
        data.colors.push_back(colorData);
    }
    LOGI("Resume from gaussians OK! Read %{public}zu points.", data.positions.size());
}

void DataLoader::loadExtraGaussianData(std::string& gaussianPath, DataLoader::InputInitStruct& data) {
    std::ifstream file(gaussianPath, std::ios::binary);
    if (!file.is_open()){
        LOGE("Failed to open PLY file from %{public}s.", gaussianPath.c_str());
        return;
    }

    PlyFile plyFile;
    plyFile.parse_header(file);

    // Declare shared pointers to request each property from the PLY file
    std::shared_ptr<PlyData> x, y, z;
    std::shared_ptr<PlyData> scale_x, scale_y, scale_z;
    std::shared_ptr<PlyData> rotate_0, rotate_1, rotate_2, rotate_3;
    std::shared_ptr<PlyData> opacity;
    std::shared_ptr<PlyData> r, g, b;
    std::vector<std::shared_ptr<PlyData>> colorProps;
    
    try {
        // Request each property from the "vertex" element
        x = plyFile.request_properties_from_element("vertex", { "x" });
        y = plyFile.request_properties_from_element("vertex", { "y" });
        z = plyFile.request_properties_from_element("vertex", { "z" });
        scale_x = plyFile.request_properties_from_element("vertex", { "scale_0" });
        scale_y = plyFile.request_properties_from_element("vertex", { "scale_1" });
        scale_z = plyFile.request_properties_from_element("vertex", { "scale_2" });
        rotate_0 = plyFile.request_properties_from_element("vertex", { "rot_0" });
        rotate_1 = plyFile.request_properties_from_element("vertex", { "rot_1" });
        rotate_2 = plyFile.request_properties_from_element("vertex", { "rot_2" });
        rotate_3 = plyFile.request_properties_from_element("vertex", { "rot_3" });
        opacity = plyFile.request_properties_from_element("vertex", { "opacity" });
        
        for (int i = 0; i < 3; ++i) {
            std::string propName = "f_dc_" + std::to_string(i);
            colorProps.push_back(plyFile.request_properties_from_element("vertex", { propName }));
        }

        for (int i = 0; i < 45; ++i) {
            std::string propName = "f_rest_" + std::to_string(i);
            colorProps.push_back(plyFile.request_properties_from_element("vertex", { propName }));
        }
        
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading properties: " << e.what() << std::endl;
    }
    
    plyFile.read(file);
    size_t numVertices = x->count;
    
    for (size_t i = 0; i < numVertices; ++i) {
        Base::Math::Vec3 pos(
            reinterpret_cast<float*>(x->buffer.get())[i],
            reinterpret_cast<float*>(y->buffer.get())[i],
            reinterpret_cast<float*>(z->buffer.get())[i]
        );
        data.positions.push_back(pos);

        Base::Math::Vec3 scale(
            reinterpret_cast<float*>(scale_x->buffer.get())[i],
            reinterpret_cast<float*>(scale_y->buffer.get())[i],
            reinterpret_cast<float*>(scale_z->buffer.get())[i]
        );
        data.scales.push_back(scale);

        Base::Math::Vec4 rotate(
            reinterpret_cast<float*>(rotate_0->buffer.get())[i],
            reinterpret_cast<float*>(rotate_1->buffer.get())[i],
            reinterpret_cast<float*>(rotate_2->buffer.get())[i],
            reinterpret_cast<float*>(rotate_3->buffer.get())[i]
        );
        data.rotations.push_back(rotate);

        data.opacities.push_back(reinterpret_cast<float*>(opacity->buffer.get())[i]);

        // 处理颜色数据
        DataLoader::InputInitStruct::Data colorData = {};
        for (size_t j = 0; j < colorProps.size(); ++j) {
            float value = reinterpret_cast<float*>(colorProps[j]->buffer.get())[i];
            if(j < 3){
                colorData.values[j / 3 + (j % 3) * 16] = value;
            }
            else if(j < 18){
                colorData.values[j-2] = value;
            }
            else if(j < 33){
                colorData.values[j-1] = value;
            }
            else{
                colorData.values[j] = value;
            }
        }
        data.colors.push_back(colorData);
    }
    LOGI("Resume from gaussians OK! Read %{public}zu points.", data.positions.size());
}

DataLoader::InputInitStruct  DataLoader::loadHeadGaussians(std::string dataPath, DataLoader::InputInitStruct& data) {
    std::string resume_gaussian_path = dataPath + "/head/point_cloud.ply";
    loadInputGaussianData(resume_gaussian_path, data );
    return data;
}

void DataLoader::saveToPLY_tinyply(const std::string& fullDataPath, const DataLoader::InputInitStruct& data, bool save_head) {
    if (!data.head_bbox.empty() && data.head_bbox.size() == 6 && save_head) {
        float x_min = data.head_bbox[0];
        float x_max = data.head_bbox[1];
        float y_min = data.head_bbox[2];
        float y_max = data.head_bbox[3];
        float z_min = data.head_bbox[4];
        float z_max = data.head_bbox[5];
        
        float x_range = x_max - x_min;
        float y_range = y_max - y_min;
        float z_range = z_max - z_min;
        
        x_min -= x_range * 0.1;
        x_max += x_range * 0.1;
        y_min -= y_range * 0.1;
        y_max += y_range * 0.1;
        z_min -= z_range * 0.1;
        z_max += z_range * 0.1;

        std::vector<float> head_positions, head_scales, head_rotations, head_opacities, head_colors, head_f_rest;
        std::vector<float> other_positions, other_scales, other_rotations, other_opacities, other_colors, other_f_rest;

        size_t numPoints = data.positions.size();
        float stddev = sqrt(data.variance) + 1e-6f;

        head_positions.reserve(numPoints * 3);
        head_scales.reserve(numPoints * 3);
        head_rotations.reserve(numPoints * 4);
        head_opacities.reserve(numPoints);
        head_colors.reserve(numPoints * 3);
        head_f_rest.reserve(numPoints * 45);

        other_positions.reserve(numPoints * 3);
        other_scales.reserve(numPoints * 3);
        other_rotations.reserve(numPoints * 4);
        other_opacities.reserve(numPoints);
        other_colors.reserve(numPoints * 3);
        other_f_rest.reserve(numPoints * 45);

        for (size_t i = 0; i < numPoints; ++i) {
            Base::Math::Vec3 pos = data.positions[i];
//             glm::vec3 orig_pos = data.positions[i] * stddev + data.mean;
            
            bool inside = (pos.x >= x_min && pos.x <= x_max) &&
                          (pos.y >= y_min && pos.y <= y_max) &&
                          (pos.z >= z_min && pos.z <= z_max);
            LOGI("x: %{public}f, y: %{public}f, z: %{public}f", pos.x, pos.y, pos.z);
            Base::Math::Vec3 scale = (i < data.scales.size()) ? data.scales[i] : Base::Math::Vec3(1.0f, 1.0f, 1.0f);
            Base::Math::Vec4 rotation = (i < data.rotations.size()) ? data.rotations[i] : Base::Math::Vec4(0, 0, 0, 1);
            float opacity = (i < data.opacities.size()) ? data.opacities[i] : 1.0f;
            Base::Math::Vec3 color(1.0f, 1.0f, 1.0f);
            std::vector<float> f_rest_values;

            if (i < data.colors.size()) {
                color = Base::Math::Vec3(
                    data.colors[i].values[0 * 16],
                    data.colors[i].values[1 * 16],
                    data.colors[i].values[2 * 16]
                );
                for (int j = 1; j < 16; ++j) {
                    f_rest_values.push_back(data.colors[i].values[0 * 16 + j]);
                    f_rest_values.push_back(data.colors[i].values[1 * 16 + j]);
                    f_rest_values.push_back(data.colors[i].values[2 * 16 + j]);
                }
            }
            if (inside) {
                head_positions.insert(head_positions.end(), {pos.x, pos.y, pos.z});
                head_scales.insert(head_scales.end(), {scale.x, scale.y, scale.z});
                head_rotations.insert(head_rotations.end(), {rotation.x, rotation.y, rotation.z, rotation.w});
                head_opacities.push_back(opacity);
                head_colors.insert(head_colors.end(), {color.x, color.y, color.z});
                head_f_rest.insert(head_f_rest.end(), f_rest_values.begin(), f_rest_values.end());
            } else {
                other_positions.insert(other_positions.end(), {pos.x, pos.y, pos.z});
                other_scales.insert(other_scales.end(), {scale.x, scale.y, scale.z});
                other_rotations.insert(other_rotations.end(), {rotation.x, rotation.y, rotation.z, rotation.w});
                other_opacities.push_back(opacity);
                other_colors.insert(other_colors.end(), {color.x, color.y, color.z});
                other_f_rest.insert(other_f_rest.end(), f_rest_values.begin(), f_rest_values.end());
            }
        }

        size_t head_numPoints = head_positions.size() / 3;
        if (head_numPoints > 0) {
            std::filesystem::create_directories(fullDataPath + "/head");
            
            std::string head_gaussians_path = fullDataPath + "/head/point_cloud.ply";
            std::ofstream headOutFile(head_gaussians_path, std::ios::binary);
            if (headOutFile.is_open()) {
                tinyply::PlyFile headPlyFile;
                headPlyFile.add_properties_to_element("vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, head_numPoints, reinterpret_cast<uint8_t*>(head_positions.data()), tinyply::Type::INVALID, 0);
                headPlyFile.add_properties_to_element("vertex", {"scale_0", "scale_1", "scale_2"}, tinyply::Type::FLOAT32, head_numPoints, reinterpret_cast<uint8_t*>(head_scales.data()), tinyply::Type::INVALID, 0);
                headPlyFile.add_properties_to_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}, tinyply::Type::FLOAT32, head_numPoints, reinterpret_cast<uint8_t*>(head_rotations.data()), tinyply::Type::INVALID, 0);
                headPlyFile.add_properties_to_element("vertex", {"opacity"}, tinyply::Type::FLOAT32, head_numPoints, reinterpret_cast<uint8_t*>(head_opacities.data()), tinyply::Type::INVALID, 0);
                headPlyFile.add_properties_to_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"}, tinyply::Type::FLOAT32, head_numPoints, reinterpret_cast<uint8_t*>(head_colors.data()), tinyply::Type::INVALID, 0);

                for (int i = 0; i < 45; ++i) {
                    headPlyFile.add_properties_to_element("vertex", {"f_rest_" + std::to_string(i)}, tinyply::Type::FLOAT32, head_numPoints, reinterpret_cast<uint8_t*>(&head_f_rest[i * head_numPoints]), tinyply::Type::INVALID, 0);
                }

                headPlyFile.write(headOutFile, true);
                headOutFile.close();
                LOGI("Saved %{public}zu head points to head_gaussians.ply", head_numPoints);
            } else {
                LOGE("Error: Unable to open head_gaussians.ply");
            }
        } else {
            LOGI("No head points to save.");
        }

        size_t other_numPoints = other_positions.size() / 3;
        if (other_numPoints > 0) {
            std::string other_gaussians_path = fullDataPath + "/head/other_gaussians.ply";
            
            std::ofstream otherOutFile(other_gaussians_path, std::ios::binary);
            if (otherOutFile.is_open()) {
                tinyply::PlyFile otherPlyFile;
                otherPlyFile.add_properties_to_element("vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, other_numPoints, reinterpret_cast<uint8_t*>(other_positions.data()), tinyply::Type::INVALID, 0);
                otherPlyFile.add_properties_to_element("vertex", {"scale_0", "scale_1", "scale_2"}, tinyply::Type::FLOAT32, other_numPoints, reinterpret_cast<uint8_t*>(other_scales.data()), tinyply::Type::INVALID, 0);
                otherPlyFile.add_properties_to_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}, tinyply::Type::FLOAT32, other_numPoints, reinterpret_cast<uint8_t*>(other_rotations.data()), tinyply::Type::INVALID, 0);
                otherPlyFile.add_properties_to_element("vertex", {"opacity"}, tinyply::Type::FLOAT32, other_numPoints, reinterpret_cast<uint8_t*>(other_opacities.data()), tinyply::Type::INVALID, 0);
                otherPlyFile.add_properties_to_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"}, tinyply::Type::FLOAT32, other_numPoints, reinterpret_cast<uint8_t*>(other_colors.data()), tinyply::Type::INVALID, 0);

                for (int i = 0; i < 45; ++i) {
                    otherPlyFile.add_properties_to_element("vertex", {"f_rest_" + std::to_string(i)}, tinyply::Type::FLOAT32, other_numPoints, reinterpret_cast<uint8_t*>(&other_f_rest[i * other_numPoints]), tinyply::Type::INVALID, 0);
                }

                otherPlyFile.write(otherOutFile, true);
                otherOutFile.close();
                LOGI("Saved %{public}zu other points to other_gaussians.ply", other_numPoints);
            } else {
                LOGE("Error: Unable to open other_gaussians.ply");
            }
        } else {
            LOGI("No other points to save.");
        }
    } 
    else {
        std::string filename = fullDataPath + "/model.ply";
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile.is_open()) {
            LOGE("Error: Unable to open file %{public}s", filename.c_str());
            return;
        } else {
            LOGI("Saving gaussians to %{public}s", filename.c_str());
        }

        tinyply::PlyFile plyFile;
        size_t numPoints = data.positions.size();

        std::vector<float> positions, scales, rotations, opacities, colors, f_rest;
        positions.reserve(numPoints * 3);
        scales.reserve(numPoints * 3);
        rotations.reserve(numPoints * 4);
        opacities.reserve(numPoints);
        colors.reserve(numPoints * 3);
        f_rest.reserve(numPoints * 45);
        std::vector<float> normals(numPoints * 3, 0.0);
        
        for (size_t i = 0; i < numPoints; ++i) {
            Base::Math::Vec3 pos = data.positions[i];
            Base::Math::Vec3 scale = (i < data.scales.size()) ? data.scales[i]  : Base::Math::Vec3(1.0f, 1.0f, 1.0f);
            Base::Math::Vec4 rotation = (i < data.rotations.size()) ? data.rotations[i] : Base::Math::Vec4(0, 0, 0, 1);
            
            float opacity = (i < data.opacities.size()) ? data.opacities[i] : 1.0f;
            Base::Math::Vec3 color(1.0f, 1.0f, 1.0f);
            std::vector<float> f_rest_values;

            if (i < data.colors.size()) {
                color = Base::Math::Vec3(
                    data.colors[i].values[0 * 16],
                    data.colors[i].values[1 * 16],
                    data.colors[i].values[2 * 16]
                );
                for (int j = 1; j < 16; ++j) {
                    f_rest_values.push_back(data.colors[i].values[0 * 16 + j]);
                    f_rest_values.push_back(data.colors[i].values[1 * 16 + j]);
                    f_rest_values.push_back(data.colors[i].values[2 * 16 + j]);
                }
            }

            positions.insert(positions.end(), {pos.x, pos.y, pos.z});
            scales.insert(scales.end(), {scale.x, scale.y, scale.z});
            rotations.insert(rotations.end(), {rotation.x, rotation.y, rotation.z, rotation.w});
            opacities.push_back(opacity);
            colors.insert(colors.end(), {color.x, color.y, color.z});
            f_rest.insert(f_rest.end(), f_rest_values.begin(), f_rest_values.end());
        }

        plyFile.add_properties_to_element("vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(positions.data()), tinyply::Type::INVALID, 0);
        plyFile.add_properties_to_element("vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(normals.data()), tinyply::Type::INVALID, 0);
        plyFile.add_properties_to_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(colors.data()), tinyply::Type::INVALID, 0);
        
        for (int i = 0; i < 45; ++i) {
            plyFile.add_properties_to_element("vertex", {"f_rest_" + std::to_string(i)}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(&f_rest[i * numPoints]), tinyply::Type::INVALID, 0);
        }

        plyFile.add_properties_to_element("vertex", {"opacity"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(opacities.data()), tinyply::Type::INVALID, 0);
        plyFile.add_properties_to_element("vertex", {"scale_0", "scale_1", "scale_2"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(scales.data()), tinyply::Type::INVALID, 0);
        plyFile.add_properties_to_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(rotations.data()), tinyply::Type::INVALID, 0);
        
        plyFile.write(outFile, true);
        outFile.close();
        std::cout << "Saved " << numPoints << " points to " << filename << " using tinyply." << std::endl;
    }
}


void DataLoader::saveCameras(const std::string fullDataPath, const DataLoader::InputInitStruct& data){
    std::string extrName = fullDataPath + "/images.txt";
    std::string intrName = fullDataPath + "/cameras.txt";

    std::ofstream extrFile(extrName);
    std::ofstream intrFile(intrName);

    if (!extrFile.is_open() && !intrFile.is_open()) {
        LOGE("Error: Unable to open file %{public}s and %{public}s", extrName.c_str(), intrName.c_str());
        return;
    }
    assert(data.img_list.size() == data.extr_tmp.size());
    assert(data.intr_tmp.size() == data.extr_tmp.size() || data.intr_tmp.size() == 1);
    
    // 1. cameras.txt
    intrFile << "# CMAERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx, fy, cx, cy]\n";
    for (int i = 0; i < data.intr_tmp.size(); i++) {
        intrFile << "0 PINHOLE " << data.width << " " << data.height << " ";
        intrFile << data.intr_tmp[i][0] << " " << data.intr_tmp[i][1] << " " << data.intr_tmp[i][2] << " " << data.intr_tmp[i][3] << "\n";
    }
    intrFile.close();

    // 2. images.txt
    extrFile << "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    for(int i = 0; i < data.extr_tmp.size(); i++) {
        std::string filename = std::filesystem::path(data.img_list[i]).filename().string();
        extrFile << i << " " << data.extr_tmp[i].values[0] << " ";
        extrFile << data.extr_tmp[i].values[1] << " ";
        extrFile << data.extr_tmp[i].values[2] << " ";
        extrFile << data.extr_tmp[i].values[3] << " ";
        extrFile << data.extr_tmp[i].values[4] << " ";
        extrFile << data.extr_tmp[i].values[5] << " ";
        extrFile << data.extr_tmp[i].values[6] << " ";
        extrFile << "0 " << filename << "\n\n";
    }
    extrFile.close();
}


void load3DPoints(std::string& pointsFilePath, DataLoader::InputInitStruct& data){
    std::ifstream pointsFile(pointsFilePath);
    if (!pointsFile.is_open()) {
        LOGE("Failed to open %{public}s file.", pointsFilePath.c_str());
    }
    std::string line;
    std::unordered_set<Base::Math::Vec3, Vec3Hash, Vec3Equal> uniquePoints;
    float C0 = 0.28209479177387814;
    while (std::getline(pointsFile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int pointId;
        float x, y, z;
        int r, g, b;
        float alpha; // Although alpha is provided, we will not use it.
        if (!(iss >> pointId >> x >> y >> z >> r >> g >> b >> alpha)) {
            continue; // Skip invalid lines.
        }

        Base::Math::Vec3 pos(x, y, z);
        // Check for duplicate points
        if (uniquePoints.find(pos) != uniquePoints.end()) {
            continue;
        }
        uniquePoints.insert(pos);
        
        // Store the point position.
        data.positions.push_back(pos);
        
        // Default attributes (scales, rotations, opacities) to zeros.
        data.scales.push_back(Base::Math::Vec3(-4.5f, -4.5f, -4.5f));
        data.rotations.push_back(Base::Math::Vec4(1.0f, 0.0f, 0.0f, 0.0f));

        data.opacities.push_back(-2.197225f);
        
        // Process the color data.
        // We interpret r, g, b (from points3D.txt) as f_dc_0, f_dc_1, and f_dc_2.
        DataLoader::InputInitStruct::Data colorData = {};

        // Save r in index 0, g in index 16, and b in index 32.
        colorData.values[0]  = static_cast<float>((r/255.0-0.5)/C0);
        colorData.values[16] = static_cast<float>((g/255.0-0.5)/C0);
        colorData.values[32] = static_cast<float>((b/255.0-0.5)/C0);
        data.colors.push_back(colorData);
    }
    LOGI("Unique points loaded: %{public}zu", data.positions.size());
    pointsFile.close();
}



std::vector<cv::Point> DataLoader::projectHeadCenter(DataLoader::InputInitStruct& data)
{
    // Calculate the center of the 3D bounding box
    float center_x = (data.head_bbox[0] + data.head_bbox[1]) / 2.0f;
    float center_y = (data.head_bbox[2] + data.head_bbox[3]) / 2.0f;
    float center_z = (data.head_bbox[4] + data.head_bbox[5]) / 2.0f;
    
    // Create a 3D point for the center in homogeneous coordinates
    Eigen::Vector4f point_3d(center_x, center_y, center_z, 1.0f);
    
    // Vector to store results
    std::vector<cv::Point> projected_points;
    projected_points.reserve(data.extr_tmp.size());
    
    // For each camera pose
    for (size_t i = 0; i < data.extr_tmp.size(); ++i) {
        // Get the current extrinsics
        const InputInitStruct::DataExtr extr = data.extr_tmp[i];
        
        // Get the corresponding intrinsics (either shared or per-camera)
        const std::vector<float> intr = (data.intr_tmp.size() == 1) ? data.intr_tmp[0] : data.intr_tmp[i];
        if (intr.size() != 4) {
            throw std::invalid_argument("Intrinsics must have 4 elements (fx, fy, cx, cy)");
        }
        
        // Extract quaternion components and translation vector
        float qw = extr.values[0];
        float qx = extr.values[1];
        float qy = extr.values[2];
        float qz = extr.values[3];
        float tx = extr.values[4];
        float ty = extr.values[5];
        float tz = extr.values[6];
        
        // Convert quaternion to rotation matrix
        Eigen::Quaternionf q(qw, qx, qy, qz);
        Eigen::Matrix3f R = q.normalized().toRotationMatrix();
        
        // Create transformation matrix (extrinsic matrix)
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = Eigen::Vector3f(tx, ty, tz);
        
        // Transform 3D point from world to camera coordinates
        Eigen::Vector4f point_camera = T * point_3d;
        
        // Check if the point is in front of the camera
        if (point_camera[2] <= 0) {
            // Point is behind the camera, store an invalid point
            projected_points.push_back(cv::Point(-1, -1));
            continue;
        }
        
        // Project 3D point to 2D image coordinates
        float fx = intr[0];
        float fy = intr[1];
        float cx = intr[2];
        float cy = intr[3];
        
        float x_2d = (fx * point_camera[0] / point_camera[2]) + cx;
        float y_2d = (fy * point_camera[1] / point_camera[2]) + cy; 
        LOGI("x_2d: %{public}f, y_2d: %{public}f", x_2d, y_2d);
        // Store the projected 2D point
        projected_points.push_back(cv::Point(x_2d, y_2d));
    }
    
    return projected_points;
}
#if 0
void DataLoader::createEnlargedFacev1(const std::string scenepath, DataLoader::InputInitStruct& data, 
                                   std::vector<cv::Point>& faceCenters, std::vector<std::string>& selectedImages) {
    // Create directories
    std::filesystem::create_directories(scenepath + "/human");
    std::filesystem::create_directories(scenepath + "/human/images");
    
    // Copy intr_tmp multiple times to support adding face regions
    if (data.intr_tmp.size() == 1 && data.extr_tmp.size() > 1) {
        size_t N = data.extr_tmp.size();
        std::vector<float> first_intrinsic = data.intr_tmp[0];
        data.intr_tmp.resize(N, first_intrinsic);
    }
    
    cv::Mat orig_img = cv::imread(data.img_list[0]);
    float scale_factor = orig_img.size[0] / 640;
    
    // Parse keypoints file to calculate optimal face dimensions
    std::string keypointsPath = scenepath + "/keypoints.txt";
    std::vector<KeypointData> keypoints = parseKeypointsFile(keypointsPath);
    cv::Size optimalFaceSize = calculateOptimalFaceDimensions(keypoints);
    optimalFaceSize = cv::Size(optimalFaceSize.width * scale_factor, optimalFaceSize.height * scale_factor);
    
    LOGI("Optimal face dimensions: width=%{public}d, height=%{public}d", 
         optimalFaceSize.width, optimalFaceSize.height);
    std::vector<cv::Rect2d> roi_bboxes;
    // Process each face center
    for (int i = 0; i < faceCenters.size(); ++i) {
        if(faceCenters[i].x <= 0 || faceCenters[i].y <= 0 || faceCenters[i].x >= orig_img.cols || faceCenters[i].y >= orig_img.rows){
            continue;
        }
        std::string basename = getBaseName(data.img_list[i]);
        
        // Read original image
        cv::Mat orig_img = cv::imread(data.img_list[i]);
        cv::Point scaledFaceCenter = faceCenters[i] * scale_factor;
        if(basename == getBaseName(selectedImages[0]) || basename == getBaseName(selectedImages[1])){
            cv::Point2d center(scaledFaceCenter.x, scaledFaceCenter.y);
            cv::Point2d half(optimalFaceSize.width * 0.5, optimalFaceSize.height * 0.5);
            cv::Point2d pt1 = center - half;
            cv::Point2d pt2 = center + half;
            roi_bboxes.push_back(cv::Rect2d(pt1, pt2));
        }
        // Schedule 2: Optimal face crops (no intrinsics calculation)
        auto begin = clk::now();
        cropAndSaveFace(orig_img, scaledFaceCenter, optimalFaceSize, scenepath, basename);
        LOGW("Beauty filter takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        
        // Schedule 1: Original 480x640 crops with intrinsics calculation
        cv::Point topLeft;
        cv::Mat cropped_img_480x640 = cropImage(orig_img, scaledFaceCenter, 
                                                cv::Size(480, 640), topLeft);
        cv::imwrite(scenepath + "/human/images/" + basename, cropped_img_480x640);
        data.img_list.push_back(scenepath + "/human/images/" + basename);
        
        // Calculate new intrinsics for the 480x640 crop
        std::vector<float> intr = data.intr_tmp[0];
        std::vector<float> new_intr = calculateNewIntrinsics(intr[0], intr[1], intr[2], intr[3], 
                                                            topLeft, cv::Size(480, 640), scale_factor);
        LOGI("fx:%{public}f fy:%{public}f cx:%{public}f cy:%{public}f", 
             new_intr[0], new_intr[1], new_intr[2], new_intr[3]);
        data.intr_tmp.push_back(new_intr);
        data.extr_tmp.push_back(data.extr_tmp[i]);
        data.time_stamps.push_back(data.time_stamps[i]);
    }
    data.extr_tmp_test = data.extr_tmp;
}
#endif

// Modified createEnlargedFace function
void DataLoader::createEnlargedFace(const std::string scenepath, DataLoader::InputInitStruct& data, 
                                   const std::vector<FaceProcessingInfo>& faceInfos,
                                   const std::pair<int, int>& selectedIndices) {
    // Create directories
    std::filesystem::create_directories(scenepath + "/human");
    std::filesystem::create_directories(scenepath + "/human/images");
    std::filesystem::create_directories(scenepath + "/human/faces");
    
    // Copy intr_tmp multiple times to support adding face regions
    if (data.intr_tmp.size() == 1 && data.extr_tmp.size() > 1) {
        size_t N = data.extr_tmp.size();
        std::vector<float> first_intrinsic = data.intr_tmp[0];
        data.intr_tmp.resize(N, first_intrinsic);
    }
    
    cv::Mat orig_img = cv::imread(data.img_list[0]);
    float scale_factor = orig_img.size[0] / 640.0;
    
    // Parse keypoints file to calculate optimal face dimensions
    std::string keypointsPath = scenepath + "/keypoints.txt";
    std::vector<KeypointData> keypoints = parseKeypointsFile(keypointsPath);
    cv::Size optimalFaceSize = calculateOptimalFaceDimensions(keypoints);
    optimalFaceSize = cv::Size(optimalFaceSize.width * scale_factor, optimalFaceSize.height * scale_factor);
    
    LOGI("Optimal face dimensions: width=%{public}d, height=%{public}d", 
         optimalFaceSize.width, optimalFaceSize.height);
    
    std::vector<cv::Rect2d> roi_bboxes;
    
    // Process each face
    for (int i = 0; i < faceInfos.size(); ++i) {
        if (!faceInfos[i].isWithinBounds) {
            continue;
        }
        
        std::string basename = faceInfos[i].imageName;
        int imgIndex = -1;
        
        // Find the corresponding image index
        for (int j = 0; j < data.img_list.size(); ++j) {
            if (getBaseName(data.img_list[j]) == basename) {
                imgIndex = j;
                break;
            }
        }
        
        if (imgIndex == -1) {
            continue;
        }
        
        // Read original image
        cv::Mat orig_img = cv::imread(data.img_list[imgIndex]);
        cv::Point scaledFaceCenter = faceInfos[i].faceCenter;
        
        // Check if this is one of the selected images and calculate roi_bbox
        if (i == selectedIndices.first || i == selectedIndices.second) {
            cv::Point2d center(scaledFaceCenter.x, scaledFaceCenter.y);
            cv::Point2d half(optimalFaceSize.width * 0.5, optimalFaceSize.height * 0.5);
            cv::Point2d pt1 = center - half;
            cv::Point2d pt2 = center + half;
            roi_bboxes.push_back(cv::Rect2d(pt1, pt2));
        }
        
        // Schedule 2: Optimal face crops (no intrinsics calculation)
        auto begin = clk::now();
        cropAndSaveFace(orig_img, scaledFaceCenter, optimalFaceSize, scenepath, basename);
        LOGW("Beauty filter takes %{public}f seconds", std::chrono::duration<double>(clk::now() - begin).count());
        
        // Schedule 1: Original 480x640 crops with intrinsics calculation
        cv::Point topLeft;
        orig_img = cv::imread(data.img_list[imgIndex]); // Reload beautifed face
        cv::Mat cropped_img_480x640 = cropImage(orig_img, scaledFaceCenter, 
                                                cv::Size(480, 640), topLeft);
        cv::imwrite(scenepath + "/human/images/" + basename, cropped_img_480x640);
        data.img_list.push_back(scenepath + "/human/images/" + basename);
        
        // Calculate new intrinsics for the 480x640 crop
        std::vector<float> intr = data.intr_tmp[0];
        std::vector<float> new_intr = calculateNewIntrinsics(intr[0], intr[1], intr[2], intr[3], 
                                                            topLeft, cv::Size(480, 640), scale_factor);
        LOGI("fx:%{public}f fy:%{public}f cx:%{public}f cy:%{public}f", 
             new_intr[0], new_intr[1], new_intr[2], new_intr[3]);
        data.intr_tmp.push_back(new_intr);
        data.extr_tmp.push_back(data.extr_tmp[imgIndex]);
        data.time_stamps.push_back(data.time_stamps[imgIndex]);
    }
    
    data.extr_tmp_test = data.extr_tmp;
}

// calculate robust min/max
void computeRobustBounds(
    const std::vector<Base::Math::Vec3>& pts,
    float percentile,
    Base::Math::Vec3& out_min,
    Base::Math::Vec3& out_max
)
{
    size_t n = pts.size();
    if (n == 0) 
        return;
    
    std::vector<float> xs(n), ys(n), zs(n);
    for (size_t i = 0; i < n; ++i) {
        xs[i] = pts[i].x;
        ys[i] = pts[i].y;
        zs[i] = pts[i].z;
    }
    
    auto getQuantile = [&](std::vector<float>& arr, float q) {
        std::sort(arr.begin(), arr.end());
        float idx = q * (arr.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(idx));
        size_t hi = static_cast<size_t>(std::ceil(idx));
        if (hi == lo) return arr[lo];
        float t = idx - lo;
        return arr[lo] * (1 - t) + arr[hi] * t;
    };
    
    out_min.x = getQuantile(xs, percentile);
    out_min.y = getQuantile(ys, percentile);
    out_min.z = getQuantile(zs, percentile);

    out_max.x = getQuantile(xs, 1.0f - percentile);
    out_max.y = getQuantile(ys, 1.0f - percentile);
    out_max.z = getQuantile(zs, 1.0f - percentile);
    
}

// Fibonacci sphere sampling
void generateFibonacciSphere(
    int samples,
    std::vector<Base::Math::Vec3>& out_points,
    std::vector<Base::Math::Vec3>& out_normals
)
{
    out_points.reserve(samples);
    out_normals.reserve(samples);
    const float goldenAngle = M_PI * (3.0f - std::sqrt(5.0f));
    for (int i = 0; i < samples; ++i) {
        float t = float(i) / float(samples - 1);             // [0,1]
        float y = 1.0f - 2.0f * t;                          // [-1,1]
        float radius = std::sqrt(1.0f - y * y);
        float theta = goldenAngle * i;
        float x = std::cos(theta) * radius;
        float z = std::sin(theta) * radius;
        Base::Math::Vec3 dir(x, y, z);
        out_points.push_back(dir);
        Base::Math::Vec3 normal = -dir;                            // 指向中心
        out_normals.push_back(Base::Math::Normalize(normal));
    }
}

void prependSkyboxPoints(
    DataLoader::InputInitStruct& data,
    int sky_num_gs,
    float percentile = 0.01f,
    float radius_scale = 1.0f
)
{
    const float C0 = 0.28209479177387814f;
    // 1. calc min/max
    Base::Math::Vec3 xyz_min, xyz_max;
    computeRobustBounds(data.positions, percentile, xyz_min, xyz_max);
    
    // 2. calc direction and normal in shpere
    std::vector<Base::Math::Vec3> dirs, norms;
    generateFibonacciSphere(sky_num_gs, dirs, norms);

    // 3. calc scaling and translation
    Base::Math::Vec3 center = (xyz_min + xyz_max) * 0.5f;
    Base::Math::Vec3 halfSize = (xyz_max - xyz_min) * 0.5f * radius_scale;
    float whiteScaled = (1.0f - 0.5f) / C0;
    
    // 4. generate all the sky GS
    std::vector<Base::Math::Vec3>       skyPositions;
    std::vector<Base::Math::Vec3>       skyScales;
    std::vector<Base::Math::Vec4>       skyRotations;
    std::vector<float>           skyOpacities;
    std::vector<DataLoader::InputInitStruct::Data> skyColors;

    skyPositions.reserve(sky_num_gs);
    skyScales.reserve(sky_num_gs);
    skyRotations.reserve(sky_num_gs);
    skyOpacities.reserve(sky_num_gs);
    skyColors.reserve(sky_num_gs);

    for (int i = 0; i < sky_num_gs; ++i) {
        Base::Math::Vec3 pos = dirs[i] * halfSize + center;
        skyPositions.push_back(pos);
        
        skyScales.push_back(Base::Math::Vec3(-4.5f, -4.5f, -4.5f));
        skyRotations.push_back(Base::Math::Vec4(1.0f, 0.0f, 0.0f, 0.0f));
        skyOpacities.push_back(-2.197224617004395f);
        
        DataLoader::InputInitStruct::Data cd{};
        cd.values[0] = whiteScaled;
        cd.values[16] = whiteScaled;
        cd.values[32] = whiteScaled;
        skyColors.push_back(cd);
    }
    
    data.positions.insert(data.positions.begin(), skyPositions.begin(), skyPositions.end());
    data.scales.insert(data.scales.begin(), skyScales.begin(), skyScales.end());
    data.rotations.insert(data.rotations.begin(), skyRotations.begin(), skyRotations.end());
    data.opacities.insert(data.opacities.begin(), skyOpacities.begin(), skyOpacities.end());
    data.colors.insert(data.colors.begin(), skyColors.begin(), skyColors.end());
}

void DataLoader::loadInput12(const std::string _scenepath, DataLoader::InputInitStruct& data, bool& resumed, int height, int width)
{
    std::string basePath = data.basepath;
    std::string scenepath = _scenepath;
    std::string slampath = scenepath;
    std::string mediapath = scenepath + "/media";

    std::string imagesFilePath = slampath + "/images.txt";
    std::string cameraFilePath = slampath + "/cameras.txt";             // TODO: slampath?
    std::string videoInfoFilePath = mediapath + "/videoInfo.txt";
//     std::filesystem::create_directories(scenepath + "/temp");
    
    // 2. Load intrinsics from cameras.txt
//     int height, width;
    std::string line;
    std::ifstream camerasFile(cameraFilePath);
    if (!camerasFile.is_open()){
        LOGE("Cannot open cameras.txt!");
    }
    while (std::getline(camerasFile, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        int cameraId;
        std::string camera_type;
        float fx, fy, cx, cy;
        
        // Read the first 6 tokens:
        // TODO: media: height >> width
        // TODO:  slam: width >> height

        if (cameraFilePath == mediapath + "/cameras.txt") {
            if (!(iss >> cameraId >> camera_type >> data.height >> data.width >> fx >> fy >> cy >> cx)) {
                continue;
            }
        }
        else if (cameraFilePath == slampath + "/cameras.txt") {
            if (!(iss >> cameraId >> camera_type >> data.width >> data.height >> fx >> fy >> cx >> cy)) {
                continue;
            }
        }
            
        // width = 480, height = 640
        float ratiox = 1.0 * width / data.width;
        float ratioy = 1.0 * height / data.height;
        fx *= ratiox;
        fy *= ratioy;
        cx *= ratiox;
        cy *= ratioy;
        data.width = width;
        data.height = height;

        assert(camera_type == "PINHOLE" || camera_type == "pinhole");

        std::vector<float> intr = { fx,fy,cx,cy };
        data.intr_tmp.push_back(intr);
    }
    
    // 3. Load image file names and extrinsics from images.txt
    std::unordered_set<std::string> uniqueImages;
    std::string ext;
    std::ifstream imagesFile(imagesFilePath);
    if (!imagesFile.is_open()){
        throw std::runtime_error("Cannot open images.txt!");
    }
    
    int idx = 0;
    while (std::getline(imagesFile, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        int frameId;
        float qw, qx, qy, qz;
        float tx, ty, tz;
        int camera_id;
        std::string imageName;
        
        if (!(iss >> frameId >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> imageName)) {
            continue;
        }
        
        if (uniqueImages.find(imageName) != uniqueImages.end()) {
            continue;
        }
        uniqueImages.insert(imageName);
        
        if(!std::filesystem::exists(scenepath + "/images/" + imageName))
            continue;
            
        float norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        DataLoader::InputInitStruct::DataExtr extr = { qw / norm, qx / norm, qy / norm, qz / norm, tx, ty, tz };
        data.extr_tmp.push_back(extr);
        data.img_list.push_back(scenepath + "/images/" + imageName);
        data.time_stamps.push_back(float(idx));
        ++idx;

        if (ext.size() == 0){
            size_t dotPos = imageName.rfind('.');
            if (dotPos != std::string::npos) {
                ext = imageName.substr(dotPos);
            }
        }
        std::getline(imagesFile, line);
    }
    for (auto& time : data.time_stamps) {
        time /= float(data.time_stamps.size());
    }
    
    data.extr_tmp_test = data.extr_tmp;
    assert(data.intr_tmp.size() == 1 || data.intr_tmp.size() == data.extr_tmp.size());

    // 1. Load video data
    // Main code replacing the OpenCV section
    std::vector<std::string> allVideoFile;
    getFilesInDirectory(scenepath, "mp4", allVideoFile);
    if (allVideoFile.size() == 0) {
        LOGE("No video file!");
    }
    
    std::string resume_gaussian_path = scenepath + "/point_cloud.ply";
    loadInputGaussianData(resume_gaussian_path, data);
    if (data.positions.size() == 0){
        LOGW("Resuming from ply failed, starting from points3D.txt");
        std::map<std::string, double> faceAngles;
        bool is_human = HumanPreprocess(scenepath, data.head_bbox, data.body_bbox, faceAngles);
        
        std::string pointsFilePath = slampath + "/points3D.txt";
        load3DPoints(pointsFilePath, data);
        
        if (is_human) {
            std::string smplPath = scenepath + "/smpl.txt";
            load3DPoints(smplPath, data);
            for(int i=0; i<data.body_bbox.size(); ++i){
                LOGI("body bbox: %{public}f", data.body_bbox[i]);
            }
            
            // Calculate face centers
            std::vector<cv::Point> faceCenters = projectHeadCenter(data);
            
            // Analyze faces without requiring saved images
            std::vector<FaceProcessingInfo> faceInfos = analyzeFaces(faceAngles, faceCenters, data.img_list);
            
            // Select two images based on analysis
            std::pair<int, int> selectedIndices = selectTwoImageIndices(faceInfos);
            
            if (selectedIndices.first != -1 && selectedIndices.second != -1) {
                // Process faces and create enlarged faces
                data.max_movement = (data.body_bbox[3] - data.body_bbox[2]) * 0.01;
                createEnlargedFace(scenepath, data, faceInfos, selectedIndices);
                
                // Now we can build the selected image paths for ProcessLandmarks
                std::vector<std::string> selectedImages;
                selectedImages.push_back(scenepath + "/human/faces/" + faceInfos[selectedIndices.first].imageName);
                selectedImages.push_back(scenepath + "/human/faces/" + faceInfos[selectedIndices.second].imageName);
                
                // ProcessLandmarks(scenepath, selectedImages);
            }
        }
    }
    else{
        resumed = true;
    }
    
    prependSkyboxPoints(data, 10000, 0.01, 10.0);

    imagesFile.close();
    camerasFile.close();
}