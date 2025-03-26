#include "dataloader.h"
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <dirent.h>
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
    #include <libavutil/display.h>
}
#include <unordered_set>
#include "human/human.h"

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
        // glm::vec3
        data.xyz.emplace_back(
            reinterpret_cast<float*>(x->buffer.get())[i],
            reinterpret_cast<float*>(y->buffer.get())[i],
            reinterpret_cast<float*>(z->buffer.get())[i]
        );

        // glm::vec3
        data.scale.emplace_back(
            reinterpret_cast<float*>(scale_x->buffer.get())[i],
            reinterpret_cast<float*>(scale_y->buffer.get())[i],
            reinterpret_cast<float*>(scale_z->buffer.get())[i]
        );

        // glm::vec4 
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
    size_t operator()(const glm::vec3& v) const {
        std::hash<float> floatHasher;
        size_t h1 = floatHasher(v.x);
        size_t h2 = floatHasher(v.y);
        size_t h3 = floatHasher(v.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);  // 组合哈希值
    }
};

struct Vec3Equal {
    bool operator()(const glm::vec3& v1, const glm::vec3& v2) const {
        return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
    }
};

glm::mat4 quat2matrix(DataLoader::InputInitStruct::DataExtr& extr) {
    glm::mat4 W2C(1.0f);
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
}

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
    
    this->mean = mean;
    this->variance = variance;
    
    std::cout << "Position Mean: (" << mean.x << ", " << mean.y << ", " << mean.z << ")" << std::endl;
    std::cout << "Position Variance: " << variance << std::endl;  // **variance 现在是单个 float**
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
        glm::vec3 pos(
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
        glm::vec3 pos(
            reinterpret_cast<float*>(x->buffer.get())[i],
            reinterpret_cast<float*>(y->buffer.get())[i],
            reinterpret_cast<float*>(z->buffer.get())[i]
        );
        data.positions.push_back(pos);

        glm::vec3 scale(
            reinterpret_cast<float*>(scale_x->buffer.get())[i],
            reinterpret_cast<float*>(scale_y->buffer.get())[i],
            reinterpret_cast<float*>(scale_z->buffer.get())[i]
        );
        data.scales.push_back(scale);

        glm::vec4 rotate(
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
    std::string resume_gaussian_path = data.basepath + dataPath + "/head/point_cloud.ply";
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
        float stddev = glm::sqrt(data.variance) + 1e-6f;

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
            glm::vec3 pos = data.positions[i];
//             glm::vec3 orig_pos = data.positions[i] * stddev + data.mean;
            
            bool inside = (pos.x >= x_min && pos.x <= x_max) &&
                          (pos.y >= y_min && pos.y <= y_max) &&
                          (pos.z >= z_min && pos.z <= z_max);
            LOGI("x: %{public}f, y: %{public}f, z: %{public}f", pos.x, pos.y, pos.z);
            glm::vec3 scale = (i < data.scales.size()) ? data.scales[i] : glm::vec3(1.0f);
            glm::vec4 rotation = (i < data.rotations.size()) ? data.rotations[i] : glm::vec4(0, 0, 0, 1);
            float opacity = (i < data.opacities.size()) ? data.opacities[i] : 1.0f;
            glm::vec3 color(1.0f, 1.0f, 1.0f);
            std::vector<float> f_rest_values;

            if (i < data.colors.size()) {
                color = glm::vec3(
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
    } else {
        std::string filename = fullDataPath + "/point_cloud.ply";
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

        float stddev = glm::sqrt(data.variance) + 1e-6f;
        for (size_t i = 0; i < numPoints; ++i) {
//             glm::vec3 pos = data.positions[i] * stddev + data.mean;
//             glm::vec3 scale = (i < data.scales.size()) ? data.scales[i] + glm::log(stddev) : glm::vec3(1.0f);
//             glm::vec4 rotation = (i < data.rotations.size()) ? data.rotations[i] : glm::vec4(0, 0, 0, 1);
            
            glm::vec3 pos = data.positions[i];
            glm::vec3 scale = (i < data.scales.size()) ? data.scales[i]  : glm::vec3(1.0f);
            glm::vec4 rotation = (i < data.rotations.size()) ? data.rotations[i] : glm::vec4(0, 0, 0, 1);
            
            float opacity = (i < data.opacities.size()) ? data.opacities[i] : 1.0f;
            glm::vec3 color(1.0f, 1.0f, 1.0f);
            std::vector<float> f_rest_values;

            if (i < data.colors.size()) {
                color = glm::vec3(
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
        plyFile.add_properties_to_element("vertex", {"scale_0", "scale_1", "scale_2"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(scales.data()), tinyply::Type::INVALID, 0);
        plyFile.add_properties_to_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(rotations.data()), tinyply::Type::INVALID, 0);
        plyFile.add_properties_to_element("vertex", {"opacity"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(opacities.data()), tinyply::Type::INVALID, 0);
        plyFile.add_properties_to_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(colors.data()), tinyply::Type::INVALID, 0);

        for (int i = 0; i < 45; ++i) {
            plyFile.add_properties_to_element("vertex", {"f_rest_" + std::to_string(i)}, tinyply::Type::FLOAT32, numPoints, reinterpret_cast<uint8_t*>(&f_rest[i * numPoints]), tinyply::Type::INVALID, 0);
        }

        plyFile.write(outFile, true);
        outFile.close();
        std::cout << "Saved " << numPoints << " points to " << filename << " using tinyply." << std::endl;
    }
}


void load3DPoints(std::string& pointsFilePath, DataLoader::InputInitStruct& data){
    std::ifstream pointsFile(pointsFilePath);
    if (!pointsFile.is_open()) {
        LOGE("Failed to open %{public}s file.", pointsFilePath.c_str());
    }
    std::string line;
    std::unordered_set<glm::vec3, Vec3Hash, Vec3Equal> uniquePoints;
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

        glm::vec3 pos(x, y, z);
        // Check for duplicate points
        if (uniquePoints.find(pos) != uniquePoints.end()) {
            continue;
        }
        uniquePoints.insert(pos);
        
        // Store the point position.
        data.positions.push_back(pos);
        
        // Default attributes (scales, rotations, opacities) to zeros.
        data.scales.push_back(glm::vec3(-4.5f, -4.5f, -4.5f));
        data.rotations.push_back(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));

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

// Function to save a frame as an image
bool saveFrameAsImage(AVFrame* frame, const std::string& filename, int width, int height) {
    // Check file extension
    std::string ext = filename.substr(filename.find_last_of("."));
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // Use MJPEG encoding for image output
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!codec) {
        LOGE("MJPEG codec not found");
        return false;
    }
    
    AVCodecContext* codecContext = avcodec_alloc_context3(codec);
    if (!codecContext) {
        LOGE("Could not allocate codec context");
        return false;
    }
    
    codecContext->width = width;
    codecContext->height = height;
    codecContext->time_base = {1, 25};
    codecContext->pix_fmt = AV_PIX_FMT_YUVJ420P;  // JPEG format
    codecContext->qmin = 1;  // High quality
    codecContext->qmax = 1;
    
    if (avcodec_open2(codecContext, codec, nullptr) < 0) {
        avcodec_free_context(&codecContext);
        return false;
    }
    
    // Convert RGB to YUV for JPEG
    AVFrame* yuvFrame = av_frame_alloc();
    if (!yuvFrame) {
        avcodec_free_context(&codecContext);
        return false;
    }
    
    yuvFrame->width = width;
    yuvFrame->height = height;
    yuvFrame->format = AV_PIX_FMT_YUVJ420P;
    
    if (av_frame_get_buffer(yuvFrame, 0) < 0) {
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    SwsContext* swsContext = sws_getContext(
        width, height, AV_PIX_FMT_RGB24,
        width, height, AV_PIX_FMT_YUVJ420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    
    if (!swsContext) {
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    if (sws_scale(swsContext, frame->data, frame->linesize, 0, height,
                 yuvFrame->data, yuvFrame->linesize) <= 0) {
        sws_freeContext(swsContext);
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    sws_freeContext(swsContext);
    
    // Encode frame to JPEG
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    if (avcodec_send_frame(codecContext, yuvFrame) < 0) {
        av_packet_free(&packet);
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    int ret = avcodec_receive_packet(codecContext, packet);
    if (ret < 0) {
        av_packet_free(&packet);
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    // Write JPEG data to file
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) {
        av_packet_free(&packet);
        av_frame_free(&yuvFrame);
        avcodec_free_context(&codecContext);
        return false;
    }
    
    fwrite(packet->data, 1, packet->size, f);
    fclose(f);
    
    // Clean up
    av_packet_free(&packet);
    av_frame_free(&yuvFrame);
    avcodec_free_context(&codecContext);
    
    return true;
}

// Get the rotation angle from video metadata
int getVideoRotation(AVStream* stream) {
    AVDictionaryEntry* rotateTag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
    if (rotateTag) {
        // Rotation specified in the metadata
        int angle = std::stoi(rotateTag->value);
        return angle;
    }
    
    // Check for display matrix
    uint8_t* displayMatrix = av_stream_get_side_data(stream, AV_PKT_DATA_DISPLAYMATRIX, nullptr);
    if (displayMatrix) {
        double rotation = av_display_rotation_get((int32_t*)displayMatrix);
        if(rotation < 0){
            return 360 + (int)rotation;
        }
        return (int)(rotation);
    }
    
    return 0; // No rotation
}

// Process video with FFmpeg
bool processVideoWithFFmpeg(
    const std::string& videoPath, 
    const std::vector<std::string>& timestamps,
    const std::unordered_set<std::string>& uniqueImages,
    const std::string& scenepath,
    const std::string& ext,
    int width, int height,
    std::vector<std::string>& img_list) 
{
    // Initialize FFmpeg resources
    AVFormatContext* formatContext = nullptr;
    AVCodecContext* codecContext = nullptr;
    const AVCodec* codec = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* rgbFrame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* swsContext = nullptr;
    uint8_t* buffer = nullptr;
    
    try {
        // Open input file
        if (avformat_open_input(&formatContext, videoPath.c_str(), nullptr, nullptr) != 0) {
            throw std::runtime_error("Could not open video file");
        }
        
        // Retrieve stream information
        if (avformat_find_stream_info(formatContext, nullptr) < 0) {
            throw std::runtime_error("Could not find stream information");
        }
        
        // Find the video stream
        int videoStreamIndex = -1;
        for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
            if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStreamIndex = i;
                break;
            }
        }
        
        if (videoStreamIndex == -1) {
            throw std::runtime_error("Could not find video stream");
        }
        
        // Get video rotation
        int rotation = getVideoRotation(formatContext->streams[videoStreamIndex]);
        LOGI("Video rotation angle: %{public}d degrees", rotation);
        
        // Get codec parameters
        AVCodecParameters* codecParameters = formatContext->streams[videoStreamIndex]->codecpar;
        
        // Find decoder
        codec = avcodec_find_decoder(codecParameters->codec_id);
        if (!codec) {
            throw std::runtime_error("Unsupported codec");
        }
        
        // Create codec context
        codecContext = avcodec_alloc_context3(codec);
        if (!codecContext) {
            throw std::runtime_error("Failed to allocate codec context");
        }
        
        // Copy codec parameters to codec context
        if (avcodec_parameters_to_context(codecContext, codecParameters) < 0) {
            throw std::runtime_error("Failed to copy codec parameters to codec context");
        }
        
        // Open codec
        if (avcodec_open2(codecContext, codec, nullptr) < 0) {
            throw std::runtime_error("Failed to open codec");
        }
        
        // Allocate frames
        frame = av_frame_alloc();
        rgbFrame = av_frame_alloc();
        if (!frame || !rgbFrame) {
            throw std::runtime_error("Failed to allocate frames");
        }

        LOGI("Original height: %{public}d, Original width %{public}d", codecContext->height, codecContext->width);

        // Allocate RGB frame buffer with scaling dimensions
        int numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, codecContext->width, codecContext->height, 1);
        buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));
        if (!buffer) {
            throw std::runtime_error("Failed to allocate buffer");
        }
        
        // Setup RGB frame with the buffer
        if (av_image_fill_arrays(rgbFrame->data, rgbFrame->linesize, buffer,
                                AV_PIX_FMT_RGB24, codecContext->width, codecContext->height, 1) < 0) {
            throw std::runtime_error("Failed to setup RGB frame");
        }

        // Initialize scaling context with rotation handling
        int swsFlags = SWS_BILINEAR;
        swsContext = nullptr;
        
        // Apply correct rotation during scaling
        swsContext = sws_getContext(
            codecContext->width, codecContext->height, codecContext->pix_fmt,
            codecContext->width, codecContext->height, AV_PIX_FMT_BGR24,
            swsFlags | SWS_FULL_CHR_H_INT, nullptr, nullptr, nullptr);
        
        if (!swsContext) {
            throw std::runtime_error("Failed to initialize scaling context");
        }
        
        // Allocate packet
        packet = av_packet_alloc();
        if (!packet) {
            throw std::runtime_error("Failed to allocate packet");
        }
        
        // Create output directory
        std::filesystem::create_directories(scenepath + "/images/");
        
        // Read frames from the video
        int frameCount = 0;
        while (av_read_frame(formatContext, packet) >= 0) {
            if (packet->stream_index == videoStreamIndex) {
                // Send packet to decoder
                int response = avcodec_send_packet(codecContext, packet);
                if (response < 0) {
                    throw std::runtime_error("Error sending packet to decoder");
                }
                
                while (response >= 0) {
                    response = avcodec_receive_frame(codecContext, frame);
                    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                        break;
                    } else if (response < 0) {
                        throw std::runtime_error("Error during decoding");
                    }
                    
                    // If we have a valid frame and it matches our timestamps
                    if (frameCount < timestamps.size()) {
                        std::string timestamp = timestamps[frameCount];
                        if (uniqueImages.find(timestamp + ext) != uniqueImages.end()) {

                            sws_scale(swsContext, frame->data, frame->linesize, 0, codecContext->height, rgbFrame->data, rgbFrame->linesize);
                            cv::Mat img(codecContext->height, codecContext->width, CV_8UC3, rgbFrame->data[0]);
                            cv::Mat rotated;
                            
                            if (rotation == 270) 
                                cv::rotate(img, rotated, cv::ROTATE_90_CLOCKWISE);
                            else if (rotation == 90) 
                                cv::rotate(img, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
                            else if (rotation == 180) 
                                cv::rotate(img, rotated, cv::ROTATE_180);
                            else
                                rotated = img;

                            cv::resize(rotated, rotated, cv::Size(width, height));
                            std::string imagePath = scenepath + "/images/" + timestamp + ext;
                            img_list.push_back(imagePath);
                            cv::imwrite(imagePath, rotated);
                        }
                        frameCount++;
                    }
                }
            }
            av_packet_unref(packet);
        }
        std::cout << frameCount << std::endl;
    } catch (const std::exception& e) {
        LOGE("Error processing video: %s", e.what());
        
        // Cleanup FFmpeg resources
        if (buffer) av_free(buffer);
        if (swsContext) sws_freeContext(swsContext);
        if (frame) av_frame_free(&frame);
        if (rgbFrame) av_frame_free(&rgbFrame);
        if (packet) av_packet_free(&packet);
        if (codecContext) avcodec_free_context(&codecContext);
        if (formatContext) avformat_close_input(&formatContext);
        
        return false;
    }
    
    // Cleanup FFmpeg resources

    if (buffer) av_free(buffer);
    if (swsContext) sws_freeContext(swsContext);
    if (frame) av_frame_free(&frame);
    if (rgbFrame) av_frame_free(&rgbFrame);
    if (packet) av_packet_free(&packet);
    if (codecContext) avcodec_free_context(&codecContext);
    if (formatContext) avformat_close_input(&formatContext);
    
    return true;
}

void DataLoader::loadInput12(const std::string _scenepath, DataLoader::InputInitStruct& data, bool& resumed)
{
    auto basePath = FileOperator::FileOperator::GetInstance()->GetHapFilesDir();
    data.basepath = basePath;
    
    std::string scenepath = basePath + _scenepath;
    std::string slampath = scenepath + "/slam";
    std::string mediapath = scenepath + "/media";

    std::string imagesFilePath = slampath + "/images.txt";
    std::string cameraFilePath = mediapath + "/cameras.txt";             // TODO: slampath?
    std::string videoInfoFilePath = mediapath + "/videoInfo.txt";
    
    // 2. Load intrinsics from cameras.txt
    int height, width;
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
            if (!(iss >> cameraId >> camera_type >> height >> width >> fx >> fy >> cy >> cx)) {
                continue;
            }
        }
        else if (cameraFilePath == slampath + "/cameras.txt") {
            if (!(iss >> cameraId >> camera_type >> width >> height >> fx >> fy >> cx >> cy)) {
                continue;
            }

            // width = 480, height = 640
            float ratiox = 480.0 / width;
            float ratioy = 640.0 / height;
            fx *= ratiox;
            fy *= ratioy;
            cx *= ratiox;
            cy *= ratioy;
            width = 480;
            height = 640;
        }

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
        
        float norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        DataLoader::InputInitStruct::DataExtr extr = { qw / norm, qx / norm, qy / norm, qz / norm, tx, ty, tz };
        data.extr_tmp.push_back(extr);
        
        if(std::filesystem::exists(scenepath + "/images/" + imageName)){
            data.img_list.push_back(scenepath + "/images/" + imageName);
        }
        if (ext.size() == 0){
            size_t dotPos = imageName.rfind('.');
            if (dotPos != std::string::npos) {
                ext = imageName.substr(dotPos);
            }
        }
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
    
    std::vector<std::string> timestamps;
    std::ifstream videoInfoFile(videoInfoFilePath);
    if (!videoInfoFile.is_open()){
        LOGE("Cannot open videoInfo.txt!");
    }
    while (std::getline(videoInfoFile, line)) {
        if (line.empty() || line[0] == '#')
            continue;
    
        if (line.substr(0, 8) == "Rotation" || line.substr(0, 8) == "rotation")
            continue;
    
        std::istringstream iss(line);
        int frameId, zero;
        std::string timestamp;
        if (!(iss >> frameId >> timestamp >> zero)) {
            continue; // Skip invalid lines.
        }
        timestamps.push_back(timestamp);
    }
    
    if (data.img_list.size() != data.extr_tmp.size()){
        std::vector<std::string>().swap(data.img_list);
        
        // Use FFmpeg instead of OpenCV to process the video
        if (!processVideoWithFFmpeg(allVideoFile[0], timestamps, uniqueImages, 
                                 scenepath, ext, width, height, data.img_list)) {
            LOGE("Failed to process video with FFmpeg!");
        }
    }
    
    std::string resume_gaussian_path = basePath + _scenepath + "/point_cloud.ply";
    loadInputGaussianData(resume_gaussian_path, data);
    if (data.positions.size() == 0){
        LOGW("Resuming from ply failed, starting from points3D.txt");
        std::vector<float> head_bbox;
        bool is_human = HumanPreprocess(scenepath, head_bbox);
        
        std::string pointsFilePath = scenepath + "/slam/points3D.txt";
        load3DPoints(pointsFilePath, data);
        
        if(is_human){
            std::string smplPath = scenepath + "/slam/smpl.txt";
            load3DPoints(smplPath, data);
            for(int i=0; i<head_bbox.size(); ++i){
                LOGI("head bbox: %{public}f", head_bbox[i]);
            }
            data.head_bbox = head_bbox;
        }
    }
    else{
        resumed = true;
    }
    
    videoInfoFile.close();
    
    imagesFile.close();
    camerasFile.close();
}


DataLoader::InputInitStruct DataLoader::loadInputTo(const std::string _scenepath)
{
    auto basePath = FileOperator::FileOperator::GetInstance()->GetHapFilesDir();
    DataLoader::InputInitStruct data;
    data.basepath = basePath;
    std::string scenepath = basePath + "/" + _scenepath;
    
    const std::string pointsDenseFilePath = scenepath + "/downsample2.ply";
    const std::string pointsFilePath = scenepath + "/slam/points3D.txt";
    const std::string imagesFilePath = scenepath + "/slam/images.txt";
    const std::string cameraFilePath = scenepath + "/slam/cameras.txt";
    
    // 1. Load points and color data from points3D.txt
    std::ifstream pointsFile(pointsDenseFilePath, std::ios::binary);
    std::string line;
    std::unordered_set<glm::vec3, Vec3Hash, Vec3Equal> uniquePoints;
    if(!pointsFile.is_open()){
        pointsFile.close();
        pointsFile.open(pointsFilePath);
        if (!pointsFile.is_open()) {
            throw std::runtime_error("Failed to open points3D.txt file.");
        }
        float C0 = 0.28209479177387814;
        while (std::getline(pointsFile, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream iss(line);
            int pointId;
            float x, y, z;
            int r, g, b;
            float alpha; // Although alpha is provided, we will not use it.
            if (!(iss >> pointId >> x >> y >> z >> r >> g >> b >> alpha)) {
                continue; // Skip invalid lines.
            }
    
            glm::vec3 pos(x, y, z);
            // Check for duplicate points
            if (uniquePoints.find(pos) != uniquePoints.end()) {
                continue;
            }
            uniquePoints.insert(pos);
    
            // Store the point position.
            data.positions.push_back(pos);
    
            // Default attributes (scales, rotations, opacities) to zeros.
            data.scales.push_back(glm::vec3(-4.5f, -4.5f, -4.5f));
            data.rotations.push_back(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
            data.opacities.push_back(-2.197224617004395);
    
            // Process the color data.
            // We interpret r, g, b (from points3D.txt) as f_dc_0, f_dc_1, and f_dc_2.
            DataLoader::InputInitStruct::Data colorData = {};
            // Save r in index 0, g in index 16, and b in index 32.
            colorData.values[0] = static_cast<float>((r / 255.0 - 0.5) / C0);
            colorData.values[16] = static_cast<float>((g / 255.0 - 0.5) / C0);
            colorData.values[32] = static_cast<float>((b / 255.0 - 0.5) / C0);
            data.colors.push_back(colorData);
        }
    }
    else {
        PlyFile plyFile;
        plyFile.parse_header(pointsFile);
        std::shared_ptr<PlyData> x, y, z;
        std::shared_ptr<PlyData> r, g, b;
        
        try {
            x = plyFile.request_properties_from_element("vertex", { "x" });
            y = plyFile.request_properties_from_element("vertex", { "y" });
            z = plyFile.request_properties_from_element("vertex", { "z" });
            r = plyFile.request_properties_from_element("vertex", { "red" });
            g = plyFile.request_properties_from_element("vertex", { "green" });
            b = plyFile.request_properties_from_element("vertex", { "blue" });
        }
        catch (const std::exception& e) {
            std::cerr << "Error reading properties: " << e.what() << std::endl;
        }

        size_t numVertices = x->count;
        plyFile.read(pointsFile);
        
        float C0 = 0.28209479177387814;
        for (size_t i = 0; i < numVertices; ++i) {
            glm::vec3 pos(
                reinterpret_cast<float*>(x->buffer.get())[i],
                reinterpret_cast<float*>(y->buffer.get())[i],
                reinterpret_cast<float*>(z->buffer.get())[i]
            );
            
            if (uniquePoints.find(pos) != uniquePoints.end()) {
                continue;
            }
            
            uniquePoints.insert(pos);
            
            data.positions.push_back(pos);
    
            data.scales.push_back(glm::vec3(-4.5f, -4.5f, -4.5f));
            data.rotations.push_back(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
            data.opacities.push_back(-2.197224617004395);
            
            DataLoader::InputInitStruct::Data colorData = {};
    
            float _r = (reinterpret_cast<uint8_t*>(r->buffer.get())[i] / 255.0 - 0.5) / C0;
            float _g = (reinterpret_cast<uint8_t*>(g->buffer.get())[i] / 255.0 - 0.5) / C0;
            float _b = (reinterpret_cast<uint8_t*>(b->buffer.get())[i] / 255.0 - 0.5) / C0;
    
            colorData.values[0] = _r;
            colorData.values[16] = _g;
            colorData.values[32] = _b;
            data.colors.push_back(colorData);
        }

    }
    
    // 2. Load image file names and extrinsics from images.txt
    std::ifstream imagesFile(imagesFilePath);
    if (!imagesFile.is_open()) {
        throw std::runtime_error("Failed to open images.txt file.");
    }
    std::unordered_set<std::string> uniqueImages;
    while (std::getline(imagesFile, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        int frameId;
        float qw, qx, qy, qz;
        float tx, ty, tz;
        int camera_id;
        std::string imageName;

        // Read the first 10 tokens:
        // frame_id, qx, qy, qz, qw, tx, ty, tz, camera_id, image_name
        if (!(iss >> frameId >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> imageName)) {
            continue; // Skip invalid lines.
        }

        //if (frameId % 3 != 0) continue;

        if (uniqueImages.find(imageName) != uniqueImages.end()) {
            continue;
        }
        uniqueImages.insert(imageName);

        float norm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        DataLoader::InputInitStruct::DataExtr extr = { qw / norm,qx / norm,qy / norm,qz / norm, tx, ty, tz };
        data.extr_tmp.push_back(extr);
        
        // Store the image name.
        data.img_list.push_back(scenepath + "/images/" + imageName);
    }
    data.extr_tmp_test = data.extr_tmp;

    // 3. Load intrinsics from cameras.txt
    std::ifstream camerasFile(cameraFilePath);
    if (!camerasFile.is_open()) {
        throw std::runtime_error("Failed to open cameras.txt file.");
    }
    while (std::getline(camerasFile, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        int cameraId;
        std::string camera_type;
        float fx, fy, cx, cy;
        int height, width;
        int camera_id;
        std::string imageName;

        // Read the first 6 tokens:
        if (!(iss >> cameraId >> camera_type >> width >> height >> fx >> fy >> cx >> cy)) {
            continue; // Skip invalid lines.
        }

        assert(camera_type == "PINHOLE" || camera_type == "pinhole");

        std::vector<float> intr = { fx,fy,cx,cy };
        data.intr_tmp.push_back(intr);
    }

    assert(data.intr_tmp.size() == 1 || data.intr_tmp.size() == data.extr_tmp.size());
    
    pointsFile.close();
    imagesFile.close();
    camerasFile.close();
    
    return data;
}

