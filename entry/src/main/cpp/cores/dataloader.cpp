#include "dataloader.h"
#include "base/math/matrix.h"
#include "base/math/matrix_util.h"
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <dirent.h>
#include "visual_hull.h"
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


double quatDist(DataLoader::InputInitStruct::DataExtr& A, DataLoader::InputInitStruct::DataExtr& B){
    auto dot = A.values[0] * B.values[0] + A.values[1] * B.values[1] + A.values[2] * B.values[2] + A.values[3] * B.values[3];
    dot = std::max(-1.0f, std::min(1.0f, dot));
    double angle = std::acos(std::abs(dot));
    return angle;
}

template<typename T> 
void swapElem(std::vector<T>& elems, std::vector<int> arr) {
    
    assert(elems.size() == arr.size());
    
    std::vector<T> elems_swap;
    for (int i = 0; i < elems.size(); i++){
        elems_swap.push_back(elems[arr[i]]);
    }
    elems_swap.swap(elems);
}


void DataLoader::InputInitStruct::shuffleImage() {
    int N = this->img_list.size();
    
    std::vector<int> arr(N, 0);
    for(int i = 0; i < N; i++)
        arr[i] = i;
    
    std::mt19937 gen;
    std::random_device rd;
    gen.seed(rd());
    std::shuffle(arr.begin(), arr.end(), gen);
    
//     swapElem(mask_list, arr);
    swapElem(img_list, arr);
    swapElem(extr_tmp, arr);
    swapElem(extr_tmp_test, arr);
    
    if (intr_tmp.size() > 1)
        swapElem(intr_tmp, arr);
        
}

int DataLoader::InputInitStruct::findMaxPose(std::set<int>& chosen) {
    if (chosen.size() == img_list.size())
        return -1;
        
    double maxAngle = -100000;
    int maxID = 100000;
    
    
    for (int i = 0; i < img_list.size(); i++) {
        
        if (chosen.find(i) != chosen.end())
            continue;
            
        if (chosen.size() == img_list.size() - 1) {
            maxID = i;
            break;
        }
            
        double minAngle = 100000;
        
        for (int j = 0; j < trainOrder.size(); j++){
            double angle = quatDist(extr_tmp[i], extr_tmp[trainOrder[j]]);
            if(angle < minAngle) {
                minAngle = angle;
            }
        }
        
        if (minAngle > maxAngle) {
            maxAngle = minAngle;
            maxID = i;
        }
    }
    
    return maxID;
}

void DataLoader::InputInitStruct::getTrainOrder(int first_id) 
{
    int N = this->img_list.size();
    if (N <= 0)
        return;
        
    std::set<int> chosen;
    trainOrder.clear();
    if (first_id >= N)
        first_id = 0;
    trainOrder.push_back(first_id);
    chosen.insert(first_id);
    
    for(int i = 1; i < N; i++) {
        auto id = findMaxPose(chosen);
        trainOrder.push_back(id);
        chosen.insert(id);
    }
}

void DataLoader::InputInitStruct::getTrainOrders()
{
    int N = this->img_list.size();
    if (N <= 0)
        return;
    
    for (int i = 0; i < N; i++) {
        trainOrder.clear();
        getTrainOrder(i);
        trainOrders.push_back(trainOrder);
    }
}

void projectPoints(
    const Eigen::Matrix<float, Eigen::Dynamic, 3>& points_world,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& T,
    float fx, float fy, float cx, float cy,
    Eigen::Matrix<float, Eigen::Dynamic, 3>& projected_points
)
{
    Eigen::Matrix<float, Eigen::Dynamic, 3> points_cam = (R * points_world.transpose()).transpose();
    points_cam.rowwise() += T.transpose();
    
    // Project to image plane (vectorized)
    projected_points.resize(points_cam.rows(), 3);
    Eigen::VectorXf z_inv = points_cam.col(2).cwiseInverse();
    projected_points.col(0) = (points_cam.col(0).array() * z_inv.array()) * fx + cx;
    projected_points.col(1) = (points_cam.col(1).array() * z_inv.array()) * fy + cy;
    projected_points.col(2) = points_cam.col(2);
}

void DataLoader::InputInitStruct::compute_3D_interval()
{
    intervals = std::vector<float>(positions.size(), 100000);
    
    Eigen::Matrix<float, Eigen::Dynamic, 3> all_positions(positions.size(), 3);
    for (size_t i = 0; i < positions.size(); ++i) {
        all_positions(i, 0) = positions[i].x;
        all_positions(i, 1) = positions[i].y;
        all_positions(i, 2) = positions[i].z;
    }
    float fx = intr_tmp[0][0];
    float fy = intr_tmp[0][1];
    float cx = intr_tmp[0][2];
    float cy = intr_tmp[0][3];
    
    float max_dist = -1;
    float min_dist = 10000;
    
    for (auto& extr : extr_tmp) {
        Eigen::Quaternionf q(extr.values[0], extr.values[1], extr.values[2], extr.values[3]);
        q.normalize();
        auto R = q.toRotationMatrix();
        Eigen::Vector3f T(extr.values[4], extr.values[5], extr.values[6]);
        
        Eigen::Matrix<float, Eigen::Dynamic, 3> projected_points;
        projectPoints(all_positions, R, T, fx, fy, cx, cy, projected_points);
        
        for(int i = 0; i < positions.size(); i++) {
            
            float u = projected_points(i, 0);
            float v = projected_points(i, 1);
            float d = projected_points(i, 2);
            
            if (u < 0 || u >= width || v < 0 || v >= height || d < 0.2)
                continue;
            
            intervals[i] = std::min(intervals[i], d);
            max_dist = std::max(max_dist, d);
            min_dist = std::min(min_dist, d);
        }
    }
    
    for (int i = 0; i < intervals.size(); i++) {
        if (intervals[i] > 1000){
            intervals[i] = max_dist / fx;
        }
        else{
            intervals[i] /= fx;
        }
    }
    max_dist = log(max_dist / fx);
    min_dist = log(min_dist / fx);
    
    // normalize
    for(int i = 0; i < intervals.size(); i++) {
        intervals[i] = (intervals[i] - min_dist) / (max_dist - min_dist);
    }
}


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

void DataLoader::saveToPLY_tinyply(const std::string& fullDataPath, const DataLoader::InputInitStruct& data) {
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


void DataLoader::saveCameras(const std::string fullDataPath, const DataLoader::InputInitStruct& data){
    std::string extrName = fullDataPath + "/images2.txt";
    std::string intrName = fullDataPath + "/cameras2.txt";

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
        intrFile << i << " PINHOLE " << data.width << " " << data.height << " ";
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
        extrFile << i << " " << filename << "\n\n";
    }
    extrFile.close();
}


void load3DPoints(std::string& pointsFilePath, DataLoader::InputInitStruct& data, bool filter=false){
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
//         if(filter && data.body_bbox.size() > 0){
//             if(x < data.body_bbox[0] || x > data.body_bbox[1] || 
//                 y < data.body_bbox[2] || y > data.body_bbox[3] || 
//                 z < data.body_bbox[4] || z > data.body_bbox[5]){
//                 continue;
//             }
//         }
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
    std::vector<Base::Math::Vec3>       skyNormals;

    skyPositions.reserve(sky_num_gs);
    skyScales.reserve(sky_num_gs);
    skyRotations.reserve(sky_num_gs);
    skyOpacities.reserve(sky_num_gs);
    skyColors.reserve(sky_num_gs);
    skyNormals.reserve(sky_num_gs);
    

    for (int i = 0; i < sky_num_gs; ++i) {
        Base::Math::Vec3 pos = dirs[i] * halfSize + center;
        skyPositions.push_back(pos);
        
        skyScales.push_back(Base::Math::Vec3(-4.5f, -4.5f, -4.5f));
        skyRotations.push_back(Base::Math::Vec4(1.0f, 0.0f, 0.0f, 0.0f));
        skyOpacities.push_back(-2.197224617004395f);
        skyNormals.push_back(Base::Math::Vec3(1.0f, 1.0f, 1.0f));
        
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
    if(data.normals.size() > 0) {
        data.normals.insert(data.normals.begin(), skyNormals.begin(), skyNormals.end());
    }
}

void DataLoader::InputInitStruct::processAndCropForeground(const std::string& scenepath) {
    // Create foreground directory
    const std::vector<string> target = this->img_list;
    std::filesystem::create_directories(scenepath + "/foreground");
    
    // Ensure intr_tmp has enough entries for all images if needed
    if (this->intr_tmp.size() == 1 && target.size() > 1) {
        std::vector<float> first_intrinsic = this->intr_tmp[0];
        this->intr_tmp.resize(target.size(), first_intrinsic);
    }
    
    // Process each image
    for (size_t i = 0; i < target.size(); ++i) {
        std::string imagePath = target[i];
        
        // Extract filename from path
        std::string filename = std::filesystem::path(imagePath).filename().string();
        std::string nameWithoutExt = std::filesystem::path(filename).stem().string();
        
        // Construct mask path (replace "images" with "segmented" and .jpg with .png)
        std::string maskPath = imagePath;
        size_t pos = maskPath.find("/images/");
        if (pos != std::string::npos) {
            maskPath.replace(pos, 8, "/segmented/");
        }
        // Replace extension from .jpg to .png
        size_t lastDot = maskPath.find_last_of('.');
        if (lastDot != std::string::npos) {
            maskPath = maskPath.substr(0, lastDot) + ".png";
        }
        
        // Read mask
        cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            LOGE("Failed to read mask: %s", maskPath.c_str());
            continue;
        }
        
        // Find bounding box of foreground (white pixels)
        std::vector<cv::Point> whitePixels;
        cv::findNonZero(mask, whitePixels);
        
        if (whitePixels.empty()) {
            LOGE("No foreground pixels found in mask: %s", maskPath.c_str());
            continue;
        }
        
        cv::Rect boundingBox = cv::boundingRect(whitePixels);
        
        // Find center of bounding box
        cv::Point center(boundingBox.x + boundingBox.width / 2, 
                        boundingBox.y + boundingBox.height / 2);
        
        LOGI("Image %s: Bounding box center at (%d, %d)", 
             filename.c_str(), center.x, center.y);
        
        // Read original image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            LOGE("Failed to read image: %s", imagePath.c_str());
            continue;
        }
        
        // Crop image from center with 640 width and 480 height
        cv::Point actualTopLeft;
        cv::Mat croppedImage = cropImage(image, center, cv::Size(480, 640), actualTopLeft);
        
        // Save cropped image to foreground folder (keeping original filename)
        std::string outputPath = scenepath + "/foreground/obj_" + filename;
        cv::imwrite(outputPath, croppedImage);
        
        LOGI("Saved cropped image to: %{public}s", outputPath.c_str());
        
        // Add cropped image path to data
        this->img_list.push_back(outputPath);
        
        // Calculate new intrinsics for the cropped image
        std::vector<float> intr = this->intr_tmp[i];
        float scale_factor = image.cols / 480.0; // No scaling, just cropping at original resolution
        std::vector<float> new_intr = calculateNewIntrinsics(
            intr[0], intr[1], intr[2], intr[3],
            actualTopLeft, cv::Size(480, 640), scale_factor
        );
        
        LOGI("New intrinsics - fx:%{public}f fy:%{public}f cx:%{public}f cy:%{public}f", 
             new_intr[0], new_intr[1], new_intr[2], new_intr[3]);
        
        // Add new intrinsics to data
        this->intr_tmp.push_back(new_intr);
        
        // If there are extrinsics, copy them as well
        if (i < this->extr_tmp.size()) {
            this->extr_tmp.push_back(this->extr_tmp[i]);
        }
        
        // If there are timestamps, copy them as well
        if (i < this->time_stamps.size()) {
            this->time_stamps.push_back(this->time_stamps[i]);
        }
    }
    
    // Update test extrinsics if needed
    if (!this->extr_tmp.empty()) {
        this->extr_tmp_test = this->extr_tmp;
    }
    
    LOGI("Processed %zu images for foreground cropping", target.size());
}

void DataLoader::loadInput12(const std::string _scenepath, DataLoader::InputInitStruct& data, bool& resumed, int trainingMode,
                            int height, int width)
{
    std::string scenepath = _scenepath;
    data.basepath = _scenepath;
    std::string slampath = scenepath;
    std::string mediapath = scenepath + "/media";

    std::string imagesFilePath = slampath + "/images.txt";
    std::string cameraFilePath = slampath + "/cameras.txt";             // TODO: slampath?
    std::string normalFilePath = slampath + "/normals.txt";
    
    // 2. Load intrinsics from cameras.txt
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
        
        std::string maskName = imageName.substr(0, imageName.find_last_of(".")) + ".png";
        std::string maskPath = scenepath + "/segmented/" + maskName;
        
        // Check if mask file exists before adding to list
        if (std::filesystem::exists(maskPath)) {
            data.mask_list.push_back(maskPath);
//             data.mask_index.push_back(idx);  // Record the original image index
        }
        
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
    
    data.sequence_length = data.img_list.size();
    
    data.extr_tmp_test = data.extr_tmp;
    assert(data.intr_tmp.size() == 1 || data.intr_tmp.size() == data.extr_tmp.size());
    
    std::string resume_gaussian_path = scenepath + "/point_cloud.ply";
    loadInputGaussianData(resume_gaussian_path, data);
    if (data.positions.size() == 0){
        LOGW("Resuming from ply failed, starting from points3D.txt");
        bool is_human = HumanPreprocess(scenepath, data, trainingMode==HUMAN);

        std::string pointsFilePath = slampath + "/points3D.txt";
        load3DPoints(pointsFilePath, data, (is_human && trainingMode==HUMAN));
        
        
        if (is_human) {
            int personIndex = 1;
            while (true) {
                std::string smplPath = scenepath + "/" + "smpl" + std::to_string(personIndex) + ".txt";
                std::ifstream file(smplPath);
                if (!file.good()) {
                    break;
                }
                load3DPoints(smplPath, data);
                ++personIndex;
            }
        }
    }
    else {
        resumed = true;
    }
    
    
    std::ifstream normalsFile(normalFilePath);
    if (normalsFile.is_open()) {
        std::string line;
        while (std::getline(normalsFile, line)){
            if (line.empty()) continue;
            std::istringstream iss(line);
            float nx, ny, nz;
            if (!(iss >> nx >> ny >> nz)) {
                continue; // Skip invalid lines.
            }
            Base::Math::Vec3 normals(nx, ny, nz);
            data.normals.push_back(normals);
        }
    }
    
    prependSkyboxPoints(data, 10000, 0.01, 10.0);

//     if(isObject){
//         data = performVisualHullSegmentation(data);
//     }
//     if(trainingMode == OBJECT){
//         data.processAndCropForeground(scenepath);
//     }

//     saveCameras(scenepath, data);
    imagesFile.close();
    camerasFile.close();
}