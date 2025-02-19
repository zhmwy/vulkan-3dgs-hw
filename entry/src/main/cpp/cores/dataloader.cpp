#include "dataloader.h"
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <dirent.h>

#include "file/file_operator.h"
#include <unordered_set>

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

DataLoader::ProjectEWAForwardStruct DataLoader::loadEWAForwardData() {
    const std::string filepath = FileOperator::GetInstance()->GetFileAbsolutePath("assets/test_ply/ProjectEWA/forward_data.ply");

    DataLoader::ProjectEWAForwardStruct data;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open PLY file.");

    PlyFile plyFile;
    plyFile.parse_header(file);

    std::shared_ptr<PlyData> x, y, z, scale_x, scale_y, scale_z, rotate_0, rotate_1, rotate_2, rotate_3,
        vis_input, u, v, d, cov2d_0, cov2d_1, cov2d_2, cov3d_0, cov3d_1, cov3d_2,
        cov3d_3, cov3d_4, cov3d_5, radius, tiles_torched, rect_0, rect_1, rect_2,
        rect_3, vis_output;

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

        vis_input = plyFile.request_properties_from_element("vertex", { "vis_input" });
        u = plyFile.request_properties_from_element("vertex", { "u" });
        v = plyFile.request_properties_from_element("vertex", { "v" });
        d = plyFile.request_properties_from_element("vertex", { "d" });
        cov2d_0 = plyFile.request_properties_from_element("vertex", { "cov2d_0" });
        cov2d_1 = plyFile.request_properties_from_element("vertex", { "cov2d_1" });
        cov2d_2 = plyFile.request_properties_from_element("vertex", { "cov2d_2" });
        cov3d_0 = plyFile.request_properties_from_element("vertex", { "cov3d_0" });
        cov3d_1 = plyFile.request_properties_from_element("vertex", { "cov3d_1" });
        cov3d_2 = plyFile.request_properties_from_element("vertex", { "cov3d_2" });
        cov3d_3 = plyFile.request_properties_from_element("vertex", { "cov3d_3" });
        cov3d_4 = plyFile.request_properties_from_element("vertex", { "cov3d_4" });
        cov3d_5 = plyFile.request_properties_from_element("vertex", { "cov3d_5" });
        radius = plyFile.request_properties_from_element("vertex", { "radius" });
        tiles_torched = plyFile.request_properties_from_element("vertex", { "tiles_torch" });
        rect_0 = plyFile.request_properties_from_element("vertex", { "rect_0" });
        rect_1 = plyFile.request_properties_from_element("vertex", { "rect_1" });
        rect_2 = plyFile.request_properties_from_element("vertex", { "rect_2" });
        rect_3 = plyFile.request_properties_from_element("vertex", { "rect_3" });
        vis_output = plyFile.request_properties_from_element("vertex", { "vis_output" });
    }
    catch (const std::exception& e) {
        std::cout << "Error reading properties: " << e.what() << std::endl;
    }

    plyFile.read(file);
    size_t numVertices = x->count;

    for (size_t i = 0; i < numVertices; ++i) {
        data.xyz.emplace_back(reinterpret_cast<float*>(x->buffer.get())[i], reinterpret_cast<float*>(y->buffer.get())[i], reinterpret_cast<float*>(z->buffer.get())[i]);
        data.scale.emplace_back(reinterpret_cast<float*>(scale_x->buffer.get())[i], reinterpret_cast<float*>(scale_y->buffer.get())[i], reinterpret_cast<float*>(scale_z->buffer.get())[i]);
        data.rotate.emplace_back(reinterpret_cast<float*>(rotate_0->buffer.get())[i], reinterpret_cast<float*>(rotate_1->buffer.get())[i], reinterpret_cast<float*>(rotate_2->buffer.get())[i], reinterpret_cast<float*>(rotate_3->buffer.get())[i]);
        data.vis_input.push_back(reinterpret_cast<float*>(vis_input->buffer.get())[i]);
        data.uvd.emplace_back(reinterpret_cast<float*>(u->buffer.get())[i], reinterpret_cast<float*>(v->buffer.get())[i], reinterpret_cast<float*>(d->buffer.get())[i]);
        data.cov2d.emplace_back(reinterpret_cast<float*>(cov2d_0->buffer.get())[i], reinterpret_cast<float*>(cov2d_1->buffer.get())[i], reinterpret_cast<float*>(cov2d_2->buffer.get())[i]);
        data.cov3d.emplace_back(reinterpret_cast<float*>(cov3d_0->buffer.get())[i], reinterpret_cast<float*>(cov3d_1->buffer.get())[i], reinterpret_cast<float*>(cov3d_2->buffer.get())[i]);
        data.cov3d.emplace_back(reinterpret_cast<float*>(cov3d_3->buffer.get())[i], reinterpret_cast<float*>(cov3d_4->buffer.get())[i], reinterpret_cast<float*>(cov3d_5->buffer.get())[i]);
        data.radius.push_back(reinterpret_cast<int*>(radius->buffer.get())[i]);
        data.tiles_torched.push_back(reinterpret_cast<int*>(tiles_torched->buffer.get())[i]);
        data.tiles_rect.emplace_back(reinterpret_cast<int*>(rect_0->buffer.get())[i], reinterpret_cast<int*>(rect_1->buffer.get())[i], reinterpret_cast<int*>(rect_2->buffer.get())[i], reinterpret_cast<int*>(rect_3->buffer.get())[i]);
        data.vis_output.push_back(reinterpret_cast<float*>(vis_output->buffer.get())[i]);
    }

    return data;
}

DataLoader::ProjectEWABackwardStruct DataLoader::loadEWABackwardData() {
    const std::string filepath = FileOperator::GetInstance()->GetFileAbsolutePath("assets/test_ply/ProjectEWA/backward_data.ply");

    DataLoader::ProjectEWABackwardStruct data;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open PLY file.");

    PlyFile plyFile;
    plyFile.parse_header(file);

    std::shared_ptr<PlyData> x, y, z, scale_x, scale_y, scale_z, rotate_0, rotate_1, rotate_2, rotate_3,
        visible, cov3d_0, cov3d_1, cov3d_2, cov3d_3, cov3d_4, cov3d_5, dL_du, dL_dv, dL_dd,
        dL_dcov2d_0, dL_dcov2d_1, dL_dcov2d_2, dL_dx, dL_dy, dL_dz, dL_dscale_x, dL_dscale_y, dL_dscale_z,
        dL_drotate_0, dL_drotate_1, dL_drotate_2, dL_drotate_3;

    try {
        // Requesting properties from elements in the PLY file
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

        visible = plyFile.request_properties_from_element("vertex", { "visible" });

        cov3d_0 = plyFile.request_properties_from_element("vertex", { "cov3d_0" });
        cov3d_1 = plyFile.request_properties_from_element("vertex", { "cov3d_1" });
        cov3d_2 = plyFile.request_properties_from_element("vertex", { "cov3d_2" });
        cov3d_3 = plyFile.request_properties_from_element("vertex", { "cov3d_3" });
        cov3d_4 = plyFile.request_properties_from_element("vertex", { "cov3d_4" });
        cov3d_5 = plyFile.request_properties_from_element("vertex", { "cov3d_5" });

        dL_du = plyFile.request_properties_from_element("vertex", { "dL_du" });
        dL_dv = plyFile.request_properties_from_element("vertex", { "dL_dv" });
        dL_dd = plyFile.request_properties_from_element("vertex", { "dL_dd" });

        dL_dcov2d_0 = plyFile.request_properties_from_element("vertex", { "dL_dcov2d_0" });
        dL_dcov2d_1 = plyFile.request_properties_from_element("vertex", { "dL_dcov2d_1" });
        dL_dcov2d_2 = plyFile.request_properties_from_element("vertex", { "dL_dcov2d_2" });

        dL_dx = plyFile.request_properties_from_element("vertex", { "dL_dx" });
        dL_dy = plyFile.request_properties_from_element("vertex", { "dL_dy" });
        dL_dz = plyFile.request_properties_from_element("vertex", { "dL_dz" });

        dL_dscale_x = plyFile.request_properties_from_element("vertex", { "dL_dscale_x" });
        dL_dscale_y = plyFile.request_properties_from_element("vertex", { "dL_dscale_y" });
        dL_dscale_z = plyFile.request_properties_from_element("vertex", { "dL_dscale_z" });

        dL_drotate_0 = plyFile.request_properties_from_element("vertex", { "dL_drotate_0" });
        dL_drotate_1 = plyFile.request_properties_from_element("vertex", { "dL_drotate_1" });
        dL_drotate_2 = plyFile.request_properties_from_element("vertex", { "dL_drotate_2" });
        dL_drotate_3 = plyFile.request_properties_from_element("vertex", { "dL_drotate_3" });
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading properties: " << e.what() << std::endl;
    }

    plyFile.read(file);
    size_t numVertices = x->count;

    // Load data into structure
    for (size_t i = 0; i < numVertices; ++i) {
        data.xyz.emplace_back(reinterpret_cast<float*>(x->buffer.get())[i], reinterpret_cast<float*>(y->buffer.get())[i], reinterpret_cast<float*>(z->buffer.get())[i]);
        data.scale.emplace_back(reinterpret_cast<float*>(scale_x->buffer.get())[i], reinterpret_cast<float*>(scale_y->buffer.get())[i], reinterpret_cast<float*>(scale_z->buffer.get())[i]);
        data.rotate.emplace_back(reinterpret_cast<float*>(rotate_0->buffer.get())[i], reinterpret_cast<float*>(rotate_1->buffer.get())[i], reinterpret_cast<float*>(rotate_2->buffer.get())[i], reinterpret_cast<float*>(rotate_3->buffer.get())[i]);
        data.visible.push_back(reinterpret_cast<int*>(visible->buffer.get())[i]);
        data.cov3d.emplace_back(reinterpret_cast<float*>(cov3d_0->buffer.get())[i], reinterpret_cast<float*>(cov3d_1->buffer.get())[i], reinterpret_cast<float*>(cov3d_2->buffer.get())[i]);
        data.cov3d.emplace_back(reinterpret_cast<float*>(cov3d_3->buffer.get())[i], reinterpret_cast<float*>(cov3d_4->buffer.get())[i], reinterpret_cast<float*>(cov3d_5->buffer.get())[i]);
        data.dL_duv.emplace_back(reinterpret_cast<float*>(dL_du->buffer.get())[i], reinterpret_cast<float*>(dL_dv->buffer.get())[i], reinterpret_cast<float*>(dL_dd->buffer.get())[i]);
        data.dL_dcov2d.emplace_back(reinterpret_cast<float*>(dL_dcov2d_0->buffer.get())[i], reinterpret_cast<float*>(dL_dcov2d_1->buffer.get())[i], reinterpret_cast<float*>(dL_dcov2d_2->buffer.get())[i]);
        data.dL_dxyz.emplace_back(reinterpret_cast<float*>(dL_dx->buffer.get())[i], reinterpret_cast<float*>(dL_dy->buffer.get())[i], reinterpret_cast<float*>(dL_dz->buffer.get())[i]);
        data.dL_dscale.emplace_back(reinterpret_cast<float*>(dL_dscale_x->buffer.get())[i], reinterpret_cast<float*>(dL_dscale_y->buffer.get())[i], reinterpret_cast<float*>(dL_dscale_z->buffer.get())[i]);
        data.dL_drotate.emplace_back(reinterpret_cast<float*>(dL_drotate_0->buffer.get())[i], reinterpret_cast<float*>(dL_drotate_1->buffer.get())[i], reinterpret_cast<float*>(dL_drotate_2->buffer.get())[i], reinterpret_cast<float*>(dL_drotate_3->buffer.get())[i]);
    }

    return data;
}

DataLoader::EvalSHForwardStruct DataLoader::loadEvalSHForwardData() {
    const std::string filepath = FileOperator::GetInstance()->GetFileAbsolutePath("assets/test_ply/EvalSH/forward_data_3.ply");

    DataLoader::EvalSHForwardStruct data;
    const int D = 16;  // 2^(n+1) where n=3
    const int num_channels = 3;  // Assuming 3 channels for sh coefficients (e.g., RGB)

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open PLY file.");

    PlyFile plyFile;
    plyFile.parse_header(file);

    std::vector<std::shared_ptr<PlyData>> sh_properties(48);  // SH properties (sh_0 to sh_47)
    std::shared_ptr<PlyData> vx, vy, vz, visible, r, g, b;

    try {
        for (int i = 0; i < 48; ++i) {
            sh_properties[i] = plyFile.request_properties_from_element("vertex", { "sh_" + std::to_string(i) });
        }

        vx = plyFile.request_properties_from_element("vertex", { "vx" });
        vy = plyFile.request_properties_from_element("vertex", { "vy" });
        vz = plyFile.request_properties_from_element("vertex", { "vz" });
        visible = plyFile.request_properties_from_element("vertex", { "visible" });
        r = plyFile.request_properties_from_element("vertex", { "r" });
        g = plyFile.request_properties_from_element("vertex", { "g" });
        b = plyFile.request_properties_from_element("vertex", { "b" });
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading properties: " << e.what() << std::endl;
    }

    plyFile.read(file);
    size_t numVertices = vx->count;

    data.sh_coefficients.resize(numVertices, std::vector<std::vector<float>>(num_channels, std::vector<float>(D)));

    for (size_t i = 0; i < numVertices; ++i) {

        for (int j = 0; j < 3; ++j) { 
            for (int k = 0; k < D; ++k) { 
                data.sh_coefficients[i][j][k] = reinterpret_cast<float*>(sh_properties[j * D + k]->buffer.get())[i];
            }
        }


        data.v.emplace_back(reinterpret_cast<float*>(vx->buffer.get())[i],
            reinterpret_cast<float*>(vy->buffer.get())[i],
            reinterpret_cast<float*>(vz->buffer.get())[i]);

        data.visible.push_back(reinterpret_cast<int*>(visible->buffer.get())[i]);

        data.color.emplace_back(reinterpret_cast<float*>(r->buffer.get())[i],
            reinterpret_cast<float*>(g->buffer.get())[i],
            reinterpret_cast<float*>(b->buffer.get())[i]);
    }

    return data;
}

DataLoader::EvalSHBackwardStruct DataLoader::loadEvalSHBackwardData() {
    const std::string filepath =
        FileOperator::GetInstance()->GetFileAbsolutePath("assets/test_ply/EvalSH/backward_data_3.ply");

    DataLoader::EvalSHBackwardStruct data;
    const int D = 16;  // 2^(n+1) where n=3
    const int num_channels = 3;  // Assuming 3 channels for sh coefficients (e.g., RGB)

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open PLY file.");

    PlyFile plyFile;
    plyFile.parse_header(file);


    std::vector<std::shared_ptr<PlyData>> sh_properties(48);      // SH coefficients (sh_0 to sh_47)
    std::vector<std::shared_ptr<PlyData>> dL_dsh_properties(48);  // Derivative SH coefficients (dL_dsh_0 to dL_dsh_47)
    std::shared_ptr<PlyData> vx, vy, vz, visible, dL_dr, dL_dg, dL_db, dL_dvx, dL_dvy, dL_dvz;

    try {
        for (int i = 0; i < 48; ++i) {
            sh_properties[i] = plyFile.request_properties_from_element("vertex", { "sh_" + std::to_string(i) });
            dL_dsh_properties[i] = plyFile.request_properties_from_element("vertex", { "dL_dsh_" + std::to_string(i) });
        }

        vx = plyFile.request_properties_from_element("vertex", { "vx" });
        vy = plyFile.request_properties_from_element("vertex", { "vy" });
        vz = plyFile.request_properties_from_element("vertex", { "vz" });
        visible = plyFile.request_properties_from_element("vertex", { "visible" });
        dL_dr = plyFile.request_properties_from_element("vertex", { "dL_dr" });
        dL_dg = plyFile.request_properties_from_element("vertex", { "dL_dg" });
        dL_db = plyFile.request_properties_from_element("vertex", { "dL_db" });
        dL_dvx = plyFile.request_properties_from_element("vertex", { "dL_dvx" });
        dL_dvy = plyFile.request_properties_from_element("vertex", { "dL_dvy" });
        dL_dvz = plyFile.request_properties_from_element("vertex", { "dL_dvz" });
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading properties: " << e.what() << std::endl;
    }

    plyFile.read(file);
    size_t numVertices = vx->count;

    data.sh_coefficients.resize(numVertices, std::vector<std::vector<float>>(num_channels, std::vector<float>(D)));
    data.dL_dsh.resize(numVertices, std::vector<std::vector<float>>(num_channels, std::vector<float>(D)));

    for (size_t i = 0; i < numVertices; ++i) {

        for (int j = 0; j < 3; ++j) { 
            for (int k = 0; k < D; ++k) {  
                data.sh_coefficients[i][j][k] = reinterpret_cast<float*>(sh_properties[j * D + k]->buffer.get())[i];
                data.dL_dsh[i][j][k] = reinterpret_cast<float*>(dL_dsh_properties[j * D + k]->buffer.get())[i];
            }
        }

        data.v.emplace_back(reinterpret_cast<float*>(vx->buffer.get())[i],
            reinterpret_cast<float*>(vy->buffer.get())[i],
            reinterpret_cast<float*>(vz->buffer.get())[i]);

        data.visible.push_back(reinterpret_cast<int*>(visible->buffer.get())[i]);

        data.dL_dcolor.emplace_back(reinterpret_cast<float*>(dL_dr->buffer.get())[i],
            reinterpret_cast<float*>(dL_dg->buffer.get())[i],
            reinterpret_cast<float*>(dL_db->buffer.get())[i]);

        data.dL_dv.emplace_back(reinterpret_cast<float*>(dL_dvx->buffer.get())[i],
            reinterpret_cast<float*>(dL_dvy->buffer.get())[i],
            reinterpret_cast<float*>(dL_dvz->buffer.get())[i]);
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


DataLoader::InputGaussianStruct DataLoader::loadInputGaussianData() {
    auto filepath = FileOperator::GetInstance()->GetFileAbsolutePath("assets/test_ply/inputGS/iter_noact_1.ply");

    DataLoader::InputGaussianStruct data;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open PLY file.");

    PlyFile plyFile;
    plyFile.parse_header(file);

    // Declare shared pointers to request each property from the PLY file
    std::shared_ptr<PlyData> x, y, z, scale_x, scale_y, scale_z;
    std::shared_ptr<PlyData> rotate_0, rotate_1, rotate_2, rotate_3;
    std::shared_ptr<PlyData> opacity, r, g, b;

    try {
        // Request each property from the "vertex" element
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
        opacity = plyFile.request_properties_from_element("vertex", { "opacity" });
        r = plyFile.request_properties_from_element("vertex", { "r" });
        g = plyFile.request_properties_from_element("vertex", { "g" });
        b = plyFile.request_properties_from_element("vertex", { "b" });
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading properties: " << e.what() << std::endl;
    }

    // Read the data from the file
    plyFile.read(file);
    size_t numVertices = x->count;

    // Resize the vectors to fit the number of vertices
    data.positions.resize(numVertices);
    data.scales.resize(numVertices);
    data.rotations.resize(numVertices);
    data.opacities.resize(numVertices);
    data.colors.resize(numVertices);

    // Load the data into the structure
    for (size_t i = 0; i < numVertices; ++i) {
        // Load position (x, y, z)
        data.positions[i] = glm::vec3(
            reinterpret_cast<float*>(x->buffer.get())[i],
            reinterpret_cast<float*>(y->buffer.get())[i],
            reinterpret_cast<float*>(z->buffer.get())[i]
        );

        // Load scale (scale_x, scale_y, scale_z)
        data.scales[i] = glm::vec3(
            reinterpret_cast<float*>(scale_x->buffer.get())[i],
            reinterpret_cast<float*>(scale_y->buffer.get())[i],
            reinterpret_cast<float*>(scale_z->buffer.get())[i]
        );

        // Load rotation (as quaternion)
        data.rotations[i] = glm::vec4(
            reinterpret_cast<float*>(rotate_0->buffer.get())[i],
            reinterpret_cast<float*>(rotate_1->buffer.get())[i],
            reinterpret_cast<float*>(rotate_2->buffer.get())[i],
            reinterpret_cast<float*>(rotate_3->buffer.get())[i]
        );

        // Load opacity
        data.opacities[i] = reinterpret_cast<float*>(opacity->buffer.get())[i];

        // Load color (r, g, b)
        data.colors[i] = glm::vec3(
            reinterpret_cast<float*>(r->buffer.get())[i],
            reinterpret_cast<float*>(g->buffer.get())[i],
            reinterpret_cast<float*>(b->buffer.get())[i]
        );
    }

    return data;
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

// 自定义 glm::vec3 的哈希函数
struct Vec3Hash {
    size_t operator()(const glm::vec3& v) const {
        std::hash<float> floatHasher;
        size_t h1 = floatHasher(v.x);
        size_t h2 = floatHasher(v.y);
        size_t h3 = floatHasher(v.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);  // 组合哈希值
    }
};

// 自定义 glm::vec3 的相等比较
struct Vec3Equal {
    bool operator()(const glm::vec3& v1, const glm::vec3& v2) const {
        return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
    }
};


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


DataLoader::InputLegoInitStruct DataLoader::loadInputLegoInitData() {
    auto basePath = FileOperator::GetInstance()->GetHapFilesDir();
    const std::string filepath = basePath + "/" + "assets/AREngine/2024_09_27_11_20_29/input_AREngine.ply";

    DataLoader::InputLegoInitStruct data;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open PLY file.");

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

        /*for (int i = 0; i < 45; ++i) {
            std::string propName = "f_rest_" + std::to_string(i);
            colorProps.push_back(plyFile.request_properties_from_element("vertex", { propName }));
        }*/
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading properties: " << e.what() << std::endl;
    }

    plyFile.read(file);
    size_t numVertices = x->count;

    // 记录唯一的点
    std::unordered_set<glm::vec3, Vec3Hash, Vec3Equal> uniquePoints;

    // 预分配空间
    data.positions.reserve(numVertices);
    data.scales.reserve(numVertices);
    data.rotations.reserve(numVertices);
    data.opacities.reserve(numVertices);
    data.colors.reserve(numVertices);

    for (size_t i = 0; i < numVertices; ++i) {
        glm::vec3 pos(
            reinterpret_cast<float*>(x->buffer.get())[i],
            reinterpret_cast<float*>(y->buffer.get())[i],
            reinterpret_cast<float*>(z->buffer.get())[i]
        );

        // 检查点是否已经存在
        if (uniquePoints.find(pos) != uniquePoints.end()) {
            continue;  // 如果已经存在，则跳过
        }

        // 添加到哈希表中
        uniquePoints.insert(pos);

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
        DataLoader::InputLegoInitStruct::Data colorData = {};
        for (size_t j = 0; j < colorProps.size(); ++j) {
            float value = reinterpret_cast<float*>(colorProps[j]->buffer.get())[i];
            colorData.values[j / 3 + (j % 3) * 16] = value;
        }
        data.colors.push_back(colorData);
    }

    std::cout << "Unique points: " << data.positions.size() << " / " << numVertices << std::endl;

    // Get the list of PNG and JPG files in the directory
    std::string path = basePath + "/assets/AREngine/2024_09_27_11_20_29/images";
    std::vector<std::string> imgListPng;
    std::vector<std::string> imgListJpg;

    // Get PNG and JPG files
    getFilesInDirectory(path, ".png", imgListPng);
    getFilesInDirectory(path, ".jpg", imgListJpg);

    // Combine the two lists
    imgListPng.insert(imgListPng.end(), imgListJpg.begin(), imgListJpg.end());

    // Sort the combined image list
    std::sort(
        imgListPng.begin(),
        imgListPng.end(),
        [](const std::string& a, const std::string& b) {
            return extractNumber(a) < extractNumber(b);
        }
    );
    std::vector<std::string> filteredList;
    for (size_t i = 0; i < imgListPng.size(); i += 3) {
        filteredList.push_back(imgListPng[i]);
    }

    //data.img_list = filteredList;
    // Assign the combined and sorted list to data.img_list
    data.img_list = filteredList;

    std::cout << "data.img_list.size() = " << data.img_list.size() << std::endl;

    // Process the extrinsic files for training
    path = basePath + "/assets/AREngine/2024_09_27_11_20_29/sparse/";
    std::vector<std::string> matchingFiles;
    getFilesInDirectory(path, ".txt", matchingFiles);

    // Sort txt list
    std::sort(
        matchingFiles.begin(),
        matchingFiles.end(),
        [](const std::string& a, const std::string& b) {
            return extractNumber(a) < extractNumber(b);
        }
    );

    extractExtrinsics(matchingFiles, data.extr_tmp);
    std::vector<std::vector<float>> filteredExtrList;
    for (size_t i = 0; i < data.extr_tmp.size(); i += 3) {
        filteredExtrList.push_back(data.extr_tmp[i]);
    }
    data.extr_tmp = filteredExtrList;

    // Process the extrinsic files for testing
    path = basePath + "/assets/AREngine/2024_09_27_11_20_29/sparse_test/";
    std::vector<std::string> matchingFiles_test;
    getFilesInDirectory(path, ".txt", matchingFiles_test);

    // Sort txt list
    std::sort(
        matchingFiles_test.begin(),
        matchingFiles_test.end(),
        [](const std::string& a, const std::string& b) {
            return extractNumber(a) < extractNumber(b);
        }
    );

    extractExtrinsics(matchingFiles_test, data.extr_tmp_test);

    std::cout << "Extrinsics for training: " << data.extr_tmp.size() << std::endl;
    std::cout << "Extrinsics for testing: " << data.extr_tmp_test.size() << std::endl;

    return data;
}


DataLoader::InputLegoInitStruct DataLoader::loadInputLegoInitDatav2() {
auto basePath = FileOperator::FileOperator::GetInstance()->GetHapFilesDir();
    DataLoader::InputLegoInitStruct data;
    
    // 1. Load points and color data from points3D.txt
    std::string pointsFilePath = basePath + "/assets/case1/slam/points3D.txt";
    std::ifstream pointsFile(pointsFilePath);
    if (!pointsFile.is_open()) {
        throw std::runtime_error("Failed to open points3D.txt file.");
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
        data.opacities.push_back(1.0f);
        
        // Process the color data.
        // We interpret r, g, b (from points3D.txt) as f_dc_0, f_dc_1, and f_dc_2.
        DataLoader::InputLegoInitStruct::Data colorData = {};
        // Save r in index 0, g in index 16, and b in index 32.
        colorData.values[0]  = static_cast<float>((r/255.0-0.5)/C0);
        colorData.values[16] = static_cast<float>((g/255.0-0.5)/C0);
        colorData.values[32] = static_cast<float>((b/255.0-0.5)/C0);
        data.colors.push_back(colorData);
    }
    LOGI("Unique points loaded: %{public}zu", data.positions.size());
    
    // 2. Load image file names and extrinsics from images.txt
    std::string imagesFilePath = basePath + "/assets/case1/slam/images.txt";
    std::ifstream imagesFile(imagesFilePath);
    if (!imagesFile.is_open()) {
        throw std::runtime_error("Failed to open images.txt file.");
    }

    while (std::getline(imagesFile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int frameId;
        float qx, qy, qz, qw;
        float tx, ty, tz;
        int camera_id;
        std::string imageName;

        // Read the first 10 tokens:
        // frame_id, qx, qy, qz, qw, tx, ty, tz, camera_id, image_name
        if (!(iss >> frameId >> qx >> qy >> qz >> qw >> tx >> ty >> tz >> camera_id >> imageName)) {
            continue; // Skip invalid lines.
        }
        
        // Get the rest of the line which contains the extrinsic matrix in the form "mat4x4((...))"
        std::string extrinsicsToken;
        std::getline(iss, extrinsicsToken);
        size_t posExtr = extrinsicsToken.find("mat4x4(");
        if (posExtr != std::string::npos) {
            std::string mStr = extrinsicsToken.substr(posExtr);
            const std::string prefix = "mat4x4(";
            if (mStr.compare(0, prefix.size(), prefix) == 0) {
                mStr = mStr.substr(prefix.size());
            }
            // Remove trailing ')' characters.
            while (!mStr.empty() && mStr.back() == ')') {
                mStr.pop_back();
            }
            // Replace any '(' , ')' or ',' with spaces.
            std::string cleanedMatrix;
            for (char c : mStr) {
                if (c == '(' || c == ')' || c == ',') {
                    cleanedMatrix.push_back(' ');
                } else {
                    cleanedMatrix.push_back(c);
                }
            }
            // Now extract all floating-point values.
            std::istringstream issMatrix(cleanedMatrix);
            std::vector<float> mValues;
            float mVal;
            while (issMatrix >> mVal) {
                mValues.push_back(mVal);
            }
            // Expect 16 values for a full 4x4 matrix stored in column-major order.
            if (mValues.size() >= 16) {
                std::vector<float> extr(12);
                extr[0]  = mValues[0];
                extr[1]  = mValues[4];
                extr[2]  = mValues[8];
                extr[3]  = mValues[12];
                extr[4]  = -mValues[1];
                extr[5]  = -mValues[5];
                extr[6]  = -mValues[9];
                extr[7]  = -mValues[13];
                extr[8]  = -mValues[2];
                extr[9]  = -mValues[6];
                extr[10] = -mValues[10];
                extr[11] = -mValues[14];
                data.extr_tmp.push_back(extr);
            }
        }
        // Store the image name.
        data.img_list.push_back(basePath + "/assets/case1/frames/" + imageName);
    }
    data.extr_tmp_test = data.extr_tmp;
    
    // The testing extrinsics field is left empty.
    return data;
}

