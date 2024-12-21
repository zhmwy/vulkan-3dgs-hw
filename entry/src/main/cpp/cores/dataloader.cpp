#include "dataloader.h"
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <dirent.h>

#include "file/file_operator.h"

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



DataLoader::InputLegoInitStruct DataLoader::loadInputLegoInitData() {
    auto basePath = FileOperator::GetInstance()->GetHapFilesDir();
    const std::string filepath = basePath + "/" + "assets/lego/input.ply";
//     const std::string filepath = basePath + "/" + "assets/input_7000.ply";
    LOGI("InputLegoInitStruct, loading input from: %{public}s", filepath.c_str());
    //  const std::string filepath = basePath + "/" + "assets/lego/points3d.ply";
     
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
        r = plyFile.request_properties_from_element("vertex", { "f_dc_0" });
        g = plyFile.request_properties_from_element("vertex", { "f_dc_1" });
        b = plyFile.request_properties_from_element("vertex", { "f_dc_2" });
        
//         x = plyFile.request_properties_from_element("vertex", { "x" });
//         y = plyFile.request_properties_from_element("vertex", { "y" });
//         z = plyFile.request_properties_from_element("vertex", { "z" });
//         scale_x = plyFile.request_properties_from_element("vertex", { "scale_x" });
//         scale_y = plyFile.request_properties_from_element("vertex", { "scale_y" });
//         scale_z = plyFile.request_properties_from_element("vertex", { "scale_z" });
//         rotate_0 = plyFile.request_properties_from_element("vertex", { "rotate_0" });
//         rotate_1 = plyFile.request_properties_from_element("vertex", { "rotate_1" });
//         rotate_2 = plyFile.request_properties_from_element("vertex", { "rotate_2" });
//         rotate_3 = plyFile.request_properties_from_element("vertex", { "rotate_3" });
//         opacity = plyFile.request_properties_from_element("vertex", { "opacity" });
//         r = plyFile.request_properties_from_element("vertex", { "r" });
//         g = plyFile.request_properties_from_element("vertex", { "g" });
//         b = plyFile.request_properties_from_element("vertex", { "b" });
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

    std::string path = basePath + "/" + "assets/lego/ktx";
    std::string extension = ".ktx";

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
        if (filename.rfind("r_", 0) == 0 && filename.size() > 4 &&
            filename.substr(filename.size() - 4) == ".ktx") {
            data.img_list.push_back(path + "/" + filename);
        }
    }
    closedir(dir);
    // sort image list 
    std::sort(
        data.img_list.begin(), 
        data.img_list.end(),
        [](const std::string& a, const std::string& b) {
            return extractNumberKtx(a) < extractNumberKtx(b);
        }
    );
    std::cout << "data.img_list.size() = " << data.img_list.size() << std::endl;

    path = basePath + "/" + "assets/lego/sparse/";
    extension = ".txt";
    std::vector<std::string> matchingFiles;

    dir = opendir(path.c_str());
    if (!dir) {
        std::cerr << "Error: Could not open directory " << path << std::endl;
    }

    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;

        // Skip "." and ".." entries
        if (filename == "." || filename == "..") {
            continue;
        }

        // Check if the filename matches the pattern "r_*.ktx"
        if (filename.rfind("r_", 0) == 0 && filename.size() > 4 &&
            filename.substr(filename.size() - 4) == ".txt") {
            matchingFiles.push_back(path + "/" + filename);
        }
    }
    closedir(dir);
    // sort txt list
    std::sort(
        matchingFiles.begin(), 
        matchingFiles.end(),
        [](const std::string& a, const std::string& b) {
            return extractNumberTxt(a) < extractNumberTxt(b);
        }
    );
    std::cout << "Txt matchingFiles.size() = " << matchingFiles.size() << std::endl;

    for (const auto& entry : matchingFiles) {
        if (entry.find("camera.txt") == std::string::npos) {
            std::ifstream file(entry);
            if (!file.is_open()) throw std::runtime_error("Failed to open file.");

            std::string line;
            std::vector<float> extrinsics;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                float value;

                // Read four values per line
                while (iss >> value) {
                    extrinsics.push_back(value);
                }
            }
            data.extr_tmp.push_back(extrinsics);
        }
    }

    path = basePath + "/" + "assets/lego/sparse_test/";
    extension = ".txt";
    std::vector<std::string> matchingFiles_test;

    dir = opendir(path.c_str());
    if (!dir) {
        std::cerr << "Error: Could not open directory " << path << std::endl;
    }

    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;

        // Skip "." and ".." entries
        if (filename == "." || filename == "..") {
            continue;
        }

        // Check if the filename matches the pattern "r_*.ktx"
        if (filename.rfind("r_", 0) == 0 && filename.size() > 4 &&
            filename.substr(filename.size() - 4) == ".txt") {
            matchingFiles_test.push_back(path + "/" + filename);
        }
    }
    closedir(dir);
    // sort txt list
    std::sort(
        matchingFiles_test.begin(), 
        matchingFiles_test.end(),
        [](const std::string& a, const std::string& b) {
            return extractNumberTxt(a) < extractNumberTxt(b);
        }
    );
    std::cout << "Txt matchingFiles_test.size() = " << matchingFiles_test.size() << std::endl;

    for (const auto& entry : matchingFiles_test) {
        if (entry.find("camera.txt") == std::string::npos) {
            std::ifstream file(entry);
            if (!file.is_open()) throw std::runtime_error("Failed to open file.");

            std::string line;
            std::vector<float> extrinsics;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                float value;

                // Read four values per line
                while (iss >> value) {
                    extrinsics.push_back(value);
                }
            }
            data.extr_tmp_test.push_back(extrinsics);
        }
    }
    LOGI("InputLegoInitStruct successfully");

    return data;
}
