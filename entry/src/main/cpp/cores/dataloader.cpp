#include "dataloader.h"
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <dirent.h>

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


void DataLoader::InputInitStruct::computeNormalization() {
    cam_centers.clear();  // 清空之前存的相机中心

    // 遍历所有相机的外参矩阵
    for (const auto& extrinsic : extr_tmp) {
        if (extrinsic.size() != 12) {
            std::cerr << "Error: Each extrinsic matrix must have 12 elements (3x4)!" << std::endl;
            continue;
        }

        glm::mat4 W2C(1.0f);  // 4x4 齐次矩阵
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                W2C[i][j] = static_cast<float>(extrinsic[i * 4 + j]);
            }
        }

        // 计算 W2C 的逆矩阵，得到 C2W
        glm::mat4 C2W = glm::inverse(W2C);

        // 提取 C2W 的位移分量（最后一列的前三个元素）
        glm::vec3 cam_center(C2W[0][3], C2W[1][3], C2W[2][3]);
        cam_centers.push_back(cam_center);
    }

    if (cam_centers.empty()) {
        std::cerr << "Error: No valid camera centers computed!" << std::endl;
        return;
    }

    // **计算相机中心的均值**
    glm::vec3 center(0.0f);
    for (const auto& c : cam_centers) {
        center += c;
    }
    center /= static_cast<float>(cam_centers.size());

    // **计算最大欧几里得距离（对角线）**
    float max_dist = 0.0f;
    for (const auto& c : cam_centers) {
        float dist = glm::length(c - center);
        if (dist > max_dist) {
            max_dist = dist;
        }
    }

    // **计算 radius（加 10% buffer）**
    radius = max_dist * 1.1f;

    // **计算 translate（相机中心对齐到原点）**
    translate = -center;


    std::cout << "Translate: (" << translate.x << ", " << translate.y << ", " << translate.z << ")" << std::endl;
    std::cout << "Radius: " << radius << std::endl;

    return;
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
    
    // **标准化 extr_tmp（w2c 矩阵）**
    for (auto& extr : extr_tmp) {
        if (extr.size() < 12) continue;

        glm::mat4 W2C(1.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                W2C[i][j] = extr[i * 4 + j];
            }
        }

        glm::mat4 C2W = glm::inverse(W2C);
        glm::vec3 cam_center(C2W[0][3], C2W[1][3], C2W[2][3]);

        // **标准化 cam_center**
        glm::vec3 normalized_center = (cam_center - mean) / stddev;

        C2W[0][3] = normalized_center.x;
        C2W[1][3] = normalized_center.y;
        C2W[2][3] = normalized_center.z;

        W2C = glm::inverse(C2W);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                extr[i * 4 + j] = W2C[i][j];
            }
        }
    }

    // **标准化 extr_tmp_test（同样方式）**
    for (auto& extr : extr_tmp_test) {
        if (extr.size() < 12) continue;

        glm::mat4 W2C(1.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                W2C[i][j] = extr[i * 4 + j];
            }
        }

        glm::mat4 C2W = glm::inverse(W2C);
        glm::vec3 cam_center(C2W[0][3], C2W[1][3], C2W[2][3]);

        glm::vec3 normalized_center = (cam_center - mean) / stddev;
        C2W[0][3] = normalized_center.x;
        C2W[1][3] = normalized_center.y;
        C2W[2][3] = normalized_center.z;

        W2C = glm::inverse(C2W);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                extr[i * 4 + j] = W2C[i][j];
            }
        }
    }

    std::cout << "Positions, extr_tmp, extr_tmp_test normalized correctly." << std::endl;
}


void DataLoader::InputInitStruct::computePositionStats() {
    if (positions.empty()) {
        std::cerr << "Error: No position data available!" << std::endl;
        return;
    }

    // **计算均值**
    mean = glm::vec3(0.0, 0.0, 0.0);
    for (const auto& pos : positions) {
        mean += pos;
    }
    mean /= static_cast<float>(positions.size());

    // **计算方差**
    double sum = 0;
    for (const auto& pos : positions) {
        float dist = glm::length(pos - mean);  // **计算欧几里得距离**
        sum += dist * dist;  // **累加平方距离**
    }
    variance = sum / static_cast<double>(positions.size());  // **求均值**

    // **存储到结构体**
    this->mean = mean;
    this->variance = variance;

    // **输出均值和方差**
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

    // 预分配空间
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

#if 1
DataLoader::InputInitStruct DataLoader::loadInputLegoInitData() {
    auto basePath = FileOperator::GetInstance()->GetHapFilesDir();
    std::string filepath = basePath + "/" + "assets/AREngine/2024_09_27_11_20_29/input_AREngine.ply";
    DataLoader::InputInitStruct data;
    data.basepath = basePath;
#if 1
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
#endif


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
    
    path = basePath + "/assets/AREngine/2024_09_27_11_20_29/intrinsics/";
    std::vector<std::string> matchingFiles_intr;
    getFilesInDirectory(path, ".txt", matchingFiles_intr);

    std::sort(matchingFiles_intr.begin(), matchingFiles_intr.end(),
        [](const std::string& a, const std::string& b) {
            return extractNumber(a) < extractNumber(b);
        });

    extractIntrinsics(matchingFiles_intr, data.intr_tmp);
    
    return data;
}
#endif

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
}


DataLoader::InputInitStruct DataLoader::loadInputInitDataPM() {
    auto basePath = FileOperator::GetInstance()->GetHapFilesDir();
    const std::string filepath = basePath + "/" + "assets/AREngine/2024_09_27_11_20_29/downsample.ply";

    DataLoader::InputInitStruct data;
    data.basepath = basePath;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open PLY file.");

    PlyFile plyFile;
    plyFile.parse_header(file);

    // Declare shared pointers to request each property from the PLY file
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
    
    float C0 = 0.28209479177387814;
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

        data.scales.push_back(glm::vec3(-4.5f, -4.5f, -4.5f));
        data.rotations.push_back(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        data.opacities.push_back(-2.197224617004395);
        
        // 处理颜色数据
        DataLoader::InputInitStruct::Data colorData = {};
        
        float _r = (reinterpret_cast<uint8_t*>(r->buffer.get())[i] / 255.0 - 0.5) / C0;
        float _g = (reinterpret_cast<uint8_t*>(g->buffer.get())[i] / 255.0 - 0.5) / C0;
        float _b = (reinterpret_cast<uint8_t*>(b->buffer.get())[i] / 255.0 - 0.5) / C0;
        
        colorData.values[0] = _r;
        colorData.values[16] = _g;
        colorData.values[32] = _b;
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
    
    path = basePath + "/assets/AREngine/2024_09_27_11_20_29/intrinsics/";
    std::vector<std::string> matchingFiles_intr;
    getFilesInDirectory(path, ".txt", matchingFiles_intr);

    std::sort(matchingFiles_intr.begin(), matchingFiles_intr.end(),
        [](const std::string& a, const std::string& b) {
            return extractNumber(a) < extractNumber(b);
        });

    extractIntrinsics(matchingFiles_intr, data.intr_tmp);
    
    return data;
}


void DataLoader::loadExtrinsics(std::string dataPath, DataLoader::InputInitStruct& data){
    std::string imagesFilePath = dataPath + "/slam/images.txt";
    std::string imageFolder = "/images/";
    std::ifstream imagesFile(imagesFilePath);
    if (!imagesFile.is_open()) {
        throw std::runtime_error("Failed to open images.txt file.");
    }

    std::string line;

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
        data.img_list.push_back(dataPath + imageFolder + imageName);
    }
    data.extr_tmp_test = data.extr_tmp;
    
    std::vector<std::string> filteredList;
    for (size_t i = 0; i < data.img_list.size(); i += 1) {
        filteredList.push_back(data.img_list[i]);
    }
    data.img_list = filteredList;
    
    std::vector<std::vector<float>> filteredExtrList;
    for (size_t i = 0; i < data.extr_tmp.size(); i += 1) {
        filteredExtrList.push_back(data.extr_tmp[i]);
    }
    data.extr_tmp = filteredExtrList;
}

void DataLoader::loadIntrinsics(std::string dataPath, DataLoader::InputInitStruct &data){
    std::string cameraFilePath = dataPath + "/slam/cameras2.txt";
    std::ifstream camerasFile(cameraFilePath);
    std::string line;
    data.intr_tmp.clear();
    while (std::getline(camerasFile, line)) {
        if (line.empty() || line[0] == '#'){
            LOGE("Line is empty");
            continue;
        }
        std::istringstream iss(line);
        int cameraId;
        std::string camera_type;
        float fx, fy, cx, cy;
        int height, width;
        int camera_id;
        std::string imageName;

        // Read the first 8 tokens:
        if (!(iss >> cameraId >> camera_type >> width >> height >> fx >> fy >> cx >> cy)) {
            LOGW("Invalid line encountered.");
            continue; // Skip invalid lines.
        }

        assert(camera_type == "PINHOLE" || camera_type == "pinhole");

        std::vector<float> intr = { fx,fy,cx,cy };
        data.intr_tmp.push_back(intr);
    }
    LOGI("Intr size: %{public}zu; Extr size: %{public}zu", data.intr_tmp.size(), data.extr_tmp.size());
    assert(data.intr_tmp.size() == 1 || data.intr_tmp.size() == data.extr_tmp.size());
}

void DataLoader::loadInputLegoInitDatav2(std::string dataFolder, DataLoader::InputInitStruct& data, bool& resumed) {
    auto basePath = FileOperator::FileOperator::GetInstance()->GetHapFilesDir();
    std::string resume_gaussian_path = basePath + dataFolder + "/point_cloud.ply";

    loadInputGaussianData(resume_gaussian_path, data);
    data.basepath = basePath;
    std::string dataPath = basePath + dataFolder;
    
    if (data.positions.size() == 0){
        LOGW("Resuming from ply failed, starting from points3D.txt");
        std::vector<float> head_bbox;
        bool is_human = HumanPreprocess(dataPath, head_bbox);
//         bool is_human = false;
        
        std::string pointsFilePath = dataPath + "/slam/points3D.txt";
        load3DPoints(pointsFilePath, data);
        
        if(is_human){
            std::string smplPath = dataPath + "/slam/smpl.txt";
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
    
    loadExtrinsics(dataPath, data);
    loadIntrinsics(dataPath, data);
    
}


DataLoader::InputInitStruct DataLoader::loadHeadGaussians(std::string dataPath, DataLoader::InputInitStruct& data) {
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