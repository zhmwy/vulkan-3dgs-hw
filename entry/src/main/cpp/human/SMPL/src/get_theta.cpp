#include "fit_smpl.h"


// Joint names
vector<string> skeleton_names = {
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
};

// Map joint names to their coordinates
map<string, Vector3d> current_joint_coords;
map<string, bool> is_valid_joint;

// New function to check if a point is valid
bool is_point_valid(const Vector3d& point) {
    return !(point[0] == 0.0 && point[1] == 0.0 && point[2] == 0.0);
}

// Modified function to read the PLY file and check for valid keypoints
bool read_ply_file(const string& filename, map<string, Vector3d>& joint_coords, map<string, bool>& valid_joints) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return false;
    }

    string line;
    int vertex_count = 0;
    bool header_ended = false;

    // Read header
    while (getline(infile, line)) {
        if (line.substr(0, 14) == "element vertex") {
            // Extract the number of vertices
            stringstream ss(line);
            string element_str, vertex_str;
            ss >> element_str >> vertex_str >> vertex_count;
        } else if (line == "end_header") {
            header_ended = true;
            break;
        }
    }

    if (!header_ended) {
        cerr << "Error: end_header not found in " << filename << endl;
        return false;
    }

    // Read vertex data
    int vertex_index = 0;
    while (vertex_index < vertex_count && getline(infile, line)) {
        stringstream ss(line);
        double x, y, z;
        int r, g, b;
        ss >> x >> y >> z >> r >> g >> b;

        if (vertex_index >= skeleton_names.size()) {
            cerr << "Warning: More vertices than expected in " << filename << endl;
            break;
        }

        string joint_name = skeleton_names[vertex_index];
        joint_coords[joint_name] = Vector3d(x, y, z);
        
        // Check if the joint is valid (not 0,0,0)
        valid_joints[joint_name] = is_point_valid(Vector3d(x, y, z));

        vertex_index++;
    }

    if (vertex_index < vertex_count) {
        cerr << "Error: Not enough vertex data in " << filename << endl;
        return false;
    }

    infile.close();
    return true;
}

int get_theta_from_skeleton(string path, std::vector<Vector3d>& thetas) {
    string filename = path + "/skeleton.ply";

    // Read the skeleton data and track valid points
    if (!read_ply_file(filename, current_joint_coords, is_valid_joint)) {
        cerr << "Failed to read skeleton data from " << filename << endl;
        return 1;
    }

    // Count valid keypoints in the limbs (keypoints 5-16)
    int valid_limb_count = 0;
    for (int i = 5; i <= 16; i++) {
        if (i < skeleton_names.size() && is_valid_joint[skeleton_names[i]]) {
            valid_limb_count++;
        }
    }

    // Check if we have at least 4 valid keypoints in the limbs
    if (valid_limb_count < 4) {
        cerr << "Not enough valid limb keypoints for theta estimation" << endl;
        return 1;
    }

    Vector3d zero_vec(0, 0, 0);
    Matrix3d identity_mat = Matrix3d::Identity();

    // Left Shoulder
    Vector3d left_shoulder_theta;
    Matrix3d left_R_shoulder;
    if (is_valid_joint["left_shoulder"] && is_valid_joint["left_elbow"]) {
        tie(left_shoulder_theta, left_R_shoulder) = compute_shoulder_axis_angle("left", current_joint_coords);
    } else {
        left_shoulder_theta = zero_vec;
        left_R_shoulder = identity_mat;
    }
    thetas.push_back(left_shoulder_theta);

    // Right Shoulder
    Vector3d right_shoulder_theta;
    Matrix3d right_R_shoulder;
    if (is_valid_joint["right_shoulder"] && is_valid_joint["right_elbow"]) {
        tie(right_shoulder_theta, right_R_shoulder) = compute_shoulder_axis_angle("right", current_joint_coords);
    } else {
        right_shoulder_theta = zero_vec;
        right_R_shoulder = identity_mat;
    }
    thetas.push_back(right_shoulder_theta);

    // Left Elbow
    Vector3d left_elbow_theta;
    if (is_valid_joint["left_shoulder"] && is_valid_joint["left_elbow"] && is_valid_joint["left_wrist"]) {
        left_elbow_theta = compute_elbow_axis_angle("left", current_joint_coords, left_R_shoulder);
    } else {
        left_elbow_theta = zero_vec;
    }
    thetas.push_back(left_elbow_theta);

    // Right Elbow
    Vector3d right_elbow_theta;
    if (is_valid_joint["right_shoulder"] && is_valid_joint["right_elbow"] && is_valid_joint["right_wrist"]) {
        right_elbow_theta = compute_elbow_axis_angle("right", current_joint_coords, right_R_shoulder);
    } else {
        right_elbow_theta = zero_vec;
    }
    thetas.push_back(right_elbow_theta);

    // Left Hip
    Vector3d left_hip_theta;
    Matrix3d left_R_hip;
    if (is_valid_joint["left_hip"] && is_valid_joint["left_knee"]) {
        tie(left_hip_theta, left_R_hip) = compute_hip_axis_angle("left", current_joint_coords);
    } else {
        left_hip_theta = zero_vec;
        left_R_hip = identity_mat;
    }
    thetas.push_back(left_hip_theta);

    // Right Hip
    Vector3d right_hip_theta;
    Matrix3d right_R_hip;
    if (is_valid_joint["right_hip"] && is_valid_joint["right_knee"]) {
        tie(right_hip_theta, right_R_hip) = compute_hip_axis_angle("right", current_joint_coords);
    } else {
        right_hip_theta = zero_vec;
        right_R_hip = identity_mat;
    }
    thetas.push_back(right_hip_theta);

    // Left Knee
    Vector3d left_knee_theta;
    if (is_valid_joint["left_hip"] && is_valid_joint["left_knee"] && is_valid_joint["left_ankle"]) {
        left_knee_theta = compute_knee_axis_angle("left", current_joint_coords, left_R_hip);
    } else {
        left_knee_theta = zero_vec;
    }
    thetas.push_back(left_knee_theta);

    // Right Knee
    Vector3d right_knee_theta;
    if (is_valid_joint["right_hip"] && is_valid_joint["right_knee"] && is_valid_joint["right_ankle"]) {
        right_knee_theta = compute_knee_axis_angle("right", current_joint_coords, right_R_hip);
    } else {
        right_knee_theta = zero_vec;
    }
    thetas.push_back(right_knee_theta);

    return 0;
}

// The rest of the functions are unchanged from the original code