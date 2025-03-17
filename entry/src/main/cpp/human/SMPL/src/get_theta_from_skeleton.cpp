//
// Created on 2024/11/19.
//
// Node APIs are not fully supported. To solve the compilation error of the interface cannot be found,
// please include "napi/native_api.h".
#include "fit_smpl.h"

using namespace std;
using namespace Eigen;

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

// Function declarations
Matrix3d get_joint_local_coordinate_system(const map<string, Vector3d>& joint_coords);
Matrix3d get_subjoint_local_coordinate_system(const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R, const string& parent_joint_name, const string& joint_name);
Matrix3d get_knee_local_coordinate_system(const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R, const string& parent_joint_name, const string& joint_name);
Vector3d calculate_axis_angle(const Vector3d& local_vector, const string& side, const Vector3d& default_axis, int ignore_axis = -1);
pair<Vector3d, Matrix3d> compute_joint_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const map<string, string>& joint_names, const vector<string>& parent_joints, const Vector3d& default_axis, int ignore_axis);
Vector3d compute_subjoint_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R, const map<string, string>& joint_names, const Vector3d& default_axis, int ignore_axis);
pair<Vector3d, Matrix3d> compute_shoulder_axis_angle(const string& side, const map<string, Vector3d>& joint_coords);
Vector3d compute_elbow_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R);
pair<Vector3d, Matrix3d> compute_hip_axis_angle(const string& side, const map<string, Vector3d>& joint_coords);
Vector3d compute_knee_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R);
void print_axis_angle(const string& name, const Vector3d& axis_angle);
bool read_ply_file(const string& filename, map<string, Vector3d>& joint_coords);

int get_theta_from_skeleton(string path, std::vector<Vector3d>& thetas) {
    string filename = path + "/skeleton.ply";  // Replace with your actual filename

    // Read the skeleton data from the PLY file
    if (!read_ply_file(filename, current_joint_coords)) {
        cerr << "Failed to read skeleton data from " << filename << endl;
        return 1;
    }

    // Compute axis-angle parameters for shoulders and elbows
    auto [left_shoulder_axis_angle, left_R_shoulder] = compute_shoulder_axis_angle("left", current_joint_coords);
    auto [right_shoulder_axis_angle, right_R_shoulder] = compute_shoulder_axis_angle("right", current_joint_coords);
    thetas.push_back(left_shoulder_axis_angle);
    thetas.push_back(right_shoulder_axis_angle);
    
    Vector3d left_elbow_axis_angle = compute_elbow_axis_angle("left", current_joint_coords, left_R_shoulder);
    Vector3d right_elbow_axis_angle = compute_elbow_axis_angle("right", current_joint_coords, right_R_shoulder);
    thetas.push_back(left_elbow_axis_angle);
    thetas.push_back(right_elbow_axis_angle);
    // Compute axis-angle parameters for hips and knees
    auto [left_hip_axis_angle, left_R_hip] = compute_hip_axis_angle("left", current_joint_coords);
    auto [right_hip_axis_angle, right_R_hip] = compute_hip_axis_angle("right", current_joint_coords);
    thetas.push_back(left_hip_axis_angle);
    thetas.push_back(right_hip_axis_angle);
    
    Vector3d left_knee_axis_angle = compute_knee_axis_angle("left", current_joint_coords, left_R_hip);
    Vector3d right_knee_axis_angle = compute_knee_axis_angle("right", current_joint_coords, right_R_hip);
    thetas.push_back(left_knee_axis_angle);
    thetas.push_back(right_knee_axis_angle);
    
    // Print the results
    print_axis_angle("Left Shoulder", left_shoulder_axis_angle);
    print_axis_angle("Right Shoulder", right_shoulder_axis_angle);
    print_axis_angle("Left Elbow", left_elbow_axis_angle);
    print_axis_angle("Right Elbow", right_elbow_axis_angle);
    print_axis_angle("Left Hip", left_hip_axis_angle);
    print_axis_angle("Right Hip", right_hip_axis_angle);
    print_axis_angle("Left Knee", left_knee_axis_angle);
    print_axis_angle("Right Knee", right_knee_axis_angle);

    return 0;
}

// Function to read the PLY file and populate joint_coords
bool read_ply_file(const string& filename, map<string, Vector3d>& joint_coords) {
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
            //cout << "Vertex count: " << vertex_count << endl;
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

        vertex_index++;
    }

    if (vertex_index < vertex_count) {
        cerr << "Error: Not enough vertex data in " << filename << endl;
        return false;
    }

    infile.close();
    return true;
}

// Function definitions

Matrix3d get_joint_local_coordinate_system(const map<string, Vector3d>& joint_coords) {
    /*
    Determines the local coordinate system at the shoulders or hips.
    Returns a rotation matrix whose columns are the local x, y, z axes.
    */
    Vector3d left_joint = joint_coords.at("left_shoulder");
    Vector3d right_joint = joint_coords.at("right_shoulder");
    Vector3d left_hip = joint_coords.at("left_hip");
    Vector3d right_hip = joint_coords.at("right_hip");

    // x-axis from right_joint to left_joint (consistent for both sides)
    Vector3d x_axis = left_joint - right_joint;
    x_axis.normalize();

    // y-axis is the average direction from hips to joints
    Vector3d v1 = left_joint - left_hip;
    v1.normalize();
    Vector3d v2 = right_joint - right_hip;
    v2.normalize();
    Vector3d y_axis = (v1 + v2) / 2.0;
    y_axis.normalize();

    // z-axis is cross product of x and y axes
    Vector3d z_axis = x_axis.cross(y_axis);
    z_axis.normalize();

    // Re-orthogonalize y-axis to ensure orthogonality
    y_axis = z_axis.cross(x_axis);
    y_axis.normalize();

    // Create rotation matrix R
    Matrix3d R;
    R.col(0) = x_axis;
    R.col(1) = y_axis;
    R.col(2) = z_axis;

    return R;
}

Matrix3d get_subjoint_local_coordinate_system(const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R, const string& parent_joint_name, const string& joint_name) {
    /*
    Determines the local coordinate system at the subjoint (elbow or knee).
    */
    Vector3d parent_joint = joint_coords.at(parent_joint_name);
    Vector3d joint = joint_coords.at(joint_name);

    Vector3d x_axis = joint - parent_joint;
    x_axis.normalize();

    // Use the y-axis of the parent joint's coordinate system
    Vector3d y_axis_parent = parent_R.col(1);

    // Make y-axis orthogonal to x-axis
    Vector3d y_axis = y_axis_parent - (y_axis_parent.dot(x_axis)) * x_axis;
    double y_axis_norm = y_axis.norm();

    if (y_axis_norm < 1e-8) {
        // If y-axis cannot be computed, choose an arbitrary vector not parallel to x_axis
        Vector3d arbitrary_vector(0, 1, 0);  // Assuming it's not parallel to x_axis
        y_axis = arbitrary_vector - (arbitrary_vector.dot(x_axis)) * x_axis;
        y_axis.normalize();
    } else {
        y_axis /= y_axis_norm;
    }

    // z-axis is cross product of x and y axes
    Vector3d z_axis = x_axis.cross(y_axis);
    z_axis.normalize();

    // Re-orthogonalize y-axis to ensure orthogonality
    y_axis = z_axis.cross(x_axis);
    y_axis.normalize();

    // Create rotation matrix R_subjoint
    Matrix3d R_subjoint;
    R_subjoint.row(0) = x_axis.transpose();
    R_subjoint.row(1) = y_axis.transpose();
    R_subjoint.row(2) = z_axis.transpose();

    return R_subjoint;
}

Matrix3d get_knee_local_coordinate_system(const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R, const string& parent_joint_name, const string& joint_name) {
    /*
    Determines the local coordinate system at the knee.
    */
    Vector3d joint = joint_coords.at(joint_name);
    Vector3d parent_joint = joint_coords.at(parent_joint_name);

    // y-axis from knee to hip
    Vector3d y_axis = parent_joint - joint;  // From knee to hip
    y_axis.normalize();

    // Use the z-axis of the parent joint's coordinate system as a reference
    Vector3d z_axis_parent = parent_R.col(2);

    // x-axis is orthogonal to y-axis, in the plane defined by y_axis and z_axis_parent
    Vector3d x_axis = y_axis.cross(z_axis_parent);
    double x_axis_norm = x_axis.norm();

    if (x_axis_norm < 1e-8) {
        // If x-axis cannot be computed, choose an arbitrary vector not parallel to y_axis
        Vector3d arbitrary_vector(1, 0, 0);  // Assuming it's not parallel to y_axis
        x_axis = y_axis.cross(arbitrary_vector);
        x_axis.normalize();
    } else {
        x_axis /= x_axis_norm;
    }

    // z-axis is cross product of x and y axes
    Vector3d z_axis = x_axis.cross(y_axis);
    z_axis.normalize();

    // Re-orthogonalize x-axis to ensure orthogonality
    x_axis = y_axis.cross(z_axis);
    x_axis.normalize();

    // Create rotation matrix R_subjoint
    Matrix3d R_subjoint;
    R_subjoint.row(0) = x_axis.transpose();
    R_subjoint.row(1) = y_axis.transpose();
    R_subjoint.row(2) = z_axis.transpose();

    return R_subjoint;
}

Vector3d calculate_axis_angle(const Vector3d& local_vector, const string& side, const Vector3d& default_axis, int ignore_axis) {
    /*
    Calculates the axis-angle representation.
    */
    // Default direction vector
    Vector3d v0 = default_axis;
    // Normalize the observed vector
    Vector3d v1 = local_vector;
    double v1_norm = v1.norm();

    if (v1_norm < 1e-8) {
        // No rotation
        return Vector3d(0, 0, 0);
    }
    v1 /= v1_norm;

    // Compute the rotation axis (cross product)
    Vector3d cross = v0.cross(v1);
    double cross_norm = cross.norm();

    // Compute the angle between v0 and v1
    double cos_theta = v0.dot(v1);
    cos_theta = std::clamp(cos_theta, -1.0, 1.0);
    double theta = acos(cos_theta);

    Vector3d axis;

    if (cross_norm < 1e-8) {
        if (cos_theta > 0) {
            // Vectors are aligned, no rotation
            return Vector3d(0, 0, 0);
        } else {
            // Vectors are opposite, 180-degree rotation
            axis = Vector3d(0, 1, 0);  // Choose arbitrary axis
        }
    } else {
        // Axis is cross product
        axis = cross / cross_norm;

        if (ignore_axis != -1) {
            axis[ignore_axis] = 0;  // Zero out specified axis
            double axis_norm = axis.norm();
            if (axis_norm < 1e-8) {
                // If axis is zero after zeroing out, choose arbitrary axis
                axis = Vector3d::Zero();
                axis[(ignore_axis + 1) % 3] = 1.0;  // Choose next axis
            } else {
                axis /= axis_norm;
            }
        }
    }

    // Calculate the axis-angle vector
    Vector3d axis_angle = axis * theta;

    return axis_angle;
}

pair<Vector3d, Matrix3d> compute_joint_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const map<string, string>& joint_names, const vector<string>& parent_joints, const Vector3d& default_axis, int ignore_axis) {
    /*
    Computes the axis-angle parameters for a joint (shoulder or hip).
    */
    Matrix3d R = get_joint_local_coordinate_system(joint_coords);
    // Determine the limb vector in local coordinates
    string distal_joint = joint_names.at("child");
    Vector3d limb_vector = joint_coords.at(distal_joint) - joint_coords.at(joint_names.at(side));
    Vector3d local_limb_vector = R.transpose() * limb_vector;
    // Calculate the axis-angle representation
    Vector3d axis_angle = calculate_axis_angle(local_limb_vector, side, default_axis, ignore_axis);

    return make_pair(axis_angle, R);
}

Vector3d compute_subjoint_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R, const map<string, string>& joint_names, const Vector3d& default_axis, int ignore_axis) {
    /*
    Computes the axis-angle parameters for a subjoint (elbow or knee).
    */
    Matrix3d R_subjoint;
    if (side == "left") {
        R_subjoint = get_subjoint_local_coordinate_system(joint_coords, parent_R, joint_names.at("parent"), joint_names.at("joint"));
    } else {
        R_subjoint = get_subjoint_local_coordinate_system(joint_coords, parent_R, joint_names.at("joint"), joint_names.at("parent"));
    }

    // Compute the limb vector in subjoint local coordinates
    string proximal_joint = joint_names.at("joint");
    string distal_joint = joint_names.at("child");
    Vector3d limb_vector = joint_coords.at(distal_joint) - joint_coords.at(proximal_joint);
    Vector3d local_limb_vector = R_subjoint * limb_vector;
    Vector3d axis_angle = calculate_axis_angle(local_limb_vector, side, default_axis, ignore_axis);

    return axis_angle;
}

pair<Vector3d, Matrix3d> compute_shoulder_axis_angle(const string& side, const map<string, Vector3d>& joint_coords) {
    /*
    Computes the axis-angle parameters for the shoulder joint.
    */
    map<string, string> joint_names = {
        {"left", "left_shoulder"},
        {"right", "right_shoulder"},
        {"child", side + "_elbow"},
    };

    vector<string> parent_joints = {"left_hip", "right_hip"};

    Vector3d default_axis;
    if (side == "left") {
        default_axis = Vector3d(1, 0, 0);  // Left arm default direction
    } else {
        default_axis = Vector3d(-1, 0, 0);  // Right arm default direction
    }

    auto [axis_angle, R] = compute_joint_axis_angle(side, joint_coords, joint_names, parent_joints, default_axis, 0);
    return make_pair(axis_angle, R);
}

Vector3d compute_elbow_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R) {
    /*
    Computes the axis-angle parameters for the elbow joint.
    */
    map<string, string> joint_names = {
        {"parent", side + "_shoulder"},
        {"joint", side + "_elbow"},
        {"child", side + "_wrist"},
    };

    Vector3d default_axis;
    if (side == "left") {
        default_axis = Vector3d(1, 0, 0);  // Left elbow default direction
    } else {
        default_axis = Vector3d(-1, 0, 0);  // Right elbow default direction
    }

    Vector3d axis_angle = compute_subjoint_axis_angle(side, joint_coords, parent_R, joint_names, default_axis, 0);
    return axis_angle;
}

pair<Vector3d, Matrix3d> compute_hip_axis_angle(const string& side, const map<string, Vector3d>& joint_coords) {
    /*
    Computes the axis-angle parameters for the hip joint.
    */
    map<string, string> joint_names = {
        {"left", "left_hip"},
        {"right", "right_hip"},
        {"child", side + "_knee"},
    };

    vector<string> parent_joints = {"left_shoulder", "right_shoulder", "nose"};

    Vector3d default_axis(0, -1, 0);  // Leg default direction (downwards)

    auto [axis_angle, R] = compute_joint_axis_angle(side, joint_coords, joint_names, parent_joints, default_axis, 1);
    return make_pair(axis_angle, R);
}

Vector3d compute_knee_axis_angle(const string& side, const map<string, Vector3d>& joint_coords, const Matrix3d& parent_R) {
    /*
    Computes the axis-angle parameters for the knee joint.
    */
    map<string, string> joint_names = {
        {"parent", side + "_hip"},
        {"joint", side + "_knee"},
        {"child", side + "_ankle"},
    };

    // Get knee local coordinate system
    Matrix3d R_subjoint = get_knee_local_coordinate_system(joint_coords, parent_R, joint_names.at("parent"), joint_names.at("joint"));

    // Compute the limb vector in subjoint local coordinates
    string proximal_joint = joint_names.at("joint");
    string distal_joint = joint_names.at("child");
    Vector3d limb_vector = joint_coords.at(distal_joint) - joint_coords.at(proximal_joint);
    Vector3d local_limb_vector = R_subjoint * limb_vector;

    // Default axis is along negative y-axis in the knee's local coordinate system
    Vector3d default_axis(0, -1, 0);  // From knee to ankle

    // Calculate the axis-angle representation
    Vector3d axis_angle = calculate_axis_angle(local_limb_vector, side, default_axis, 1);  // Ignore rotations around x-axis

    return axis_angle;
}

void print_axis_angle(const string& name, const Vector3d& axis_angle) {
    double angle = axis_angle.norm();
    Vector3d axis_normalized;
    if (angle > 1e-8) {
        axis_normalized = axis_angle / angle;
    } else {
        axis_normalized = Vector3d(0, 0, 0);
    }

    cout << fixed << setprecision(6);
    cout << name << " axis-angle parameters (rotations around specified axis are considered):" << endl;
    cout << "Axis (normalized): [" << axis_normalized[0] << ", " << axis_normalized[1] << ", " << axis_normalized[2] << "]" << endl;
    cout << "Angle (radians): " << angle << endl;
    cout << "Axis-angle vector: [" << axis_angle[0] << ", " << axis_angle[1] << ", " << axis_angle[2] << "]" << endl;
    cout << endl;
}