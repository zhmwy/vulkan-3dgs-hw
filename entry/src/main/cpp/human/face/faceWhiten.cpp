#include "base/math/vector.h"
#include "face.h"
#include <cmath>
#include <algorithm>
#include <numeric>

using Vec3 = Base::Math::Vec3;
using PointCloud = std::vector<Vec3>;

// Constants for color conversion
const float C0 = 0.28209479177387814f;

// Extract RGB color from Data structure
Vec3 extractRGBFromData(const DataLoader::InputInitStruct::Data& colorData) {
    float r = (colorData.values[0] * C0 + 0.5f) * 255.0f;
    float g = (colorData.values[16] * C0 + 0.5f) * 255.0f;
    float b = (colorData.values[32] * C0 + 0.5f) * 255.0f;
    
    // Clamp to valid range
    r = std::clamp(r, 0.0f, 255.0f);
    g = std::clamp(g, 0.0f, 255.0f);
    b = std::clamp(b, 0.0f, 255.0f);
    
    return Vec3(r, g, b);
}

// Set RGB color back to Data structure
void setRGBToData(DataLoader::InputInitStruct::Data& colorData, const Vec3& rgb) {
    colorData.values[0] = static_cast<float>((rgb.x/255.0f - 0.5f) / C0);
    colorData.values[16] = static_cast<float>((rgb.y/255.0f - 0.5f) / C0);
    colorData.values[32] = static_cast<float>((rgb.z/255.0f - 0.5f) / C0);
}

// RGB to HSV conversion
Vec3 rgbToHSV(const Vec3& rgb) {
    float r = rgb.x / 255.0f;
    float g = rgb.y / 255.0f;
    float b = rgb.z / 255.0f;
    
    float maxVal = std::max({r, g, b});
    float minVal = std::min({r, g, b});
    float delta = maxVal - minVal;
    
    float h = 0.0f;
    if (delta > 0.0001f) {
        if (maxVal == r) {
            h = 60.0f * (fmod((g - b) / delta, 6.0f));
        } else if (maxVal == g) {
            h = 60.0f * ((b - r) / delta + 2.0f);
        } else {
            h = 60.0f * ((r - g) / delta + 4.0f);
        }
    }
    if (h < 0) h += 360.0f;
    
    float s = (maxVal > 0.0001f) ? (delta / maxVal) : 0.0f;
    float v = maxVal;
    
    return Vec3(h/2.0f, s * 255.0f, v * 255.0f); // OpenCV HSV scale
}

// RGB to YCrCb conversion
Vec3 rgbToYCrCb(const Vec3& rgb) {
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;
    
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    float cr = (r - y) * 0.713f + 128.0f;
    float cb = (b - y) * 0.564f + 128.0f;
    
    return Vec3(y, cr, cb);
}

// Simple skin detection without OpenCV
bool isSkinColor(const Vec3& rgb) {
    Vec3 hsv = rgbToHSV(rgb);
    Vec3 ycrcb = rgbToYCrCb(rgb);
    
    // HSV range check
    bool hsv_check = (hsv.x >= 0 && hsv.x <= 20) && 
                     (hsv.y >= 20 && hsv.y <= 255) && 
                     (hsv.z >= 70 && hsv.z <= 255);
    
    // YCrCb range check
    bool ycrcb_check = (ycrcb.y >= 133 && ycrcb.y <= 173) && 
                       (ycrcb.z >= 77 && ycrcb.z <= 127);
    
    return hsv_check && ycrcb_check;
}

// RGB to LAB conversion
Vec3 rgbToLAB(const Vec3& rgb) {
    // First convert to XYZ
    float r = rgb.x / 255.0f;
    float g = rgb.y / 255.0f;
    float b = rgb.z / 255.0f;
    
    // Apply gamma correction
    auto gammaCorrect = [](float val) {
        return (val > 0.04045f) ? 
               std::pow((val + 0.055f) / 1.055f, 2.4f) : 
               val / 12.92f;
    };
    
    r = gammaCorrect(r);
    g = gammaCorrect(g);
    b = gammaCorrect(b);
    
    // RGB to XYZ
    float x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    float y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    float z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
    
    // Normalize by D65 illuminant
    x /= 0.95047f;
    y /= 1.00000f;
    z /= 1.08883f;
    
    // XYZ to LAB
    auto f = [](float t) {
        return (t > 0.008856f) ? 
               std::cbrt(t) : 
               (7.787f * t + 16.0f / 116.0f);
    };
    
    float fx = f(x);
    float fy = f(y);
    float fz = f(z);
    
    float L = 116.0f * fy - 16.0f;
    float a = 500.0f * (fx - fy);
    float b_val = 200.0f * (fy - fz);
    
    return Vec3(L, a, b_val);
}

// LAB to RGB conversion
Vec3 labToRGB(const Vec3& lab) {
    float L = lab.x;
    float a = lab.y;
    float b = lab.z;
    
    // LAB to XYZ
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b / 200.0f;
    
    auto invF = [](float t) {
        float t3 = t * t * t;
        return (t3 > 0.008856f) ? t3 : (t - 16.0f / 116.0f) / 7.787f;
    };
    
    float x = invF(fx) * 0.95047f;
    float y = invF(fy) * 1.00000f;
    float z = invF(fz) * 1.08883f;
    
    // XYZ to RGB
    float r = x * 3.2404542f - y * 1.5371385f - z * 0.4985314f;
    float g = -x * 0.9692660f + y * 1.8760108f + z * 0.0415560f;
    float b_val = x * 0.0556434f - y * 0.2040259f + z * 1.0572252f;
    
    // Apply inverse gamma correction
    auto invGamma = [](float val) {
        return (val > 0.0031308f) ? 
               1.055f * std::pow(val, 1.0f / 2.4f) - 0.055f : 
               12.92f * val;
    };
    
    r = invGamma(r) * 255.0f;
    g = invGamma(g) * 255.0f;
    b_val = invGamma(b_val) * 255.0f;
    
    // Clamp to valid range
    r = std::clamp(r, 0.0f, 255.0f);
    g = std::clamp(g, 0.0f, 255.0f);
    b_val = std::clamp(b_val, 0.0f, 255.0f);
    
    return Vec3(r, g, b_val);
}

// K-nearest neighbors search with k neighbors
void query_knn_k(
    const PointCloud& points,
    const PointCloud& queries,
    int k,
    std::vector<std::vector<int>>& indices,
    std::vector<std::vector<float>>& distances)
{
    indices.resize(queries.size());
    distances.resize(queries.size());
    
    for (size_t qi = 0; qi < queries.size(); ++qi) {
        std::vector<std::pair<float, int>> dist_idx_pairs;
        
        for (size_t pi = 0; pi < points.size(); ++pi) {
            if (pi == qi) continue; // Skip self
            Vec3 diff = queries[qi] - points[pi];
            float dist = std::sqrt(length2(diff));
            dist_idx_pairs.push_back({dist, static_cast<int>(pi)});
        }
        
        // Sort and take k nearest
        std::partial_sort(dist_idx_pairs.begin(), 
                         dist_idx_pairs.begin() + std::min(k, (int)dist_idx_pairs.size()),
                         dist_idx_pairs.end());
        
        indices[qi].clear();
        distances[qi].clear();
        for (int i = 0; i < std::min(k, (int)dist_idx_pairs.size()); ++i) {
            indices[qi].push_back(dist_idx_pairs[i].second);
            distances[qi].push_back(dist_idx_pairs[i].first);
        }
    }
}

// Bilateral filter for color smoothing
Vec3 bilateralFilterPoint(
    const Vec3& targetColor,
    const std::vector<Vec3>& neighborColors,
    const std::vector<float>& spatialDistances,
    float spatialSigma = 0.01f,
    float colorSigma = 10.0f)
{
    if (neighborColors.empty()) {
        return targetColor;
    }
    
    Vec3 filteredColor(0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;
    
    for (size_t i = 0; i < neighborColors.size(); ++i) {
        // Spatial weight (Gaussian based on distance)
        float spatialWeight = std::exp(-spatialDistances[i] * spatialDistances[i] / 
                                      (2.0f * spatialSigma * spatialSigma));
        
        // Color weight (Gaussian based on color difference)
        Vec3 colorDiff = targetColor - neighborColors[i];
        float colorDist = std::sqrt(length2(colorDiff));
        float colorWeight = std::exp(-colorDist * colorDist / 
                                    (2.0f * colorSigma * colorSigma));
        
        float weight = spatialWeight * colorWeight;
        filteredColor = filteredColor + neighborColors[i] * weight;
        totalWeight += weight;
    }
    
    if (totalWeight > 0.0f) {
        filteredColor = filteredColor * (1.0f / totalWeight);
    } else {
        filteredColor = targetColor;
    }
    
    return filteredColor;
}

// Main beautification function
void BeautifyGaussians(DataLoader::InputInitStruct& inputGaussian, std::string scenePath)
{
    LOGW("Beautify Gaussians begins");
    // Check if head_bbox is valid
    if (inputGaussian.head_bboxes.size() != 6) {
        std::cerr << "Warning: Invalid head_bbox, processing all points" << std::endl;
    }
    
    // Step 1: Filter points within head_bbox and identify skin points
    std::vector<size_t> skin_indices;
    PointCloud skin_positions;
    std::vector<Vec3> skin_colors;
    
    for (size_t i = 0; i < inputGaussian.positions.size(); ++i) {
        bool inHeadBBox = (inputGaussian.head_bboxes.size() == 6) ? 
                         isPointInBBox(inputGaussian.positions[i], inputGaussian.head_bboxes) : 
                         true;
        
        if (inHeadBBox) {
            Vec3 rgb = extractRGBFromData(inputGaussian.colors[i]);
            if (isSkinColor(rgb)) {
                skin_indices.push_back(i);
                skin_positions.push_back(inputGaussian.positions[i]);
                skin_colors.push_back(rgb);
            }
        }
    }
    
    LOGI("Found %{public}zu skin points out of %{public}zu total points", skin_indices.size(), inputGaussian.positions.size());
    
    if (skin_indices.empty()) {
        LOGE("Warning: No skin points detected");
        return;
    }
    
    // Step 2: Find k-nearest neighbors for each skin point
    const int k = 10; // Number of neighbors
    std::vector<std::vector<int>> knn_indices;
    std::vector<std::vector<float>> knn_distances;
    LOGI("Querying KNN");
    query_knn_k(skin_positions, skin_positions, k, knn_indices, knn_distances);
    LOGI("Finished!");
    
    // Step 3: Apply bilateral filter for smoothing
    std::vector<Vec3> smoothed_colors(skin_colors.size());
    
    for (size_t i = 0; i < skin_colors.size(); ++i) {
        std::vector<Vec3> neighbor_colors;
        for (int idx : knn_indices[i]) {
            neighbor_colors.push_back(skin_colors[idx]);
        }
        
        smoothed_colors[i] = bilateralFilterPoint(
            skin_colors[i], 
            neighbor_colors, 
            knn_distances[i],
            0.1f,  // spatial sigma
            30.0f   // color sigma
        );
    }
    
    // Step 4: Convert to LAB and apply whitening
    std::vector<Vec3> beautified_colors(smoothed_colors.size());

    for (size_t i = 0; i < smoothed_colors.size(); ++i) {
        Vec3 lab = rgbToLAB(smoothed_colors[i]);

        // Whitening: increase L (lightness) and slightly adjust a,b
        lab.x = std::min(100.0f, lab.x * 1.08f + 5.0f);  // Increase lightness
        lab.y = lab.y * 0.95f;  // Slightly reduce redness
        lab.z = lab.z * 0.98f;  // Slightly reduce yellowness

        beautified_colors[i] = labToRGB(lab);
    }
    
    // Step 5: Update the original colors in inputGaussian
    for (size_t i = 0; i < skin_indices.size(); ++i) {
        setRGBToData(inputGaussian.colors[skin_indices[i]], smoothed_colors[i]);
    }
    
    // Step 6: Save the beautified gaussians
    std::filesystem::create_directories(scenePath + "/beautified_gaussians");
    std::string saveProcessedGSFile = scenePath + "/beautified_gaussians";
    DataLoader::saveToPLY_tinyply(saveProcessedGSFile, inputGaussian);
    
    std::cout << "Beautified gaussians saved to: " << saveProcessedGSFile << std::endl;
}