#include "utils.slang"

// Test function for Get3DPoint
float3 testGet3DPoint()
{
    float3 point;

    // Test case 1: Basic case with standard camera parameters
    if (false)
    {
        float4 cameraK = float4(500.0, 500.0, 320.0, 240.0); // fx, fy, cx, cy
        int2 pixel = int2(320, 240); // Center of image
        float depth = 2.0;

        point = Get3DPoint(cameraK, pixel, depth);
        //printf("Test 1 - Center pixel:");
        //printf("  Expected: (0.0, 0.0, 2.0)");
        //printf("  Actual: (%.3f, %.3f, %.3f)", point.x, point.y, point.z);
        //assert(abs(point.x) < 0.001 && abs(point.y) < 0.001 && abs(point.z - 2.0) < 0.001);
    }

    // Test case 2: Corner pixel with depth 1.0
    if (false)
    {
        float4 cameraK = float4(500.0, 500.0, 320.0, 240.0);
        int2 pixel = int2(0, 0); // Top-left corner
        float depth = 1.0;

        point = Get3DPoint(cameraK, pixel, depth);
        //printf("\nTest 2 - Corner pixel:");
        //printf("  Expected: (-0.64, -0.48, 1.0)");
        //printf("  Actual: (%.3f, %.3f, %.3f)", point.x, point.y, point.z);
        //assert(abs(point.x - (-0.64)) < 0.001 && abs(point.y - (-0.48)) < 0.001 && abs(point.z - 1.0) < 0.001);
    }

    // Test case 3: Different focal lengths in x and y
    if (false)
    {
        float4 cameraK = float4(600.0, 400.0, 320.0, 240.0);
        int2 pixel = int2(420, 340);
        float depth = 3.0;

        point = Get3DPoint(cameraK, pixel, depth);
        //printf("\nTest 3 - Different focal lengths:");
        //printf("  Expected: (0.5, 0.75, 3.0)");
        //printf("  Actual: (%.3f, %.3f, %.3f)", point.x, point.y, point.z);
        //assert(abs(point.x - 0.5) < 0.001 && abs(point.y - 0.75) < 0.001 && abs(point.z - 3.0) < 0.001);
    }

    // Test case 4: Zero depth (should return zero vector)
    if (true)
    {
        float4 cameraK = float4(500.0, 500.0, 320.0, 240.0);
        int2 pixel = int2(100, 100);
        float depth = 0.0;

        point = Get3DPoint(cameraK, pixel, depth);
        //printf("\nTest 4 - Zero depth:");
        //printf("  Expected: (0.0, 0.0, 0.0)");
        //printf("  Actual: (%.3f, %.3f, %.3f)", point.x, point.y, point.z);
        //assert(abs(point.x) < 0.001 && abs(point.y) < 0.001 && abs(point.z) < 0.001);
    }

    // Test case 5: Negative depth (valid if your application allows it)
    if (false)
    {
        float4 cameraK = float4(500.0, 500.0, 320.0, 240.0);
        int2 pixel = int2(400, 300);
        float depth = -1.5;

        point = Get3DPoint(cameraK, pixel, depth);
        //printf("\nTest 5 - Negative depth:");
        //printf("  Expected: (0.24, 0.18, -1.5)");
        //printf("  Actual: (%.3f, %.3f, %.3f)", point.x, point.y, point.z);
        //assert(abs(point.x - 0.24) < 0.001 && abs(point.y - 0.18) < 0.001 && abs(point.z - (-1.5)) < 0.001);
    }

    //printf("\nAll tests passed!");

    return point;
}

void testHomoGraphy()
{

    Camera  camera1;
    camera1.R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    camera1.t = {0.0, 0.0, 0.0};
    camera1.K = {600, 400, 320, 240};

    Camera  camera2;
    float theta = 0.01745329251 * 30;
    camera2.R = {cos(theta), 0.0, sin(theta), 0.0, 1.0, 0.0, -sin(theta), 0.0, cos(theta)};
    camera2.t = {0.0, 0.0, 0.0};
    //camera2.R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    //camera2.t = {0.1, 0.0, 0.0};
    camera2.K = {600, 400, 320, 240};

    float H[9];
    ComputeHomography(camera1, camera2, {0, 0, 1, -2}, H);

    //Homography Matrix (with rotation):
    // [[ 5.99358737e-01  0.00000000e+00  3.85333333e+02]
    // [-2.00000000e-01  1.00000000e+00  3.18460969e+01]
    // [-8.33333333e-04  0.00000000e+00  1.13269207e+00]]
    //
    //Homography Matrix:
    //  [[ 1.  0. 30.]
    //  [ 0.  1.  0.]
    //  [ 0.  0.  1.]]
}