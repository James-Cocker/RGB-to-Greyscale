#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include <iostream>

using namespace cv;
using namespace std;

__global__ void rgb2grey(unsigned char* in_d, unsigned char* out_d, int NUM_PIXELS) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < NUM_PIXELS) { // boundary conditions
    	int idx 	= threadID * 3;	// For r, g and b

        float red 	= in_d[idx];
        float green = in_d[idx+1];
        float blue 	= in_d[idx+2];

        // Formula to turn rgb to greyscale.
        out_d[threadID] = 0.299 * red + 0.587 * green + 0.114 * blue;
    }
}

int main(int argc, char** argv) {
    // Read the RGB image file. Either use the full path
    // or relative to your home directory
    Mat rgb_img_h = imread("/user/HS401/jc02842/Downloads/stag-hill-lake-media.jpg");

    // Check for failure
    if (rgb_img_h.empty()) {
        cout << "Could not open or find the image" << endl;
        cin.get(); // wait for any key press
        return -1;
    }

    String windowName = "Stag Hill"; // Name of the window
    namedWindow(windowName); // Create a window
    imshow(windowName, rgb_img_h); // Show our image inside the created window
    waitKey(0); // Wait for any keystroke in the window

    // Create an empty image on CPU that will hold our grey-scale output.
    // This will have the same dimensions of your original RGB
    // However, instead of having three channels of unsigned char of 8 bits each (CV_8UC3)
    // it will have a single channel of unsigned char (CV_8UC1)
    // We use the rows and columns from rgb_img_h to make sure we have the same dimensions
    int height = rgb_img_h.rows;
    int width = rgb_img_h.cols;
    Mat grey_img_h(height, width, CV_8UC1);

    // Allocate memory on GPU for both RGB (three channels) input data
    // and grey-scale (single channel) output data
    unsigned char* rgb_data_d = NULL;
    unsigned char* grey_data_d = NULL;
    cudaError_t err;

    err = cudaMalloc(&rgb_data_d, height * width * 3 * sizeof(unsigned char)); // three channels
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

    err = cudaMalloc(&grey_data_d, height * width * sizeof(unsigned char)); // one channel
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

    // The grey_img_h image we created above is an OpenCV object of type Mat.
    // The actual pixels' values (data) inside this object can be accessed
    // using rgb_img_h.data, which is a pointer of type unsigned char*.
    // Copy RGB data from host (CPU) to device (GPU) - three channels
    err = cudaMemcpy(rgb_data_d, rgb_img_h.data, height * width * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

    // Launch 2D kernel
    int threadsPerBlock = 256;
    int numBlocks = (height * width + threadsPerBlock - 1) / threadsPerBlock;
    rgb2grey<<<numBlocks, threadsPerBlock>>>(rgb_data_d, grey_data_d, width * height);

    // Copy results back to CPU - one channel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

    // Copy results back to CPU - one channel
	err = cudaMemcpy(grey_img_h.data, grey_data_d, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}

    // Display results
    imshow(windowName, grey_img_h); // Show our image inside the created window
    waitKey(0); // Wait for any keystroke in the window
    destroyWindow(windowName); // destroy the created window

    // Free memory
    cudaFree(rgb_data_d);
    cudaFree(grey_data_d);

    // Free memory
    return 0;
}
