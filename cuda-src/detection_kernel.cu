#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

const float INPUT_WIDTH = 640;
const float INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

const float FONT_SCALE = 0.7;
const int FONT_THICKNESS = 2;


// CUDA kernel for object detection
__global__ void cudaObjectDetectionKernel(uchar3* inputImage, uchar3* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        outputImage[y * width + x] = inputImage[y * width + x];

    }
}

void processVideo() {
    cv::VideoCapture cap("./samples/video.mp4");
    if (!cap.isOpened()) {
        printf("Error opening video file\n");
        return;
    }
    cv::namedWindow("Video", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            printf("End of video\n");
            break;
        }

        uchar3* d_inputImage, * d_outputImage;
        cudaMalloc(&d_inputImage, frame.rows * frame.cols * sizeof(uchar3));
        cudaMalloc(&d_outputImage, frame.rows * frame.cols * sizeof(uchar3));

        cudaMemcpy(d_inputImage, frame.data, frame.rows * frame.cols * sizeof(uchar3), cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((frame.cols + blockSize.x - 1) / blockSize.x, (frame.rows + blockSize.y - 1) / blockSize.y);

        cudaObjectDetectionKernel << <gridSize, blockSize >> > (d_inputImage, d_outputImage, frame.cols, frame.rows);

        cudaMemcpy(frame.data, d_outputImage, frame.rows * frame.cols * sizeof(uchar3), cudaMemcpyDeviceToHost);

        cudaFree(d_inputImage);
        cudaFree(d_outputImage);

        cv::imshow("Video", frame);

        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

int main() {
    processVideo();
    return 0;
}
