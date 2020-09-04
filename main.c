#include <stdio.h>
#include <math.h>
#include <time.h>

#include "ransac_tdoa.h"
// #include "ransac_tdoa_gsl.h"

#define TEST_LANDMARK_NUM 6

int main() {
    Position2D landmarks[TEST_LANDMARK_NUM];
    landmarks[0].x = 0;  // testing normal data
    landmarks[0].y = 0;
    landmarks[1].x = 0;
    landmarks[1].y = 100;
    landmarks[2].x = 100;
    landmarks[2].y = 100;
    landmarks[3].x = 100;
    landmarks[3].y = 0;
    landmarks[4].x = 200;
    landmarks[4].y = 200;
    landmarks[5].x = 300;
    landmarks[5].y = 400;    
    // landmarks[0].x = 0; // testing abnormal data, in which all landmarks in one line.
    // landmarks[0].y = 0;
    // landmarks[1].x = 100;
    // landmarks[1].y = 100;
    // landmarks[2].x = 200;
    // landmarks[2].y = 200;
    // landmarks[3].x = 300;
    // landmarks[3].y = 303;
    // landmarks[4].x = 400;
    // landmarks[4].y = 400;
    // landmarks[5].x = 500;
    // landmarks[5].y = 500;
    double difference_of_distances[TEST_LANDMARK_NUM]; // difference of distance comparing with landmark 0
    srand(1);
    double groundtruth_x = rand() / (double)RAND_MAX * 100;
    double groundtruth_y = rand() / (double)RAND_MAX * 100;
    difference_of_distances[0] = 0;
    double std_dev = 0.01;
    double distance_0 = sqrt(pow(groundtruth_x - landmarks[0].x, 2) + pow(groundtruth_y - landmarks[0].y, 2)) + (rand() / (double)RAND_MAX * std_dev - std_dev/2.0);
    // double distance_0 = sqrt(pow(groundtruth_x - landmarks[0].x, 2) + pow(groundtruth_y - landmarks[0].y, 2)); 
    for (int i = 1; i < TEST_LANDMARK_NUM; i++) {
        difference_of_distances[i] = sqrt(pow(groundtruth_x - landmarks[i].x, 2) + pow(groundtruth_y - landmarks[i].y, 2)) + (rand() / (double)RAND_MAX * std_dev - std_dev/2.0) -  distance_0;
        // difference_of_distances[i] = difference_of_distances[i] + 1000;
    }
    for (int i = 0; i < TEST_LANDMARK_NUM; i++) {
        difference_of_distances[i] = difference_of_distances[i] + 1000;
    }
    
    // difference_of_distances[4] = difference_of_distances[4] + 0.05; // uncomment this line to test that ransac can ignore the noise data.
    double estimated_x, estimated_y;
    int effective_landmark_mask[TEST_LANDMARK_NUM];
    for (int i = 0; i < TEST_LANDMARK_NUM; i++) {
        effective_landmark_mask[i] = 1;
    }
    // effective_landmark_mask[2] = 0; // uncomment this line to test that 'effective_landmark_mask' works
    clock_t start, finish;
    Position2D pos;
    start = clock();
    int success = tdoa_ransac(landmarks, difference_of_distances, effective_landmark_mask, TEST_LANDMARK_NUM, std_dev,
        &pos);
    finish = clock();
    
    printf("groundtruth_x:%lf, groundtruth_y:%lf\n", groundtruth_x, groundtruth_y);
    printf("final result: success:%d, estimated_x:%lf, estimated_y:%lf, time cost:%lfs.\n", success, pos.x, pos.y, (double)(finish - start) / CLOCKS_PER_SEC);
    
    return 0;
}