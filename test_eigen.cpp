#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <math.h>

using namespace std;

// OUTER PRODUCTS ARE PARALELIZED
// MM MULTIPLICATION IS PARALLELIZED
// MV MULTIPLICATION IS !!!NOT!!! PARALLELIZED
int main() {

    Eigen::setNbThreads(1); 
    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();
    float duration = chrono::duration<float>(end_time - start_time).count();

    Eigen::MatrixXd M = Eigen::MatrixXd::Random(1000, 1000);
    Eigen::MatrixXd M2 = Eigen::MatrixXd::Random(1000, 1000);


    Eigen::MatrixXd v = Eigen::MatrixXd::Random(1000, 1);
    Eigen::MatrixXd v2 = Eigen::MatrixXd::Random(1e7, 1);


    // transpose experiments
    double T_averages[4] = {0,0,0,0};
    for (int i = 0; i < 4; i++) {
        Eigen::setNbThreads(static_cast<int>(exp2(i))); 
        for (int j = 0; j < 3; j++) {

            start_time = chrono::high_resolution_clock::now();
            // if ((M * M2)(0,0) == 0) {
            //     cout << "weird" << endl;
            // }
            M.transpose().eval();
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            T_averages[i] += duration;
        }
        T_averages[i] /= 3;
        M2 = M * M2;
    }   

    cout << "Results for transpose: [";
    for (int i = 0; i < 4; i++) {
        cout << T_averages[i] << ",";
    }
    cout << "] \n";

    // // MM experiments
    // double MM_averages[4] = {0,0,0,0};
    // for (int i = 0; i < 4; i++) {
    //     Eigen::setNbThreads(static_cast<int>(exp2(i))); 
    //     for (int j = 0; j < 3; j++) {

    //         start_time = chrono::high_resolution_clock::now();
    //         if ((M * M2)(0,0) == 0) {
    //             cout << "weird" << endl;
    //         }
    //         end_time = chrono::high_resolution_clock::now();
    //         duration = chrono::duration<float>(end_time - start_time).count();
    //         MM_averages[i] += duration;
    //     }
    //     MM_averages[i] /= 3;
    // }

    // cout << "Results for MM: [";
    // for (int i = 0; i < 4; i++) {
    //     cout << MM_averages[i] << ",";
    // }
    // cout << "] \n";



// // MV experiments
//     double MV_averages[4] = {0,0,0,0};
//     for (int i = 0; i < 4; i++) {
//         Eigen::setNbThreads(static_cast<int>(exp2(i))); 
//         for (int j = 0; j < 3; j++) {

//             start_time = chrono::high_resolution_clock::now();
//             if ((M * v)(0,0) == 0) {
//                 cout << "weird" << endl;
//             }
//             end_time = chrono::high_resolution_clock::now();
//             duration = chrono::duration<float>(end_time - start_time).count();
//             MV_averages[i] += duration;
//         }
//         MV_averages[i] /= 3;
//     }

//     cout << "Results for MV: [";
//     for (int i = 0; i < 4; i++) {
//         cout << MV_averages[i] << ",";
//     }
//     cout << "] \n";



    // // Outer product experiments
    // double OP_averages[4] = {0,0,0,0};
    // for (int i = 0; i < 4; i++) {
    //     Eigen::setNbThreads(static_cast<int>(exp2(i))); 
    //     for (int j = 0; j < 3; j++) {

    //         start_time = chrono::high_resolution_clock::now();
    //         if ((v * v.transpose())(0,0) == 0) {
    //             cout << "weird" << endl;
    //         }
    //         end_time = chrono::high_resolution_clock::now();
    //         duration = chrono::duration<float>(end_time - start_time).count();
    //         OP_averages[i] += duration;
    //     }
    //     OP_averages[i] /= 3;
    // }

    // cout << "Results for OP: [";
    // for (int i = 0; i < 4; i++) {
    //     cout << OP_averages[i] << ",";
    // }
    // cout << "] \n";






//     // Matrix Subtraction experiments
//     double MS_averages[4] = {0,0,0,0};
//     for (int i = 0; i < 4; i++) {
//         Eigen::setNbThreads(static_cast<int>(exp2(i))); 
//         for (int j = 0; j < 3; j++) {

//             start_time = chrono::high_resolution_clock::now();
//             M = M - M2;
//             end_time = chrono::high_resolution_clock::now();
//             duration = chrono::duration<float>(end_time - start_time).count();
//             MS_averages[i] += duration;
//         }
//         MS_averages[i] /= 3;
//     }

//     cout << "Results for MS: [";
//     for (int i = 0; i < 4; i++) {
//         cout << MS_averages[i] << ",";
//     }
//     cout << "] \n";

// Inner Product experiments
    // double IP_averages[4] = {0,0,0,0};
    // for (int i = 0; i < 4; i++) {
    //     Eigen::setNbThreads(static_cast<int>(exp2(i))); 
    //     for (int j = 0; j < 3; j++) {

    //         start_time = chrono::high_resolution_clock::now();
    //         if ((v2.transpose()*v2)(0,0) == 0) {
    //             cout << "weird" << endl;
    //         }
    //         end_time = chrono::high_resolution_clock::now();
    //         duration = chrono::duration<float>(end_time - start_time).count();
    //         IP_averages[i] += duration;
    //     }
    //     IP_averages[i] /= 3;
    // }

    // cout << "Results for IP: [";
    // for (int i = 0; i < 4; i++) {
    //     cout << IP_averages[i] << ",";
    // }
    // cout << "] \n";

    // return 0;
}