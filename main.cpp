#include "randomized_QR.h"
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <omp.h>


using namespace std;

void parallel_Mv_multiplication_tests(randomized_QR test_var);
void parallel_inner_product_tests(randomized_QR test_var);
void parallel_MM_subtraction_tests(randomized_QR test_var);

void orthonormalize_columns_tests(randomized_QR test_var);
void orthonormalize_columns_householder_tests(randomized_QR test_var);

void get_P_functionality_test(randomized_QR test_var);

void QR_decomposition_standard_tests(randomized_QR test_var);
void QR_decomposition_projection_tests(randomized_QR test_var);

void Householder_QR_vtest(randomized_QR test_var);
void Householder_QR_run_test(randomized_QR test_var);
void Householder_QR_decomposition_projection_tests(randomized_QR test_var);


//                                      EXPERIMENTS

void HQR_p_expts(randomized_QR test_var);
void Projection_HQR_p_expts(randomized_QR test_var);

void Projection_GS_p_expts(randomized_QR test_var);
void GS_p_expts(randomized_QR test_var);

void strong_scaling_expts(randomized_QR test_var);
void weak_scaling_expts();
void k_scaling_expts();
void stability_expts();




int main() {
    
    size_t n = 400;
    size_t k = 50;
    randomized_QR test_var = randomized_QR(n,k);

    // auto start = chrono::high_resolution_clock::now();
    // // test_var.orthonormalizeColumns(test_var.A);
    // // test_var.QR_decomposition(test_var.A);
    // test_var.Householder_QR(test_var.A);
    // // test_var.Householder_QR_with_projections();
    // auto end = chrono::high_resolution_clock::now();
    // cout << chrono::duration<float>(end - start).count() << endl;

    // strong_scaling_expts(test_var);
    // weak_scaling_expts();
    // k_scaling_expts();
    stability_expts();



    // HQR_p_expts(test_var);
    // Projection_HQR_p_expts(test_var);

    // Projection_GS_p_expts(test_var);
    // GS_p_expts(test_var);

    //                                         UNIT TESTS
    // parallel_inner_product_tests(test_var);
    // parallel_MM_subtraction_tests(test_var);
    // parallel_Mv_multiplication_tests(test_var);
    // get_P_functionality_test(test_var);

    // orthonormalize_columns_tests(test_var);
    // orthonormalize_columns_householder_tests(test_var);


    // QR_decomposition_standard_tests(test_var);
    // QR_decomposition_projection_tests(test_var);

    // // Householder_QR_vtest(test_var);
    // Householder_QR_run_test(test_var);
    // Householder_QR_decomposition_projection_tests(test_var);

    // test_var.QR_decomposition(test_var.A);
    // test_var.QR_with_projections();

    // cout << "A: " << test_var.A << endl;
    // cout << "QR: " << test_var.Q * test_var.R << endl;

    // cout << "orthogonality error: "  << (MatrixXd::Identity(k,k) - test_var.Q.transpose() * test_var.Q).norm() << endl;
    // cout << "orthogonality error: "  << (MatrixXd::Identity(n,n) - test_var.Q.transpose() * test_var.Q).norm() << endl;
    // cout << "Random Projections QR error: " << (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm() << endl;


    return 0;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                         EXPERIMENTS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HQR_p_expts(randomized_QR test_var) {
    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();
    float duration;
    // MM experiments
    double HQR_averages[4] = {0,0,0,0};
    // 1,2,4,8
    for (int i = 0; i < 10; i++) {
        omp_set_num_threads(i+1);
        // Eigen::setNbThreads(static_cast<int>(exp2(i))); 
        Eigen::setNbThreads(i+1); 
        for (int j = 0; j < 3; j++) {

            start_time = chrono::high_resolution_clock::now();
            test_var.Householder_QR(test_var.A);
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            HQR_averages[i] += duration;
        }
        HQR_averages[i] /= 3;
    }
    
    cout << "Results for HQR (effect of p on time): [";
    for (int i = 0; i < 10; i++) {
        cout << HQR_averages[i] << ",";
    }
    cout << "] \n";
}

void Projection_HQR_p_expts(randomized_QR test_var) {
    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();
    float duration;
    // MM experiments
    double Projection_HQR_averages[4] = {0,0,0,0};
    // 1,2,4,8
    for (int i = 0; i < 4; i++) {
        omp_set_num_threads(i);
        Eigen::setNbThreads(static_cast<int>(exp2(i))); 
        for (int j = 0; j < 3; j++) {

            start_time = chrono::high_resolution_clock::now();
            test_var.Householder_QR_with_projections();
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            Projection_HQR_averages[i] += duration;
        }
        Projection_HQR_averages[i] /= 3;
    }
    
    cout << "Results for Projection HQR (effect of p on time): [";
    for (int i = 0; i < 4; i++) {
        cout << Projection_HQR_averages[i] << ",";
    }
    cout << "] \n";
}

void GS_p_expts(randomized_QR test_var) {
    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();
    float duration;
    // MM experiments
    double GS_averages[4] = {0,0,0,0};
    // 1,2,4,8
    for (int i = 0; i < 4; i++) {
        omp_set_num_threads(i);
        Eigen::setNbThreads(static_cast<int>(exp2(i))); 
        for (int j = 0; j < 3; j++) {

            start_time = chrono::high_resolution_clock::now();
            test_var.QR_decomposition(test_var.A);
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            GS_averages[i] += duration;
        }
        GS_averages[i] /= 3;
    }
    
    cout << "Results for GS (effect of p on time): [";
    for (int i = 0; i < 4; i++) {
        cout << GS_averages[i] << ",";
    }
    cout << "] \n";
}

void Projection_GS_p_expts(randomized_QR test_var) {
    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();
    float duration;
    // MM experiments
    double PGS_averages[4] = {0,0,0,0};
    // 1,2,4,8
    for (int i = 0; i < 4; i++) {
        omp_set_num_threads(i);
        Eigen::setNbThreads(static_cast<int>(exp2(i))); 
        for (int j = 0; j < 3; j++) {

            start_time = chrono::high_resolution_clock::now();
            // test_var.QR_decomposition(test_var.A);
            test_var.QR_with_projections();
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            PGS_averages[i] += duration;
        }
        PGS_averages[i] /= 3;
    }
    
    cout << "Results for projections GS (effect of p on time): [";
    for (int i = 0; i < 4; i++) {
        cout << PGS_averages[i] << ",";
    }
    cout << "] \n";
}

void strong_scaling_expts(randomized_QR test_var) {
    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();
    float duration;

    // MM experiments
    double HQR_averages[10] = {0,0,0,0,0,0,0,0,0,0};
    double Projection_HQR_averages[10] = {0,0,0,0,0,0,0,0,0,0};
    double GS_averages[10] = {0,0,0,0,0,0,0,0,0,0};
    double PGS_averages[10] = {0,0,0,0,0,0,0,0,0,0};

    // set the number of threads between 1 and 10
    for (int i = 0; i < 10; i++) {
        omp_set_num_threads(i+1);
        Eigen::setNbThreads(i+1); 
        // for each thread number, average the time for reliability
        for (int j = 0; j < 3; j++) {

            start_time = chrono::high_resolution_clock::now();
            test_var.Householder_QR(test_var.A);
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            HQR_averages[i] += duration;

            start_time = chrono::high_resolution_clock::now();
            test_var.Householder_QR_with_projections();
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            Projection_HQR_averages[i] += duration;

            start_time = chrono::high_resolution_clock::now();
            test_var.QR_decomposition(test_var.A);
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            GS_averages[i] += duration;

            start_time = chrono::high_resolution_clock::now();
            test_var.QR_with_projections();
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            PGS_averages[i] += duration;
        }
        HQR_averages[i] /= 3;
        Projection_HQR_averages[i] /= 3;
        GS_averages[i] /= 3;
        PGS_averages[i] /= 3;
    }
    
    cout << "Results for HQR (effect of p on time): [";
    for (int i = 0; i < 10; i++) {
        cout << HQR_averages[i] << ",";
    }
    cout << "] \n";

    cout << "Results for Projection HQR (effect of p on time): [";
    for (int i = 0; i < 10; i++) {
        cout << Projection_HQR_averages[i] << ",";
    }
    cout << "] \n";

    cout << "Results for GS (effect of p on time): [";
    for (int i = 0; i < 10; i++) {
        cout << GS_averages[i] << ",";
    }
    cout << "] \n";

    cout << "Results for projections GS (effect of p on time): [";
    for (int i = 0; i < 10; i++) {
        cout << PGS_averages[i] << ",";
    }
    cout << "] \n";
}

// problem size scales with n^3 for standard decomps and k*n^2 for projections
// I don't change k, so multiply p by 2 each time while hitting n with 2^1/3 and 2^1/2 respectively
void weak_scaling_expts() {
    // initialize variables
    size_t n = 300;
    size_t k = 200;
    int num_threads = 1;
    randomized_QR test_var = randomized_QR(n,k);

    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();
    float duration;

    // where the data will be stored
    double HQR_averages[4] = {0,0,0,0};
    double Projection_HQR_averages[4] = {0,0,0,0};
    double GS_averages[4] = {0,0,0,0};
    double PGS_averages[4] = {0,0,0,0};

    // set the number of threads in 1,2,4,8 and have n scale accordingly
    for (int i = 0; i < 4; i++) {
        num_threads = static_cast<int>(exp2(i));
        omp_set_num_threads(num_threads);
        Eigen::setNbThreads(num_threads); 

        // scale the problem size for the cubic growth case
        test_var = randomized_QR(floor(exp2(static_cast<double>(i)/3)*n),k);
        for (int j = 0; j < 3; j++) {
            start_time = chrono::high_resolution_clock::now();
            test_var.Householder_QR(test_var.A);
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            HQR_averages[i] += duration;

            start_time = chrono::high_resolution_clock::now();
            test_var.QR_decomposition(test_var.A);
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            GS_averages[i] += duration;
        }

        // scale the problem size for the quadratic growth case
        test_var = randomized_QR(floor(exp2(static_cast<double>(i)/2)*n),k);  
        for (int j = 0; j < 3; j++) {
            start_time = chrono::high_resolution_clock::now();
            test_var.Householder_QR_with_projections();
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            Projection_HQR_averages[i] += duration;

            start_time = chrono::high_resolution_clock::now();
            test_var.QR_with_projections();
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            PGS_averages[i] += duration;
        }
        HQR_averages[i] /= 3;
        Projection_HQR_averages[i] /= 3;
        GS_averages[i] /= 3;
        PGS_averages[i] /= 3;
    }
    
    std::cout << "Results for HQR (weak scaling): [";
    for (int i = 0; i < 4; i++) {
        std::cout << HQR_averages[i] << ",";
    }
    std::cout << "] \n";

    std::cout << "Results for Projection HQR (weak scaling): [";
    for (int i = 0; i < 4; i++) {
        std::cout << Projection_HQR_averages[i] << ",";
    }
    std::cout << "] \n";

    std::cout << "Results for GS (weak scaling): [";
    for (int i = 0; i < 4; i++) {
        std::cout << GS_averages[i] << ",";
    }
    std::cout << "] \n";

    std::cout << "Results for projections GS (weak scaling): [";
    for (int i = 0; i < 4; i++) {
        std::cout << PGS_averages[i] << ",";
    }
    std::cout << "] \n";
}


void k_scaling_expts() {
    size_t n = 600;
    size_t k = 10;
    int num_threads = 4;

    omp_set_num_threads(num_threads);
    Eigen::setNbThreads(num_threads); 

    randomized_QR test_var = randomized_QR(n,k);

    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();
    float duration;

    // where the data will be stored
    double Projection_HQR_averages[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double PGS_averages[14] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // set the number of threads in 1,2,4,8 and have n scale accordingly
    for (int i = 0; i < 14; i++) {
        cout << i << ", ";
        k = 10 + 30 * i;

        test_var = randomized_QR(n,k);

        for (int j = 0; j < 3; j++) {
            start_time = chrono::high_resolution_clock::now();
            test_var.Householder_QR_with_projections();
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            Projection_HQR_averages[i] += duration;

            start_time = chrono::high_resolution_clock::now();
            test_var.QR_with_projections();
            end_time = chrono::high_resolution_clock::now();
            duration = chrono::duration<float>(end_time - start_time).count();
            PGS_averages[i] += duration;
        }
        Projection_HQR_averages[i] /= 3;
        PGS_averages[i] /= 3;
    }
    

    std::cout << "Results for Projection HQR (k scaling): [";
    for (int i = 0; i < 14; i++) {
        std::cout << Projection_HQR_averages[i] << ",";
    }
    std::cout << "] \n";


    std::cout << "Results for projections GS (k): [";
    for (int i = 0; i < 14; i++) {
        std::cout << PGS_averages[i] << ",";
    }
    std::cout << "] \n";
}

void stability_expts() {
    randomized_QR test_var(300,100);

    // MM experiments
    double HQR_averages[3] = {0,0,0};
    double Projection_HQR_averages[3] = {0,0,0};
    double GS_averages[3] = {0,0,0};
    double PGS_averages[3] = {0,0,0};

    omp_set_num_threads(4);
    Eigen::setNbThreads(4);
    for (int i = 0; i < 3; i++) {
         
        // average the time for reliability
        for (int j = 0; j < 3; j++) {
            test_var = randomized_QR(200 + i * 100,50);

            test_var.Householder_QR(test_var.A);
            HQR_averages[i] += (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm();

            test_var.Householder_QR_with_projections();
            Projection_HQR_averages[i] += (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm();

            test_var.QR_decomposition(test_var.A);
            GS_averages[i] += (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm();

            test_var.QR_with_projections();
            PGS_averages[i] += (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm();
        }
        HQR_averages[i] /= 10;
        Projection_HQR_averages[i] /= 10;
        GS_averages[i] /= 10;
        PGS_averages[i] /= 10;
    }
    
    cout << "Results for HQR (effect of n on error): [";
    for (int i = 0; i < 3; i++) {
        cout << HQR_averages[i] << ",";
    }
    cout << "] \n";

    cout << "Results for Projection HQR (effect of n on error): [";
    for (int i = 0; i < 3; i++) {
        cout << Projection_HQR_averages[i] << ",";
    }
    cout << "] \n";

    cout << "Results for GS (effect of n on error): [";
    for (int i = 0; i < 3; i++) {
        cout << GS_averages[i] << ",";
    }
    cout << "] \n";

    cout << "Results for projections GS (effect of n on error): [";
    for (int i = 0; i < 3; i++) {
        cout << PGS_averages[i] << ",";
    }
    cout << "] \n";
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                          UNIT TESTS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
SUCCEEDS ALL
Expected output:
    5
    11
    17

    The time for my_fun time should decrease substancially with more processors
*/
void parallel_Mv_multiplication_tests(randomized_QR test_var) {
    Eigen::MatrixXd M(3,2);
    Eigen::MatrixXd v(2,1);

    M(0,0) = 1;
    M(0,1) = 2;
    M(1,0) = 3;
    M(1,1) = 4;
    M(2,0) = 5;
    M(2,1) = 6;

    v(0,0) = 1;
    v(1,0) = 2;

    std::cout << test_var.parallel_Mv_multiplication(M,v) << endl;

    Eigen::MatrixXd M2 = Eigen::MatrixXd::Random(1000, 1000);
    Eigen::MatrixXd v2 = Eigen::MatrixXd::Random(1000, 1);

    auto start_time = chrono::high_resolution_clock::now();
    test_var.parallel_Mv_multiplication(M2,v2);
    auto end_time = chrono::high_resolution_clock::now();
    float duration = chrono::duration<float>(end_time - start_time).count();
    std::cout << "my_fun time: " << duration;

    start_time = chrono::high_resolution_clock::now();
    v = M2*v2;
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration<float>(end_time - start_time).count();
    std::cout << " Eigen time: " << duration << endl;
}

/*
SUCCEEDS IN GIVING THE CORRECT VALUE
FAILED IN SCALING WITH PROCESSORS. It has since been modified to be sequential
Expected output:
    3

*/
void parallel_inner_product_tests(randomized_QR test_var) {
    Eigen::MatrixXd v(2,1);
    v(0,0) = 1;
    v(1,0) = 2;

    Eigen::MatrixXd v2(2,1);
    v2(0,0) = 5;
    v2(1,0) = -1;

    std::cout << test_var.parallel_inner_product(v,v2) << endl;


    // Eigen::MatrixXd v3 = Eigen::MatrixXd::Random(10000, 1);
    // auto start = chrono::high_resolution_clock::now();
    // v(0) = test_var.parallel_inner_product(v3,v3);
    // auto end = chrono::high_resolution_clock::now();
    // cout << "IP time: " << chrono::duration<float>(end - start).count() << endl;


}

/*
SUCCEEDS ALL
Expected output:
    1 2
    3 4
    5 6
*/
void parallel_MM_subtraction_tests(randomized_QR test_var) {
    Eigen::MatrixXd M(3,2);
    M(0,0) = 1;
    M(0,1) = 2;
    M(1,0) = 3;
    M(1,1) = 4;
    M(2,0) = 5;
    M(2,1) = 6;

    Eigen::MatrixXd M2 = M*2;
    std::cout << test_var.parallel_MM_subtraction(M2,M) << endl;
}



/*
SUCCEEDS ALL
Expected Output:
 0.169031  0.897085
 0.507093  0.276027
 0.845154 -0.345032
same span
*/
void orthonormalize_columns_tests(randomized_QR test_var) {

    Eigen::MatrixXd M(3,2);
    M(0,0) = 1;
    M(0,1) = 2;
    M(1,0) = 3;
    M(1,1) = 4;
    M(2,0) = 5;
    M(2,1) = 6;

    Eigen::MatrixXd ort_M = test_var.orthonormalizeColumns(M,2);
    cout << ort_M << endl;

    // test the span by looking at the projection matrices for each
    Eigen::MatrixXd og = M * (M.transpose() * M).inverse() * M.transpose();
    Eigen::MatrixXd ort = ort_M * (ort_M.transpose() * ort_M).inverse() * ort_M.transpose();
    if ((og - ort).norm() < 1e-6) {
        cout << "same span" << endl;
    }

}


/*
SUCCEEDS ALL
Expected Output: 
-0.169031  0.897085
-0.507093  0.276026
-0.845154 -0.345033
same span
*/
void orthonormalize_columns_householder_tests(randomized_QR test_var) {

    Eigen::MatrixXd M(3,2);
    M(0,0) = 1;
    M(0,1) = 2;
    M(1,0) = 3;
    M(1,1) = 4;
    M(2,0) = 5;
    M(2,1) = 6;

    Eigen::MatrixXd ort_M = test_var.orthonormalizeColumns_householder(M);
    cout << ort_M << endl;

    // test the span by looking at the projection matrices for each
    Eigen::MatrixXd og = M * (M.transpose() * M).inverse() * M.transpose();
    Eigen::MatrixXd ort = ort_M * (ort_M.transpose() * ort_M).inverse() * ort_M.transpose();
    if ((og - ort).norm() < 1e-6) {
        cout << "same span" << endl;
    }
}

/*
SUCCEEDS ALL
Expected Output: a small quantity (relative error of PPTA vs A)
*/
void get_P_functionality_test(randomized_QR test_var) {
    MatrixXd P = test_var.get_P();

    // test the span by looking at the projection matrices for each
    Eigen::MatrixXd og = test_var.A;
    Eigen::MatrixXd ort = P*P.transpose()*og;
    if ((og - ort).norm() < 1e-2) {
        cout << "same span" << endl;
    }
    // relative error in norm
    cout << (og - ort).norm() / test_var.A.norm() << endl;
}



/*
SUCCEEDS ALL
Expected Output:
Q
0.707107  0.408248  -0.57735
 0.707107 -0.408248   0.57735
        0  0.816497   0.57735
R
     1.41421     0.707107     0.707107
           0      1.22474     0.408248
           0            0       1.1547*/
void QR_decomposition_standard_tests(randomized_QR test_var) {
    // MatrixXd M(3,3);
    // M(0,0) = 1; M(0,1) = 1; M(0,2) = 0;
    // M(1,0) = 1; M(1,1) = 0; M(1,2) = 1;
    // M(2,1) = 0; M(2,1) = 1; M(2,2) = 1;

    // test_var.QR_decomposition(M);

    // std::cout << test_var.Q << endl;
    // std::cout << test_var.R << endl;


        // test_var.QR_with_projections();

    test_var.QR_decomposition(test_var.A);

    // cout << "A: " << test_var.A << endl;
    // cout << "QR: " << test_var.Q * test_var.R << endl;

    cout << "Standard QR error: " << (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm() << endl;
}


/*
FAILS FOR SOME k (instability)
Expected Output: a small quantity (relative error of QR vs A)
*/
void QR_decomposition_projection_tests(randomized_QR test_var) {

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Fixed Test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // // Note, k must be 2 for this test.  Also, Hard-code A as M
    // MatrixXd M(3,3);
    // M(0,0) = 1; M(0,1) = 1; M(0,2) = 2;
    // M(1,0) = 1; M(1,1) = 0; M(1,2) = 1;
    // M(2,1) = 0; M(2,1) = 1; M(2,2) = 1;

    // cout << "QR: " << test_var.Q * test_var.R << endl;

    // cout << "Random Projections QR error: " << (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm() << endl;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Random Test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    test_var.QR_with_projections();

    // cout << "A: " << test_var.A << endl;
    // cout << "QR: " << test_var.Q; //test_var.Q * test_var.R << endl;

    cout << "Random Projections QR error: " << (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm() << endl;
}

/*
SUCCEEDS ALL
Expected Output: a small quantity (relative error of QR vs A)
*/
void Householder_QR_decomposition_projection_tests(randomized_QR test_var) {

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Fixed Test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // // Note, k must be 2 for this test.  Also, Hard-code A as M
    // MatrixXd M(3,3);
    // M(0,0) = 1; M(0,1) = 1; M(0,2) = 2;
    // M(1,0) = 1; M(1,1) = 0; M(1,2) = 1;
    // M(2,1) = 0; M(2,1) = 1; M(2,2) = 1;

    // cout << "QR: " << test_var.Q * test_var.R << endl;

    // cout << "Householder Random Projections QR error: " << (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm() << endl;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Random Test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    test_var.Householder_QR_with_projections();

    // cout << "A: " << test_var.A << endl;
    // cout << "QR: " << test_var.Q; //test_var.Q * test_var.R << endl;

    cout << "Householder Random Projections QR error: " << (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm() << endl;
}

/*
SUCCEEDS ALL
To Run: uncomment the cout << v << endl; in Householder_QR
Expected Output: 
8
-4
-4
12.3246
     -2
*/
void Householder_QR_vtest(randomized_QR test_var) {
    MatrixXd M(3,3);
    M(0,0) = 2; M(0,1) = -1; M(0,2) = -2;
    M(1,0) = -4; M(1,1) = 6; M(1,2) = 3;
    M(2,0) = -4; M(2,1) = -2; M(2,2) = 8;

    test_var.Householder_QR(M);
}

/*
SUCCEEDS ALL
Expected Output: a small quantity (relative error of QR vs A)
*/
void Householder_QR_run_test(randomized_QR test_var) {
    test_var.Householder_QR(test_var.A);

    cout << "Square Householder Standard QR error: " << (test_var.Q * test_var.R - test_var.A).norm() / test_var.A.norm() << endl;

    // Eigen::MatrixXd M(3,2);
    // M(0,0) = 1;
    // M(0,1) = 2;
    // M(1,0) = 3;
    // M(1,1) = 4;
    // M(2,0) = 5;
    // M(2,1) = 6;

    // test_var.Householder_QR(M);
    // cout << "Rectangular Householder Standard QR error: " << (test_var.Q * test_var.R - M).norm() / M.norm() << endl;
}