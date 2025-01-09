#include <iostream>
#include <omp.h>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "randomized_QR.h"
#include <chrono>

using Eigen::MatrixXd;

using namespace std;

randomized_QR::randomized_QR(size_t n_in, size_t k_in) {
    n = n_in;
    k = k_in;

    // set for the rest of the initialization
    Eigen::setNbThreads(omp_get_num_threads()); 
    // MatrixXd TM(3,3);
    // TM(0,0) = 1; TM(0,1) = 2; TM(0,2) = 3;
    // TM(1,0) = 4; TM(1,1) = 5; TM(1,2) = 6;
    // TM(2,0) = 7; TM(2,1) = 8; TM(2,2) = 9;
    // A = TM;

    A = generate_approximate_rank_k_matrix();

    // cout << "orignaly A is: " << A << endl;
    // This A is dependent
    // MatrixXd M(3,3);
    // M(0,0) = 1; M(0,1) = 1; M(0,2) = 2;
    // M(1,0) = 1; M(1,1) = 0; M(1,2) = 1;
    // M(2,1) = 0; M(2,1) = 1; M(2,2) = 1;
    // A = M;
    // cout << n << ", " << k << endl;
}

MatrixXd randomized_QR::generate_random_rank_k_matrix(){
    MatrixXd mat = MatrixXd::Zero(n, n);

    MatrixXd rand_vec(n,1);

    // the sum of k outer products is rank k
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < n; j++) {
            rand_vec(j) = thread_safe_floats();
        }
        mat += rand_vec*rand_vec.transpose();
    }

    // cout << "mat is: " << mat << endl;


    // Anther way to get a rank k matrix. This creates a lopsided matrix though with increasing variance to the right of the matrix.
    // // get the independent part first
    // #pragma omp parallel for
    // for (size_t j = 0; j < k; j++) {
    //     for (size_t i = 0; i < n; i++) {
    //         mat(i,j) = thread_safe_floats();
    //     }
    // }
    // #pragma omp parallel for
    // for (size_t j = k; j < n; j++) {
    //     for (size_t ref_col = 0; ref_col < k; ref_col++){
    //         float multiplier = thread_safe_floats();
    //         for (size_t i = 0; i < n; i++) {
    //             mat(i,j) += multiplier * mat(i,ref_col); 
    //         }
    //     }
    // }

    return mat;

}

MatrixXd randomized_QR::generate_approximate_rank_k_matrix() {
    MatrixXd rK_mat = generate_random_rank_k_matrix();

    // add some small Gaussian noise to the matrix
    #pragma omp parallel for
    for (int row = 0; row < rK_mat.rows(); row++) {
        for (int col = 0; col < rK_mat.cols(); col++) {
            if (thread_safe_floats() > 0) {
                rK_mat(row,col) += gaussian_noise();
            }
        }
    }

    return rK_mat;
}

double randomized_QR::gaussian_noise() {
    static thread_local mt19937 generator(random_device{}());
    normal_distribution<double> distribution(0,.001);
    return distribution(generator);
}

// generates random thread safe floats in [-10,10]
double randomized_QR::thread_safe_floats() {
    static thread_local mt19937 generator(random_device{}());
    uniform_real_distribution<double> distribution(-10,10);
    return distribution(generator);
}

/*
Purpose: Validates that A@B is a successful matrix multiplication. If it is not, exit the program
*/
void randomized_QR::validate_dimensions(MatrixXd A, MatrixXd B) {
    if(A.cols() != B.rows()) {
            throw invalid_argument(
                "Matrix multiplication error: dimensions do not match (" +
                to_string(A.rows()) + "x" + to_string(A.cols()) +
                " cannot be multiplied with " +
                to_string(B.rows()) + "x" + to_string(B.cols()) + ").");
        }
}
/*
Purpose: Multiplies M@v and returns the matrix multiplication
Note: M and v must have compatible dimnsions
*/ 
MatrixXd randomized_QR::parallel_Mv_multiplication(MatrixXd M, MatrixXd v) {
    validate_dimensions(M,v);

    MatrixXd end_vector(M.rows(), v.cols());
    #pragma omp parallel for
    for (size_t row = 0; row < M.rows(); row++) {
        double elem_val = 0;
        for (size_t col = 0; col < M.cols(); col++) {
            elem_val += M(row,col) * v(col);
        }
        end_vector(row) = elem_val;
    }

    return end_vector;
}

MatrixXd randomized_QR::parallel_MM_subtraction(MatrixXd M1, MatrixXd M2) {
    MatrixXd end_matrix(M1.rows(), M1.cols());
    #pragma omp parallel for
    for (size_t row = 0; row < M1.rows(); row++) {
        double elem_val = 0;
        for (size_t col = 0; col < M1.cols(); col++) {
            end_matrix(row,col) = M1(row,col) - M2(row,col);
        }
    }

    return end_matrix;
}

/*
for most practical purposes, this is just sequential
*/
double randomized_QR::parallel_inner_product(MatrixXd x, MatrixXd v) {
    // only do it parallel for large inner products
    if (x.rows() < 20'000) {
        // double returnable = (x.transpose() *v)(0);
        return (x.transpose() *v)(0);
    }
    // validate_dimensions(x.transpose(),v);

    double IP = 0; // initialize it to 0

    // sumProduct of the vectors using a reduction across processors
    #pragma omp parallel for reduction(+ : IP)
    for (size_t i = 0; i < x.rows(); i++) {
        IP += x(i) * v(i);
    }

    return IP;
}

// uses modified gram schmidt to orthonormalize cols of U
// supports up to 1e9 columns (the default value for max_cols)
MatrixXd randomized_QR::orthonormalizeColumns(MatrixXd U, size_t max_cols) {

    // cout << "start of orthonormalizeColumns: " << U.cols() << U.rows() << endl << endl;
    size_t num_cols = U.cols();

    for (int column = 0; column < min(max_cols, num_cols); column++) {
        // normalize the current column
        U.col(column) /= sqrt(parallel_inner_product(U.col(column),U.col(column)));

        // subtract off the projection of current col from all subsequent ones
        #pragma omp parallel for
        for (int next_col = column + 1; next_col < min(max_cols, num_cols); next_col++) {
            U.col(next_col) -= parallel_inner_product(U.col(next_col), U.col(column)) * U.col(column);
        }
    }

    return U; 
}

MatrixXd randomized_QR::normalize_vector(MatrixXd v) {
    return v / sqrt(parallel_inner_product(v,v));
}

double randomized_QR::vector_norm(MatrixXd v) {
    return sqrt(parallel_inner_product(v,v));
}

// Gets the matrix P for which A is approximately PPTA
MatrixXd randomized_QR::get_P() {

    // U = A*Omega;    A in (n,n)      Omega in (n,k)
    // Eigen::setNbThreads(omp_get_num_threads()); 

    // Initialize the Omega matrix randomly
    MatrixXd Omega(n,k);
    #pragma omp parallel for
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < k; col++) {
            Omega(row,col) = thread_safe_floats();
        }
    }
    
    // Form the basis for Col(A), ie. the matrix U
    MatrixXd U(n,k);
    U = A * Omega;

    MatrixXd P = orthonormalizeColumns(U);

    return P;
}

// TODO FIXME: use householder for normalization here
MatrixXd randomized_QR::get_P_householder() {
    // U = A*Omega;    A in (n,n)      Omega in (n,k)
    // Eigen::setNbThreads(omp_get_num_threads()); 

    // Initialize the Omega matrix randomly
    MatrixXd Omega(n,k);
    #pragma omp parallel for
    for (size_t row = 0; row < n; row++) {
        for (size_t col = 0; col < k; col++) {
            Omega(row,col) = thread_safe_floats();
        }
    }
    
    // Form the basis for Col(A), ie. the matrix U
    MatrixXd U(n,k);
    U = A * Omega;

    // orthonormalize the basis
    // TODO FIGURE OUT MY IMPLEMENT TESTCODE
    // MatrixXd P_n = Eigen::HouseholderQR<Eigen::MatrixXd>(U).householderQ();
    // MatrixXd P = P_n.leftCols(k);
    // cout << "dims of P: " << P.rows() << ", " << P.cols() << endl;

    MatrixXd P = orthonormalizeColumns_householder(U);

    return P;
}

void randomized_QR::Householder_QR_with_projections() {
    Eigen::setNbThreads(omp_get_num_threads());
    P = get_P_householder();

    Householder_QR(P.transpose()*A, true);

    // QR_decomposition(P.transpose()*A)
    // Q = P * Q
    // // R = R    
}

void randomized_QR::QR_with_projections() {
    // Eigen::setNbThreads(omp_get_num_threads());
    P = get_P();

    QR_decomposition(P.transpose()*A, true);
}

// the default value of uses_projections is false
void randomized_QR::QR_decomposition(MatrixXd M, bool uses_projections) {


    // auto start = chrono::high_resolution_clock::now();
    

    // Q.resize(M.rows(), M.rows());
    R.resize(M.rows(), M.cols());

    // Orthonormalize a square portion. Wide matrices have many rows of ommitable 0s when orthonormalized
    Q = orthonormalizeColumns(M.leftCols(M.rows()));

    // auto end = chrono::high_resolution_clock::now();
    // cout << "orthonormalize cols time " <<chrono::duration<float>(end - start).count() << endl;

    // start = chrono::high_resolution_clock::now();
    // Get R
    for (size_t row = 0; row < M.rows(); row++) {
        // more reliable parallelism for the inner loop if uses_projections==true
        #pragma omp parallel for
        for (size_t col = row; col < M.cols(); col++) {
            R(row,col) = parallel_inner_product(Q.col(row), M.col(col));
        }
    }
    // end = chrono::high_resolution_clock::now();
    // cout << "Get R using inner products " <<chrono::duration<float>(end - start).count() << endl;

    // Get Q
    if (uses_projections) {
        Q = P * Q;
    }
    
}

MatrixXd randomized_QR::orthonormalizeColumns_householder(MatrixXd M) {
    MatrixXd Q_temp = MatrixXd::Identity(M.rows(),M.rows());

    MatrixXd v; // shrink each iteration

    double sign;
    for (size_t elim_col = 0; elim_col < min(M.rows()-1,M.cols()); elim_col++) {
        // initialize v to the appropriate sub-column
        v = M.block(elim_col,elim_col,M.rows()-elim_col,1);
        
        // v = normalized(x-(-sgn(x))*||x||*e_j)
        sign = v(0) >= 0 ? 1.0 : -1.0;
        v(0) += sign * vector_norm(v);
        v = normalize_vector(v);


        // Modify the bottom right block of M as follows (towards upper triangularization [R])
        // M - (2v)(vTM) = M-(2v)(MTv)T

        M.block(elim_col,elim_col,M.rows()-elim_col,M.cols() - elim_col) = 
        parallel_MM_subtraction(M.block(elim_col,elim_col,M.rows()-elim_col,M.cols() - elim_col),
         (2*v) * parallel_Mv_multiplication(M.block(elim_col,elim_col,M.rows()-elim_col,M.cols() - elim_col).transpose(),v).transpose());


        // Modify the right blocks of Q
        // Lower Right
        Q_temp.block(elim_col,elim_col,Q_temp.rows()-elim_col,Q_temp.cols() - elim_col) = 
        parallel_MM_subtraction(Q_temp.block(elim_col,elim_col,Q_temp.rows()-elim_col,Q_temp.cols() - elim_col),
        parallel_Mv_multiplication(Q_temp.block(elim_col,elim_col,Q_temp.rows()-elim_col,Q_temp.cols() - elim_col),v) *(2*v.transpose()));

        // Upper Right
        if (elim_col != 0) {
            Q_temp.block(0,elim_col,elim_col,Q_temp.cols() - elim_col) = 
            parallel_MM_subtraction(Q_temp.block(0,elim_col,elim_col,Q_temp.cols() - elim_col),
            parallel_Mv_multiplication(Q_temp.block(0,elim_col,elim_col,Q_temp.cols() - elim_col),v) * (2*v.transpose()));
        } 
    }

    return Q_temp.block(0,0,M.rows(),M.cols());
}

void randomized_QR::Householder_QR(MatrixXd M, bool uses_projections) {
    /* Algorithm Flow
        compute v from cur col diagonal and below
            create Gestalt for the householder matrix (don't construct)
        perform matrix mult H_j * A.block(j,j,n,n) or whatever
        go til j = n (don't do n tho cuz we're done)
    
    */

    Q = MatrixXd::Identity(M.rows(),M.rows());

    MatrixXd v; // shrink each iteration

    double sign;
    for (size_t elim_col = 0; elim_col < min(M.rows()-1,M.cols()); elim_col++) {
        // initialize v to the appropriate sub-column
        v = M.block(elim_col,elim_col,M.rows()-elim_col,1);
        
        // v = normalized(x-(-sgn(x))*||x||*e_j)
        sign = v(0) >= 0 ? 1.0 : -1.0;
        v(0) += sign * vector_norm(v);
        v = normalize_vector(v);


        // Modify the bottom right block of M as follows (towards upper triangularization [R])
        // M - (2v)(vTM) = M-(2v)(MTv)T
        // M.block(elim_col,elim_col,M.rows()-elim_col,M.cols() - elim_col) -= 
        // (2*v) * parallel_Mv_multiplication(M.block(elim_col,elim_col,M.rows()-elim_col,M.cols() - elim_col).transpose(),v).transpose();

        // // Modify the right blocks of Q
        // // Lower Right
        // Q.block(elim_col,elim_col,Q.rows()-elim_col,Q.cols() - elim_col) -= 
        // parallel_Mv_multiplication(Q.block(elim_col,elim_col,Q.rows()-elim_col,Q.cols() - elim_col),v) *(2*v.transpose());

        // // Upper Right
        // Q.block(0,elim_col,elim_col,Q.cols() - elim_col) -= 
        // parallel_Mv_multiplication(Q.block(0,elim_col,elim_col,Q.cols() - elim_col),v) * (2*v.transpose());


        M.block(elim_col,elim_col,M.rows()-elim_col,M.cols() - elim_col) = 
        parallel_MM_subtraction(M.block(elim_col,elim_col,M.rows()-elim_col,M.cols() - elim_col),
         (2*v) * parallel_Mv_multiplication(M.block(elim_col,elim_col,M.rows()-elim_col,M.cols() - elim_col).transpose(),v).transpose());


        // Modify the right blocks of Q
        // Lower Right
        Q.block(elim_col,elim_col,Q.rows()-elim_col,Q.cols() - elim_col) = 
        parallel_MM_subtraction(Q.block(elim_col,elim_col,Q.rows()-elim_col,Q.cols() - elim_col),
        parallel_Mv_multiplication(Q.block(elim_col,elim_col,Q.rows()-elim_col,Q.cols() - elim_col),v) *(2*v.transpose()));

        // Upper Right
        if (elim_col != 0) {
            Q.block(0,elim_col,elim_col,Q.cols() - elim_col) = 
            parallel_MM_subtraction(Q.block(0,elim_col,elim_col,Q.cols() - elim_col),
            parallel_Mv_multiplication(Q.block(0,elim_col,elim_col,Q.cols() - elim_col),v) * (2*v.transpose()));
        }
    }

    R = M;

    if (uses_projections) {
        Q = P * Q;
    }
}
