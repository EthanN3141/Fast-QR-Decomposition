#ifndef RANDOMIZED_QR_H_
#define RANDOMIZED_QR_H_

#include <vector>
#include <Eigen/Dense>
using Eigen::MatrixXd;


class randomized_QR {
public:

    randomized_QR(size_t n_in, size_t k_in);
    MatrixXd generate_random_rank_k_matrix();
    MatrixXd generate_approximate_rank_k_matrix();

    double thread_safe_floats(); // utility function 
    double gaussian_noise();
    MatrixXd reduce_to_rank_k(); // uses random projection to get rank k approximation to A
    void validate_dimensions(MatrixXd A, MatrixXd B);
    MatrixXd get_P();
    MatrixXd get_P_householder();
    MatrixXd orthonormalizeColumns(MatrixXd U, size_t max_cols = 1e9);
    MatrixXd orthonormalizeColumns_householder(MatrixXd U);


    void QR_with_projections();
    void Householder_QR_with_projections();
    void QR_decomposition(MatrixXd M, bool uses_projections = false);
    void Householder_QR(MatrixXd M, bool uses_projections = false);


    // Parallel matrix operations
    // MatrixXd parallel_xvT_multiplication(MatrixXd x, MatrixXd v);
    MatrixXd parallel_MM_subtraction(MatrixXd M1, MatrixXd M2);
    MatrixXd parallel_Mv_multiplication(MatrixXd M,MatrixXd v);
    double parallel_inner_product(MatrixXd x,MatrixXd v);
    MatrixXd normalize_vector(MatrixXd v);
    double vector_norm(MatrixXd v);


    // TODO FIXME: make this private
    MatrixXd A;
    MatrixXd R;
    MatrixXd Q;
    MatrixXd P;

private:
    size_t n;
    size_t k;
    // MatrixXd A;
    // MatrixXd A_reduced;
};
#endif