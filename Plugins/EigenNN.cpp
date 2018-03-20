# if defined _WIN32 || defined __CYGWIN__
#   define EXPORT_API __declspec(dllexport)
# else
#   define EXPORT_API  __attribute__ ((visibility("default")))
# endif

#include <Eigen/Dense>

using namespace Eigen;

extern "C" {
    MatrixXf* Create(int rows, int cols) {
        return new MatrixXf(MatrixXf::Zero(rows, cols));
    }

    void Delete(MatrixXf* T) {
        delete T;
    }

    int GetRows(MatrixXf* T) {
        return (*T).rows();
    }

    int GetCols(MatrixXf* T) {
        return (*T).cols();
    }

    void Add(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* OUT) {
        *OUT = *lhs + *rhs;
    }

    void Subtract(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* OUT) {
        *OUT = *lhs - *rhs;
    }

    void Product(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* OUT) {
        *OUT = *lhs * *rhs;
    }

    void Scale(MatrixXf* lhs, float value, MatrixXf* OUT) {
        *OUT = *lhs * value;
    }

    void SetValue(MatrixXf* T, int row, int col, float value) {
        (*T)(row, col) = value;
    }

    float GetValue(MatrixXf* T, int row, int col) {
        return (*T)(row, col);
    }

    void PointwiseProduct(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* OUT) {
        *OUT = (*lhs).cwiseProduct(*rhs);
    }

    void PointwiseQuotient(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* OUT) {
        *OUT = (*lhs).cwiseQuotient(*rhs);
    }

    void Normalise(MatrixXf* T, MatrixXf* mean, MatrixXf* std, MatrixXf* OUT) {
        *OUT = (*T - *mean).cwiseQuotient(*std);
    }

    void Renormalise(MatrixXf* T, MatrixXf* mean, MatrixXf* std, MatrixXf* OUT) {
        *OUT = (*T).cwiseProduct(*std) + *mean;
    }

    void Layer(MatrixXf* IN, MatrixXf* W, MatrixXf* b, MatrixXf* OUT) {
        *OUT = *W * *IN + *b;
    }

    void Blend(MatrixXf* IN, MatrixXf* W, float w) {
        *IN += *W * w;
    }

    void ELU(MatrixXf* T) {
        int rows = (*T).rows();
        for(int i=0; i<rows; i++) {
            (*T)(i, 0) = std::max((*T)(i, 0), 0.0f) + std::exp(std::min((*T)(i, 0), 0.0f)) - 1.0f;
        }
    }

    void Sigmoid(MatrixXf* T) {
        int rows = (*T).rows();
        for(int i=0; i<rows; i++) {
            (*T)(i, 0) = 1.0f / (1.0f + std::exp(-(*T)(i,0)));
        }
    }

    void TanH(MatrixXf* T) {
        int rows = (*T).rows();
        for(int i=0; i<rows; i++) {
            (*T)(i, 0) = std::tanh((*T)(i, 0));
        }
    }

    void SoftMax(MatrixXf* T) {
        float frac = 0.0f;
        int rows = (*T).rows();
        for(int i=0; i<rows; i++) {
            (*T)(i, 0) = std::exp((*T)(i, 0));
            frac += (*T)(i, 0);
        }
        for(int i=0; i<rows; i++) {
            (*T)(i, 0) /= frac;
        }
    }

    void SetZero(MatrixXf* T) {
        *T = (*T).Zero((*T).rows(), (*T).cols());
    }
}