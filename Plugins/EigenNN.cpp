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

    void Delete(MatrixXf* m) {
        delete m;
    }

    void Add(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* result) {
        *result = *lhs + *rhs;
    }

    void Subtract(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* result) {
        *result = *lhs - *rhs;
    }

    void Product(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* result) {
        *result = *lhs * *rhs;
    }

    void Scale(MatrixXf* lhs, float value, MatrixXf* result) {
        *result = *lhs * value;
    }

    void SetValue(MatrixXf* m, int row, int col, float value) {
        (*m)(row, col) = value;
    }

    float GetValue(MatrixXf* m, int row, int col) {
        return (*m)(row, col);
    }

    void PointwiseProduct(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* result) {
        *result = (*lhs).cwiseProduct(*rhs);
    }

    void PointwiseQuotient(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* result) {
        *result = (*lhs).cwiseQuotient(*rhs);
    }

    void Normalise(MatrixXf* m, MatrixXf* mean, MatrixXf* std, MatrixXf* result) {
        *result = (*m - *mean).cwiseQuotient(*std);
    }

    void Renormalise(MatrixXf* m, MatrixXf* mean, MatrixXf* std, MatrixXf* result) {
        *result = (*m).cwiseProduct(*std) + *mean;
    }

    void Layer(MatrixXf* x, MatrixXf* y, MatrixXf* W, MatrixXf* b) {
        *y = *W * *x + *b;
    }

    void Blend(MatrixXf* m, MatrixXf* W, float w, MatrixXf* result) {
        *result = *m + *W * w;
    }

    void ELU(MatrixXf* m) {
        int rows = (*m).rows();
        for(int i=0; i<rows; i++) {
            (*m)(i, 0) = std::max((*m)(i, 0), 0.0f) + std::exp(std::min((*m)(i, 0), 0.0f)) - 1.0f;
        }
    }

    void Sigmoid(MatrixXf* m) {
        int rows = (*m).rows();
        for(int i=0; i<rows; i++) {
            (*m)(i, 0) = 1.0f / (1.0f + std::exp(-(*m)(i,0)));
        }
    }

    void TanH(MatrixXf* m) {
        int rows = (*m).rows();
        for(int i=0; i<rows; i++) {
            (*m)(i, 0) = std::tanh((*m)(i, 0));
        }
    }

    void SoftMax(MatrixXf* m) {
        float frac = 0.0f;
        int rows = (*m).rows();
        for(int i=0; i<rows; i++) {
            (*m)(i, 0) = std::exp((*m)(i, 0));
            frac += (*m)(i, 0);
        }
        for(int i=0; i<rows; i++) {
            (*m)(i, 0) /= frac;
        }
    }

    void SetZero(MatrixXf* m) {
        *m = (*m).Zero((*m).rows(), (*m).cols());
    }
}