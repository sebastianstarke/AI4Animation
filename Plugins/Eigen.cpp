# if defined _WIN32 || defined __CYGWIN__
#   define EXPORT_API __declspec(dllexport)
# else
#   define EXPORT_API  __attribute__ ((visibility("default")))
# endif

//#include "stdafx.h" //Use when compiling from Visual Studio
#include "Eigen/Dense"

using namespace Eigen;
extern "C" {
	EXPORT_API MatrixXf* Create(int rows, int cols) {
		return new MatrixXf(MatrixXf::Zero(rows, cols));
	}

	EXPORT_API void Delete(MatrixXf* T) {
		delete T;
	}

	EXPORT_API int GetRows(MatrixXf* T) {
		return (*T).rows();
	}

	EXPORT_API int GetCols(MatrixXf* T) {
		return (*T).cols();
	}

	EXPORT_API void SetZero(MatrixXf* T) {
		*T = (*T).Zero((*T).rows(), (*T).cols());
	}

	EXPORT_API void SetSize(MatrixXf* T, int rows, int cols) {
		(*T).conservativeResize(rows, cols);
	}

	EXPORT_API void Add(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		*out = *lhs + *rhs;
	}

	EXPORT_API void Subtract(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		*out = *lhs - *rhs;
	}

	EXPORT_API void Product(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		*out = *lhs * *rhs;
	}

	EXPORT_API void Scale(MatrixXf* lhs, float value, MatrixXf* out) {
		*out = *lhs * value;
	}

	EXPORT_API void SetValue(MatrixXf* T, int row, int col, float value) {
		(*T)(row, col) = value;
	}

	EXPORT_API float GetValue(MatrixXf* T, int row, int col) {
		return (*T)(row, col);
	}

	EXPORT_API void PointwiseProduct(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		*out = (*lhs).cwiseProduct(*rhs);
	}

	EXPORT_API void PointwiseQuotient(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		*out = (*lhs).cwiseQuotient(*rhs);
	}

	EXPORT_API void PointwiseAbsolute(MatrixXf* in, MatrixXf* out) {
		*out = (*in).cwiseAbs();
	}

	EXPORT_API float RowSum(MatrixXf* T, int row) {
		return (*T).row(row).sum();
	}

	EXPORT_API float ColSum(MatrixXf* T, int col) {
		return (*T).col(col).sum();
	}

	EXPORT_API float RowMean(MatrixXf* T, int row) {
		return (*T).row(row).mean();
	}

	EXPORT_API float ColMean(MatrixXf* T, int col) {
		return (*T).col(col).mean();
	}

	EXPORT_API float RowStd(MatrixXf* T, int row) {
		MatrixXf diff = (*T).row(row) - (*T).row(row).mean() * MatrixXf::Ones(1, (*T).rows());
		diff = diff.cwiseProduct(diff);
		return std::sqrt(diff.sum() / (*T).cols());
	}

	EXPORT_API float ColStd(MatrixXf* T, int col) {
		MatrixXf diff = (*T).col(col) - (*T).col(col).mean() * MatrixXf::Ones((*T).rows(), 1);
		diff = diff.cwiseProduct(diff);
		return std::sqrt(diff.sum() / (*T).rows());
	}

	EXPORT_API void Normalise(MatrixXf* T, MatrixXf* mean, MatrixXf* std, MatrixXf* out) {
		*out = (*T - *mean).cwiseQuotient(*std);
	}

	EXPORT_API void Renormalise(MatrixXf* T, MatrixXf* mean, MatrixXf* std, MatrixXf* out) {
		*out = (*T).cwiseProduct(*std) + *mean;
	}

	EXPORT_API void Layer(MatrixXf* in, MatrixXf* W, MatrixXf* b, MatrixXf* out) {
		*out = *W * *in + *b;
	}

	EXPORT_API void Blend(MatrixXf* in, MatrixXf* W, float w) {
		*in += *W * w;
	}

	EXPORT_API void ELU(MatrixXf* T) {
		int rows = (*T).rows();
		for (int i = 0; i<rows; i++) {
			(*T)(i, 0) = (std::max)((*T)(i, 0), 0.0f) + std::exp((std::min)((*T)(i, 0), 0.0f)) - 1.0f;
		}
	}

	EXPORT_API void Sigmoid(MatrixXf* T) {
		int rows = (*T).rows();
		for (int i = 0; i<rows; i++) {
			(*T)(i, 0) = 1.0f / (1.0f + std::exp(-(*T)(i, 0)));
		}
	}

	EXPORT_API void TanH(MatrixXf* T) {
		int rows = (*T).rows();
		for (int i = 0; i<rows; i++) {
			(*T)(i, 0) = std::tanh((*T)(i, 0));
		}
	}

	EXPORT_API void SoftMax(MatrixXf* T) {
		float frac = 0.0f;
		int rows = (*T).rows();
		for (int i = 0; i<rows; i++) {
			(*T)(i, 0) = std::exp((*T)(i, 0));
			frac += (*T)(i, 0);
		}
		for (int i = 0; i<rows; i++) {
			(*T)(i, 0) /= frac;
		}
	}

	EXPORT_API void LogSoftMax(MatrixXf* T) {
		float frac = 0.0f;
		int rows = (*T).rows();
		for (int i = 0; i<rows; i++) {
			(*T)(i, 0) = std::exp((*T)(i, 0));
			frac += (*T)(i, 0);
		}
		for (int i = 0; i<rows; i++) {
			(*T)(i, 0) = std::log((*T)(i, 0) / frac);
		}
	}

	EXPORT_API void SoftSign(MatrixXf* T) {
		int rows = (*T).rows();
		for (int i = 0; i<rows; i++) {
			(*T)(i, 0) /= 1 + std::abs((*T)(i, 0));
		}
	}

	EXPORT_API void Exp(MatrixXf* T) {
		int rows = (*T).rows();
		for (int i = 0; i<rows; i++) {
			(*T)(i, 0) = std::exp((*T)(i, 0));
		}
	}
}