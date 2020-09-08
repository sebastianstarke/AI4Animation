# if defined _WIN32 || defined __CYGWIN__
#   define EXPORT_API __declspec(dllexport)
# else
#   define EXPORT_API  __attribute__ ((visibility("default")))
# endif

//#include "stdafx.h" //Use when compiling from Visual Studio
#include "../../Libraries/Eigen/Dense"

using namespace Eigen;
extern "C" {
	EXPORT_API MatrixXf* Create(int rows, int cols) {
		return new MatrixXf(MatrixXf::Zero(rows, cols));
	}

	EXPORT_API void Delete(MatrixXf* ptr) {
		delete(ptr);
	}

	EXPORT_API int GetRows(MatrixXf* ptr) {
		return (*ptr).rows();
	}

	EXPORT_API int GetCols(MatrixXf* ptr) {
		return (*ptr).cols();
	}

	EXPORT_API void SetZero(MatrixXf* ptr) {
		(*ptr).noalias() = (*ptr).Zero((*ptr).rows(), (*ptr).cols());
	}

	EXPORT_API void SetSize(MatrixXf* ptr, int rows, int cols) {
		(*ptr).conservativeResize(rows, cols);
	}

	EXPORT_API void Add(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		(*out).noalias() = *lhs + *rhs;
	}

	EXPORT_API void Subtract(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		(*out).noalias() = *lhs - *rhs;
	}

	EXPORT_API void Product(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		(*out).noalias() = *lhs * *rhs;
	}

	EXPORT_API void Scale(MatrixXf* lhs, float value, MatrixXf* out) {
		(*out).noalias() = *lhs * value;
	}

	EXPORT_API void SetValue(MatrixXf* ptr, int row, int col, float value) {
		(*ptr)(row, col) = value;
	}

	EXPORT_API float GetValue(MatrixXf* ptr, int row, int col) {
		return (*ptr)(row, col);
	}

	EXPORT_API void PointwiseProduct(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		(*out).noalias() = (*lhs).cwiseProduct(*rhs);
	}

	EXPORT_API void PointwiseQuotient(MatrixXf* lhs, MatrixXf* rhs, MatrixXf* out) {
		(*out).noalias() = (*lhs).cwiseQuotient(*rhs);
	}

	EXPORT_API void PointwiseAbsolute(MatrixXf* in, MatrixXf* out) {
		(*out).noalias() = (*in).cwiseAbs();
	}

	EXPORT_API float RowSum(MatrixXf* ptr, int row) {
		return (*ptr).row(row).sum();
	}

	EXPORT_API float ColSum(MatrixXf* ptr, int col) {
		return (*ptr).col(col).sum();
	}

	EXPORT_API float RowMean(MatrixXf* ptr, int row) {
		return (*ptr).row(row).mean();
	}

	EXPORT_API float ColMean(MatrixXf* ptr, int col) {
		return (*ptr).col(col).mean();
	}

	EXPORT_API float RowStd(MatrixXf* ptr, int row) {
		MatrixXf diff = (*ptr).row(row) - (*ptr).row(row).mean() * MatrixXf::Ones(1, (*ptr).rows());
		diff = diff.cwiseProduct(diff);
		return std::sqrt(diff.sum() / (*ptr).cols());
	}

	EXPORT_API float ColStd(MatrixXf* ptr, int col) {
		MatrixXf diff = (*ptr).col(col) - (*ptr).col(col).mean() * MatrixXf::Ones((*ptr).rows(), 1);
		diff = diff.cwiseProduct(diff);
		return std::sqrt(diff.sum() / (*ptr).rows());
	}

	EXPORT_API void Normalise(MatrixXf* ptr, MatrixXf* mean, MatrixXf* std, MatrixXf* out) {
		(*out).noalias() = (*ptr - *mean).cwiseQuotient(*std);
	}

	EXPORT_API void Renormalise(MatrixXf* ptr, MatrixXf* mean, MatrixXf* std, MatrixXf* out) {
		(*out).noalias() = (*ptr).cwiseProduct(*std) + *mean;
	}

	//EXPORT_API void ILayer (MatrixXf* X, MatrixXf* W, MatrixXf* b) {
	//	(*X).noalias() = *W * *X + *b;
	//}

	EXPORT_API void Layer(MatrixXf* in, MatrixXf* W, MatrixXf* b, MatrixXf* out) {
		(*out).noalias() = *W * *in + *b;
	}

	EXPORT_API void Blend(MatrixXf* in, MatrixXf* W, float w) {
		(*in).noalias() += w * *W;
	}

	EXPORT_API void BlendAll(MatrixXf* in, MatrixXf** W, float* w, int length) {
		if(length == 0) {
			SetZero(in);
		} else {
			switch(length) {
				case 1:
				(*in).noalias() = w[0]**W[0];
				break;
				case 2:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1];
				break;
				case 3:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2];
				break;
				case 4:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2] + w[3]**W[3];
				break;
				case 5:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2] + w[3]**W[3] + w[4]**W[4];
				break;
				case 6:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2] + w[3]**W[3] + w[4]**W[4] + w[5]**W[5];
				break;
				case 7:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2] + w[3]**W[3] + w[4]**W[4] + w[5]**W[5] + w[6]**W[6];
				break;
				case 8:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2] + w[3]**W[3] + w[4]**W[4] + w[5]**W[5] + w[6]**W[6] + w[7]**W[7];
				break;
				case 9:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2] + w[3]**W[3] + w[4]**W[4] + w[5]**W[5] + w[6]**W[6] + w[7]**W[7] + w[8]**W[8];
				break;
				case 10:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2] + w[3]**W[3] + w[4]**W[4] + w[5]**W[5] + w[6]**W[6] + w[7]**W[7] + w[8]**W[8] + w[9]**W[9];
				break;
				case 11:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2] + w[3]**W[3] + w[4]**W[4] + w[5]**W[5] + w[6]**W[6] + w[7]**W[7] + w[8]**W[8] + w[9]**W[9] + w[10]**W[10];
				break;
				case 12:
				(*in).noalias() = w[0]**W[0] + w[1]**W[1] + w[2]**W[2] + w[3]**W[3] + w[4]**W[4] + w[5]**W[5] + w[6]**W[6] + w[7]**W[7] + w[8]**W[8] + w[9]**W[9] + w[10]**W[10] + w[11]**W[11];
				break;
				default:
				(*in).noalias() = w[0]**W[0];
				for(int i=1; i<length; i++) {
					(*in).noalias() += w[i]**W[i];
				}
				break;
			}
		}
	}

	EXPORT_API void RELU(MatrixXf* ptr) {
		(*ptr).noalias() = (*ptr).cwiseMax(0.0f);
	}

	EXPORT_API void ELU(MatrixXf* ptr) {
		(*ptr).noalias() = ((*ptr).array().cwiseMax(0.0f) + (*ptr).array().cwiseMin(0.0f).exp() - 1.0f).matrix();
		//int rows = (*ptr).rows();
		//for (int i = 0; i<rows; i++) {
		//	(*ptr)(i, 0) = (std::max)((*ptr)(i, 0), 0.0f) + std::exp((std::min)((*ptr)(i, 0), 0.0f)) - 1.0f;
		//}
	}

	EXPORT_API void Sigmoid(MatrixXf* ptr) {
		int rows = (*ptr).rows();
		for (int i = 0; i<rows; i++) {
			(*ptr)(i, 0) = 1.0f / (1.0f + std::exp(-(*ptr)(i, 0)));
		}
	}

	EXPORT_API void TanH(MatrixXf* ptr) {
		int rows = (*ptr).rows();
		for (int i = 0; i<rows; i++) {
			(*ptr)(i, 0) = std::tanh((*ptr)(i, 0));
		}
	}

	EXPORT_API void SoftMax(MatrixXf* ptr) {
		float frac = 0.0f;
		int rows = (*ptr).rows();
		for (int i = 0; i<rows; i++) {
			(*ptr)(i, 0) = std::exp((*ptr)(i, 0));
			frac += (*ptr)(i, 0);
		}
		for (int i = 0; i<rows; i++) {
			(*ptr)(i, 0) /= frac;
		}
	}

	EXPORT_API void LogSoftMax(MatrixXf* ptr) {
		float frac = 0.0f;
		int rows = (*ptr).rows();
		for (int i = 0; i<rows; i++) {
			(*ptr)(i, 0) = std::exp((*ptr)(i, 0));
			frac += (*ptr)(i, 0);
		}
		for (int i = 0; i<rows; i++) {
			(*ptr)(i, 0) = std::log((*ptr)(i, 0) / frac);
		}
	}

	EXPORT_API void SoftSign(MatrixXf* ptr) {
		int rows = (*ptr).rows();
		for (int i = 0; i<rows; i++) {
			(*ptr)(i, 0) /= 1 + std::abs((*ptr)(i, 0));
		}
	}

	EXPORT_API void Exp(MatrixXf* ptr) {
		int rows = (*ptr).rows();
		for (int i = 0; i<rows; i++) {
			(*ptr)(i, 0) = std::exp((*ptr)(i, 0));
		}
	}
}