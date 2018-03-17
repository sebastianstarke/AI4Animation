# if defined _WIN32 || defined __CYGWIN__
#   define EXPORT_API __declspec(dllexport)
# else
#   define EXPORT_API  __attribute__ ((visibility("default")))
# endif

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

//MFNN Class
class MFNN {

    std::vector<MatrixXf*> References;

    std::vector<int> ControlNeurons;

	MatrixXf Xmean;
	MatrixXf Xstd;
	MatrixXf Ymean;
	MatrixXf Ystd;

	MatrixXf CW0, CW1, CW2;
	MatrixXf Cb0, Cb1, Cb2;

	std::vector<MatrixXf> CPa0, CPa1, CPa2;
	std::vector<MatrixXf> CPb0, CPb1, CPb2;

    MatrixXf X, Y;

    MatrixXf CP;

    int XDimBlend, HDimBlend, YDimBlend, XDim, HDim, YDim;

    public : void Initialise(int xDimBlend, int hDimBlend, int yDimBlend, int xDim, int hDim, int yDim) {
        XDimBlend = xDimBlend;
        HDimBlend = hDimBlend;
        YDimBlend = yDimBlend;
        XDim = xDim;
        HDim = hDim;
        YDim = yDim;

        Xmean = Xmean.Zero(XDim, 1);
        Xstd = Xstd.Zero(XDim, 1);
        Ymean = Ymean.Zero(YDim, 1);
        Ystd = Ystd.Zero(YDim, 1);

        CW0 = CW0.Zero(HDimBlend, XDimBlend);
        Cb0 = Cb0.Zero(HDimBlend, 1);
        CW1 = CW1.Zero(HDimBlend, HDimBlend);
        Cb1 = Cb1.Zero(HDimBlend, 1);
        CW2 = CW2.Zero(YDimBlend, HDimBlend);
        Cb2 = Cb2.Zero(YDimBlend, 1);

        CPa0.resize(YDimBlend);
        CPa1.resize(YDimBlend);
        CPa2.resize(YDimBlend);
        CPb0.resize(YDimBlend);
        CPb1.resize(YDimBlend);
        CPb2.resize(YDimBlend);
		for(int i=0; i<YDimBlend; i++) {
            CPa0[i] = CPa0[i].Zero(HDim, XDim);
            CPb0[i] = CPb0[i].Zero(HDim, 1);
            CPa1[i] = CPa1[i].Zero(HDim, HDim);
            CPb1[i] = CPb1[i].Zero(HDim, 1);
            CPa2[i] = CPa2[i].Zero(YDim, HDim);
            CPb2[i] = CPb2[i].Zero(YDim, 1);
		}

        X = X.Zero(XDim, 1);
        Y = Y.Zero(YDim, 1);

        CP = CP.Zero(YDimBlend, 1);

        References.resize(0);
        References.push_back(&Xmean);
        References.push_back(&Xstd);
        References.push_back(&Ymean);
        References.push_back(&Ystd);
        References.push_back(&CW0);
        References.push_back(&Cb0);
        References.push_back(&CW1);
        References.push_back(&Cb1);
        References.push_back(&CW2);
        References.push_back(&Cb2);
        for(int i=0; i<YDimBlend; i++) {
            References.push_back(&CPa0[i]);
            References.push_back(&CPb0[i]);
            References.push_back(&CPa1[i]);
            References.push_back(&CPb1[i]);
            References.push_back(&CPa2[i]);
            References.push_back(&CPb2[i]);
        }
        References.push_back(&X);
        References.push_back(&Y);
        References.push_back(&CP);
    }

    public : void SetValue(int matrix, int row, int col, float value) {
        if((uint)matrix >= References.size()) {
            return;
        }
        if(row > (*References[matrix]).rows() || col > (*References[matrix]).cols()) {
            return;
        }
        (*References[matrix])(row, col) = value;
    }

    public : float GetValue(int matrix, int row, int col) {
        if((uint)matrix >= References.size()) {
            return 0.0f;
        }
        if(row > (*References[matrix]).rows() || col > (*References[matrix]).cols()) {
            return 0.0f;
        }
        return (*References[matrix])(row, col);
    }

    public : void AddControlNeuron(int index) {
        ControlNeurons.push_back(index);
    }

    public : void Predict() {
        //Normalise input
        MatrixXf _X = (X - Xmean).cwiseQuotient(Xstd);

        //Process MLP
        MatrixXf CN = MatrixXf::Zero(XDimBlend, 1);
        for(uint i=0; i<ControlNeurons.size(); i++) {
            CN(i, 0) = _X(ControlNeurons[i], 0);
        }
        CP = (CW0 * CN) + Cb0; ELU(CP);
        CP = (CW1 * CP) + Cb1; ELU(CP);
        CP = (CW2 * CP) + Cb2; SoftMax(CP);

        //Control Points
		MatrixXf NNW0 = MatrixXf::Zero(HDim, XDim);
		MatrixXf NNW1 = MatrixXf::Zero(HDim, HDim);
		MatrixXf NNW2 = MatrixXf::Zero(YDim, HDim);
		MatrixXf NNb0 = MatrixXf::Zero(HDim, 1);
		MatrixXf NNb1 = MatrixXf::Zero(HDim, 1);
		MatrixXf NNb2 = MatrixXf::Zero(YDim, 1);
		for(int i=0; i<YDimBlend; i++) {
            NNW0 += CPa0[i] * CP(i, 0);
            NNW1 += CPa1[i] * CP(i, 0);
            NNW2 += CPa2[i] * CP(i, 0);
            NNb0 += CPb0[i] * CP(i, 0);
            NNb1 += CPb1[i] * CP(i, 0);
            NNb2 += CPb2[i] * CP(i, 0);
		}

        //Process NN
        Y = (NNW0 * _X) + NNb0; ELU(Y);
        Y = (NNW1 * Y) + NNb1; ELU(Y);
        Y = (NNW2 * Y) + NNb2;

        //Renormalise output
        Y = Y.cwiseProduct(Ystd) + Ymean;
    }

    void ELU(MatrixXf& m) {
        int rows = m.rows();
        for(int i=0; i<rows; i++) {
            m(i, 0) = std::max(m(i, 0), 0.0f) + std::exp(std::min(m(i, 0), 0.0f)) - 1.0f;
        }
    }

    void TanH(MatrixXf& m) {
        int rows = m.rows();
        for(int i=0; i<rows; i++) {
            m(i, 0) = std::tanh(m(i ,0));
        }
    }

    void SoftMax(MatrixXf& m) {
        int rows = m.rows();
        float frac = 0.0f;
        for(int i=0; i<rows; i++) {
            m(i, 0) = std::exp(m(i, 0));
            frac += m(i, 0);
        }
        for(int i=0; i<rows; i++) {
            m(i, 0) /= frac;
        }
    }

};

extern "C" {
    MFNN* Create() {
        return new MFNN();
    }

    void Delete(MFNN* obj) {
        delete obj;
    }

    void Initialise(MFNN* obj, int xDimBlend, int hDimBlend, int yDimBlend, int xDim, int hDim, int yDim) {
        obj->Initialise(xDimBlend, hDimBlend, yDimBlend, xDim, hDim, yDim);
    }

    void SetValue(MFNN* obj, int matrix, int row, int col, float value) {
        obj->SetValue(matrix, row, col, value);
    }

    float GetValue(MFNN* obj, int matrix, int row, int col) {
        return obj->GetValue(matrix, row, col);
    }

    void AddControlNeuron(MFNN* obj, int index) {
        obj->AddControlNeuron(index);
    }

    void Predict(MFNN* obj) {
        obj->Predict();
    }
}