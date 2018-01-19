# if defined _WIN32 || defined __CYGWIN__
#   define EXPORT_API __declspec(dllexport)
# else
#   define EXPORT_API  __attribute__ ((visibility("default")))
# endif

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

//PFNN Class
class APFNN {

    std::vector<MatrixXf*> References;

    std::vector<int> ControlNeurons;

	MatrixXf Xmean;
	MatrixXf Xstd;
	MatrixXf Ymean;
	MatrixXf Ystd;

	MatrixXf CW0, CW1, CW2;
	MatrixXf Cb0, Cb1, Cb2;

	MatrixXf CPa0[4], CPa1[4], CPa2[4];
	MatrixXf CPb0[4], CPb1[4], CPb2[4];

    MatrixXf X, Y;

    MatrixXf CP;

    int CDim, XDim, HDim, YDim;

    public : void Initialise(int cDim, int xDim, int hDim, int yDim) {
        CDim = cDim;
        XDim = xDim;
        HDim = hDim;
        YDim = yDim;

        Xmean = Xmean.Zero(XDim, 1);
        Xstd = Xstd.Zero(XDim, 1);
        Ymean = Ymean.Zero(YDim, 1);
        Ystd = Ystd.Zero(YDim, 1);

        CW0 = CW0.Zero(CDim, CDim);
        Cb0 = Cb0.Zero(CDim, 1);
        CW1 = CW1.Zero(CDim, CDim);
        Cb1 = Cb1.Zero(CDim, 1);
        CW2 = CW2.Zero(4, CDim);
        Cb2 = Cb2.Zero(4, 1);

		for(int i=0; i<4; i++) {
            CPa0[i] = CPa0[i].Zero(HDim, XDim);
            CPb0[i] = CPb0[i].Zero(HDim, 1);
            CPa1[i] = CPa1[i].Zero(HDim, HDim);
            CPb1[i] = CPb1[i].Zero(HDim, 1);
            CPa2[i] = CPa2[i].Zero(YDim, HDim);
            CPb2[i] = CPb2[i].Zero(YDim, 1);
		}

        X = X.Zero(XDim, 1);
        Y = Y.Zero(YDim, 1);

        CP = CP.Zero(4, 1);

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
        for(int i=0; i<4; i++) {
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
        MatrixXf CN = MatrixXf::Zero(CDim, 1);
        for(uint i=0; i<ControlNeurons.size(); i++) {
            CN(i, 0) = _X(ControlNeurons[i], 0);
        }
        CP = (CW0 * CN) + Cb0; ELU(CP);
        CP = (CW1 * CP) + Cb1; ELU(CP);
        CP = (CW2 * CP) + Cb2; SoftMax(CP);

        //Control Points
		MatrixXf PFNNW0 = MatrixXf::Zero(HDim, XDim);
		MatrixXf PFNNW1 = MatrixXf::Zero(HDim, HDim);
		MatrixXf PFNNW2 = MatrixXf::Zero(YDim, HDim);
		MatrixXf PFNNb0 = MatrixXf::Zero(HDim, 1);
		MatrixXf PFNNb1 = MatrixXf::Zero(HDim, 1);
		MatrixXf PFNNb2 = MatrixXf::Zero(YDim, 1);
		for(int i=0; i<4; i++) {
			PFNNW0 += CPa0[i] * CP(i, 0);
			PFNNW1 += CPa1[i] * CP(i, 0);
			PFNNW2 += CPa2[i] * CP(i, 0);
			PFNNb0 += CPb0[i] * CP(i, 0);
			PFNNb1 += CPb1[i] * CP(i, 0);
			PFNNb2 += CPb2[i] * CP(i, 0);
		}

        //Process PFNN
        Y = (PFNNW0 * _X) + PFNNb0; ELU(Y);
        Y = (PFNNW1 * Y) + PFNNb1; ELU(Y);
        Y = (PFNNW2 * Y) + PFNNb2;

        //Renormalise output
        Y = Y.cwiseProduct(Ystd) + Ymean;
    }

    void ELU(MatrixXf& m) {
        int rows = m.rows();
        for(int i=0; i<rows; i++) {
            m(i, 0) = std::max(m(i, 0), 0.0f) + std::exp(std::min(m(i, 0), 0.0f)) - 1.0f;
        }
    }

    void SoftMax(MatrixXf& m) {
        float frac = 0.0f;
        int rows = m.rows();
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
    APFNN* Create() {
        return new APFNN();
    }

    void Delete(APFNN* obj) {
        delete obj;
    }

    void Initialise(APFNN* obj, int cDim, int xDim, int hDim, int yDim) {
        obj->Initialise(cDim, xDim, hDim, yDim);
    }

    void SetValue(APFNN* obj, int matrix, int row, int col, float value) {
        obj->SetValue(matrix, row, col, value);
    }

    float GetValue(APFNN* obj, int matrix, int row, int col) {
        return obj->GetValue(matrix, row, col);
    }

    void AddControlNeuron(APFNN* obj, int index) {
        obj->AddControlNeuron(index);
    }

    void Predict(APFNN* obj) {
        obj->Predict();
    }
}