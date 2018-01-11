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

	MatrixXf PFNNXmean;
	MatrixXf PFNNXstd;
	MatrixXf PFNNYmean;
	MatrixXf PFNNYstd;

	MatrixXf MLPXmean;
	MatrixXf MLPXstd;

	MatrixXf CW0, CW1, CW2;
	MatrixXf Cb0, Cb1, Cb2;

	MatrixXf CPa0[4], CPa1[4], CPa2[4];
	MatrixXf CPb0[4], CPb1[4], CPb2[4];

    MatrixXf MLPX, MLPY;
    MatrixXf PFNNX, PFNNY;

    int CDim, XDim, HDim, YDim;

    public : void Initialise(int cDim, int xDim, int hDim, int yDim) {
        CDim = cDim;
        XDim = xDim;
        HDim = hDim;
        YDim = yDim;

        PFNNXmean = PFNNXmean.Zero(XDim, 1);
        PFNNXstd = PFNNXstd.Zero(XDim, 1);
        PFNNYmean = PFNNYmean.Zero(YDim, 1);
        PFNNYstd = PFNNYstd.Zero(YDim, 1);

        MLPXmean = MLPXmean.Zero(CDim, 1);
        MLPXstd = MLPXstd.Zero(CDim, 1);

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

        MLPX = MLPX.Zero(CDim, 1);
        MLPY = MLPY.Zero(4, 1);
        PFNNX = PFNNX.Zero(XDim, 1);
        PFNNY = PFNNY.Zero(YDim, 1);

        References.resize(0);
        References.push_back(&PFNNXmean);
        References.push_back(&PFNNXstd);
        References.push_back(&PFNNYmean);
        References.push_back(&PFNNYstd);
        References.push_back(&MLPXmean);
        References.push_back(&MLPXstd);
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
        References.push_back(&MLPX);
        References.push_back(&MLPY);
        References.push_back(&PFNNX);
        References.push_back(&PFNNY);
    }

    public : void SetValue(int index, int row, int col, float value) {
        (*References[index])(row, col) = value;
    }

    public : float GetValue(int index, int row, int col) {
        return (*References[index])(row, col);
    }

    public : void Predict() {
        //Process MLP
        MLPY = (MLPX - MLPXmean).cwiseQuotient(MLPXstd);
        MLPY = (CW0 * MLPY) + Cb0; ELU(MLPY);
        MLPY = (CW1 * MLPY) + Cb1; ELU(MLPY);
        MLPY = (CW2 * MLPY) + Cb2; SoftMax(MLPY);

        //Control Points
		MatrixXf PFNNW0 = MatrixXf::Zero(HDim, XDim);
		MatrixXf PFNNW1 = MatrixXf::Zero(HDim, HDim);
		MatrixXf PFNNW2 = MatrixXf::Zero(YDim, HDim);
		MatrixXf PFNNb0 = MatrixXf::Zero(HDim, 1);
		MatrixXf PFNNb1 = MatrixXf::Zero(HDim, 1);
		MatrixXf PFNNb2 = MatrixXf::Zero(YDim, 1);
		for(int i=0; i<4; i++) {
			PFNNW0 += CPa0[i] * MLPY(i, 0);
			PFNNW1 += CPa1[i] * MLPY(i, 0);
			PFNNW2 += CPa2[i] * MLPY(i, 0);
			PFNNb0 += CPb0[i] * MLPY(i, 0);
			PFNNb1 += CPb1[i] * MLPY(i, 0);
			PFNNb2 += CPb2[i] * MLPY(i, 0);
		}

        //Process PFNN
        PFNNY = (PFNNX - PFNNXmean).cwiseQuotient(PFNNXstd);
        PFNNY = (PFNNW0 * PFNNY) + PFNNb0; ELU(PFNNY);
        PFNNY = (PFNNW1 * PFNNY) + PFNNb1; ELU(PFNNY);
        PFNNY = (PFNNW2 * PFNNY) + PFNNb2;
        PFNNY = PFNNY.cwiseProduct(PFNNYstd) + PFNNYmean;
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

    void SetValue(APFNN* obj, int index, int row, int col, float value) {
        obj->SetValue(index, row, col, value);
    }

    float GetValue(APFNN* obj, int index, int row, int col) {
        return obj->GetValue(index, row, col);
    }

    void Predict(APFNN* obj) {
        obj->Predict();
    }
}