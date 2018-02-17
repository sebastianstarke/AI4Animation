# if defined _WIN32 || defined __CYGWIN__
#   define EXPORT_API __declspec(dllexport)
# else
#   define EXPORT_API  __attribute__ ((visibility("default")))
# endif

#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <vector>

using namespace Eigen;

//IK Class
class IK {

    MatrixXf Goal;
    std::vector<MatrixXf> Posture;
    std::vector<MatrixXf> Updates;
    MatrixXf Variables;
    MatrixXf Jacobian;
    MatrixXf Gradient;

    int Bones;
    int DOF;
    int Dimensions;

    public : void Initialise(int bones, int dof, int dimensions) {
        Bones = bones;
        DOF = dof;
        Dimensions = dimensions;
        Goal = MatrixXf::Zero(4,4);
        Posture.resize(bones);
        Updates.resize(bones);
        for(int i=0; i<bones; i++) {
            Posture[i] = MatrixXf::Zero(4,4);
            Updates[i] = MatrixXf::Zero(4,4);
        }
        Variables = MatrixXf::Zero(dof, 1);
        Jacobian = MatrixXf::Zero(dimensions, dof);
        Gradient = MatrixXf::Zero(dimensions, 1);
    }

    public : void Iterate(float step, float damping, float resolution) {

    }

    public : void SetGoal(int row, int col, float value) {
        Goal(row, col) = value;
    }

    public : void SetPosture(int matrix, int row, int col, float value) {
        Posture[matrix](row, col) = value;
    }

    public : void SetUpdate(int matrix, int row , int col, float value) {
        Updates[matrix](row, col) = value;
    }

    public : void SetVariable(int index, float value) {
        Variables(index, 0) = value;
    }

    public : float GetVariable(int index) {
        return Variables(index, 0);
    }

    private : MatrixXf FK() {
        return Matrix4f::Zero(4,4);
    }

};

extern "C" {
    IK* Create() {
        return new IK();
    }

    void Delete(IK* obj) {
        delete obj;
    }

    void Initialise(IK* obj, int bones, int dof, int dimensions) {
        obj->Initialise(bones, dof, dimensions);
    }

    void Iterate(IK* obj, float step, float damping, float resolution) {
        obj->Iterate(step, damping, resolution);
    }

    void SetGoal(IK* obj, int row, int col, float value) {
        obj->SetGoal(row, col, value);
    }

    void SetPosture(IK* obj, int matrix, int row, int col, float value) {
        obj->SetPosture(matrix, row, col, value);
    }

    void SetUpdate(IK* obj, int matrix, int row, int col, float value) {
        obj->SetUpdate(matrix, row, col, value);
    }

    void SetVariable(IK* obj, int index, float value) {
        obj->SetVariable(index, value);
    }

    float GetVariable(IK* obj, int index) {
        return obj->GetVariable(index);
    }

}