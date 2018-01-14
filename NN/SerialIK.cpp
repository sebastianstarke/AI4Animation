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
class SerialIK {

    Matrix4f Goal;
    std::vector<Matrix4f> Transformations;
    MatrixXf Variables;
    MatrixXf Jacobian;
    MatrixXf Gradient;

    int Bones;
    int DOF;
    int Dimensions;
    float Rad2Deg;
    float Differential;

    public : void Initialise(int bones, int dof, int dimensions) {
        Bones = bones;
        DOF = dof;
        Dimensions = Dimensions;
        Rad2Deg = 57.29578049f;
        Differential = 0.00001f;
        Goal = Matrix4f::Zero(4, 4);
        Transformations.resize(bones);
        for(int i=0; i<bones; i++) {
            Transformations[i] = Matrix4f::Zero(4, 4);
        }
        Variables = MatrixXf::Zero(dof, 1);
        Jacobian = MatrixXf::Zero(dimensions, dof);
        Gradient = MatrixXf::Zero(dimensions, 1);
    }

    public : void Process(int iterations, float step, float damping) {
        for(int i=0; i<iterations; i++) {
            Iterate(step, damping);
        }

        /*
		Matrix4x4 result = FK(sequence, variables);
		Vector3 tipPosition = result.GetPosition();

		//Jacobian
		for(int j=0; j<DoF; j++) {
			variables[j] += Differential;
			result = FK(sequence, variables);
			variables[j] -= Differential;

			//Delta
			Vector3 deltaPosition = (result.GetPosition() - tipPosition) / Differential;
			Jacobian.Values[0][j] = deltaPosition.x;
			Jacobian.Values[1][j] = deltaPosition.y;
			Jacobian.Values[2][j] = deltaPosition.z;
		}

		//Gradient Vector
		Vector3 gradientPosition = Step * (GoalPosition - tipPosition);
		Gradient.Values[0][0] = gradientPosition.x;
		Gradient.Values[1][0] = gradientPosition.y;
		Gradient.Values[2][0] = gradientPosition.z;

		//Jacobian Damped-Least-Squares
		Matrix DLS = DampedLeastSquares();
		for(int m=0; m<DoF; m++) {
			for(int n=0; n<Entries; n++) {
				variables[m] += DLS.Values[m][n] * Gradient.Values[n][0];
			}
		}
		*/
	}

    private : void Iterate(float step, float damping) {
        Vector3f goalPosition = Vector3f::Zero();
        goalPosition(0) = Goal(3,0);
        goalPosition(1) = Goal(3,1);
        goalPosition(2) = Goal(3,2);

        Matrix4f result = FK();

        Vector3f tipPosition = Vector3f::Zero();
        tipPosition(0) = result(3,0);
        tipPosition(1) = result(3,1);
        tipPosition(2) = result(3,2);

        //Jacobian
        for(int i=0; i<DOF; i++) {
            //Modify
            Variables(i,0) += Differential;
            result = FK();
            Variables(i,0) -= Differential;

            //Delta
            Vector3f deltaPosition = Vector3f::Zero();
            deltaPosition(0) = (result(3,0) - tipPosition(0)) / Differential;
            deltaPosition(1) = (result(3,1) - tipPosition(1)) / Differential;
            deltaPosition(2) = (result(3,2) - tipPosition(2)) / Differential;
            Jacobian(0,i) = deltaPosition(0);
            Jacobian(1,i) = deltaPosition(1);
            Jacobian(2,i) = deltaPosition(2);
        }

        //Gradient
        Vector3f gradientPosition = step * (goalPosition - tipPosition);
        Gradient(0,0) = gradientPosition(0);
        Gradient(1,0) = gradientPosition(1);
        Gradient(2,0) = gradientPosition(2);

        //DLS
        MatrixXf transpose = Jacobian.transpose();
        MatrixXf jTj = transpose * Jacobian;
        for(int i=0; i<DOF; i++) {
            jTj(i,i) += damping*damping;
        }
        MatrixXf dls = jTj.inverse() * transpose;

        //Update
        Variables = dls * Gradient;
        /*
		for(int m=0; m<DOF; m++) {
			for(int n=0; n<Dimensions; n++) {
				Variables[m] += dls(m,n) * Gradient(n,0);
			}
		}
        */
    }

	private : void DampedLeastSquares() {
		/*
		Matrix transpose = new Matrix(DoF, Entries);
		for(int m=0; m<Entries; m++) {
			for(int n=0; n<DoF; n++) {
				transpose.Values[n][m] = Jacobian.Values[m][n];
			}
		}
		Matrix jTj = transpose * Jacobian;
		for(int i=0; i<DoF; i++) {
			jTj.Values[i][i] += Damping*Damping;
		}
		Matrix dls = jTj.GetInverse() * transpose;
		return dls;
		*/
		
    }

    public : void SetGoal(int row, int col, float value) {
        Goal(row, col) = value;
    }

    public : void SetValue(int matrix, int row, int col, float value) {
        Transformations[matrix](row, col) = value;
    }

    public : void SetVariable(int index, float value) {
        Variables(index, 0) = value;
    }

    public : float GetVariable(int index) {
        return Variables(index, 0);
    }

    private : Matrix4f FK() {
        Matrix4f result = Matrix4f::Identity();
        for(uint i=0; i<Transformations.size(); i++) {
            Matrix3f rotation;
            rotation = 
                AngleAxisf(Rad2Deg * Variables(i*3+0, 0), Vector3f::UnitZ()) * 
                AngleAxisf(Rad2Deg * Variables(i*3+1, 0), Vector3f::UnitX()) * 
                AngleAxisf(Rad2Deg * Variables(i*3+2, 0), Vector3f::UnitY());
            Matrix4f update = Matrix4f::Identity();
            update.block(0,0,3,3) = rotation;
            if(i == 0) {
                result = Transformations[i] * update;
            } else {
                result = result * Transformations[i] * update;
            }
        }
        return result;
        /*
		Matrix4x4 result = Matrix4x4.identity;
		for(int i=0; i<sequence.Length; i++) {
			Matrix4x4 update = Matrix4x4.TRS(Vector3.zero, Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+0], Vector3.forward) * Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+1], Vector3.right) * Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+2], Vector3.up), Vector3.one);
			result = i == 0 ? sequence[i] * update : result * sequence[i] * update;
		}
        */
    }

};

extern "C" {
    SerialIK* Create() {
        return new SerialIK();
    }

    void Delete(SerialIK* obj) {
        delete obj;
    }

    void Initialise(SerialIK* obj, int bones, int dof, int dimensions) {
        obj->Initialise(bones, dof, dimensions);
    }

    void Process(SerialIK* obj, int iterations, float step, float damping) {
        obj->Process(iterations, step, damping);
    }

    void SetGoal(SerialIK* obj, int row, int col, float value) {
        obj->SetGoal(row, col, value);
    }

    void SetValue(SerialIK* obj, int matrix, int row, int col, float value) {
        obj->SetValue(matrix, row, col, value);
    }

    void SetVariable(SerialIK* obj, int index, float value) {
        obj->SetVariable(index, value);
    }

    float GetVariable(SerialIK* obj, int index) {
        return obj->GetVariable(index);
    }

}