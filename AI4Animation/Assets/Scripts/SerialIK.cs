using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SerialIK : MonoBehaviour {

	public Vector3 GoalPosition;
	public Transform[] Transforms;

	[Range(0f, 1f)] public float Step = 1.0f;
	[Range(0f, 1f)] public float Damping = 0.1f;

	private int DOF;
	private int Dimensions;
	private Matrix Jacobian;
	private Matrix Gradient;

	private float Differential = 0.001f;
	private int Iterations = 10;

	void Reset() {
		Transforms = new Transform[1] {transform};
	}

	public void UpdateGoal() {
		GoalPosition = GetTipPosition();
		//GoalRotation = GetTipRotation();
	}

	public void ProcessIK() {
		if(Transforms.Length == 0) {
			return;
		}

		if(RequireProcessing()) {
			Matrix4x4[] posture = GetPosture();
			float[] solution = new float[3*Transforms.Length];
			DOF = Transforms.Length * 3;
			Dimensions = 3;
			Jacobian = new Matrix(Dimensions, DOF);
			Gradient = new Matrix(Dimensions, 1);
			for(int i=0; i<Iterations; i++) {
				Iterate(posture, solution);
			}
			FK(posture, solution);
		}
	}

	private bool RequireProcessing() {
		float height = Utility.GetHeight(GoalPosition, LayerMask.GetMask("Ground"));
		//if(height > Goal.position.y - transform.root.position.y) {
			GoalPosition.y = height + (GoalPosition.y - transform.root.position.y);
			return true;
		//}
		//return false;
	}
	
	private void FK(Matrix4x4[] posture, float[] variables) {
		for(int i=0; i<Transforms.Length; i++) {
			Quaternion update = Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+0], Vector3.forward) * Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+1], Vector3.right) * Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+2], Vector3.up);
			Transforms[i].localPosition = posture[i].GetPosition();
			Transforms[i].localRotation = posture[i].GetRotation() * update;
		}
	}

	private Matrix4x4[] GetPosture() {
		Matrix4x4[] posture = new Matrix4x4[Transforms.Length];
		for(int i=0; i<posture.Length; i++) {
			posture[i] = Transforms[i].GetLocalMatrix();
		}
		return posture;
	}

	private Vector3 GetTipPosition() {
		return Transforms[Transforms.Length-1].position;
	}

	private Quaternion GetTipRotation() {
		return Transforms[Transforms.Length-1].rotation;
	}

	private void Iterate(Matrix4x4[] posture, float[] variables) {
		FK(posture, variables);
		Vector3 tipPosition = GetTipPosition();
		//Quaternion tipRotation = GetTipRotation();

		//Jacobian
		for(int j=0; j<DOF; j++) {
			variables[j] += Differential;
			FK(posture, variables);
			variables[j] -= Differential;

			Vector3 deltaPosition = (GetTipPosition() - tipPosition) / Differential;

			//Quaternion deltaRotation = Quaternion.Inverse(tipRotation) * GetTipRotation();
	
			Jacobian.Values[0][j] = deltaPosition.x;
			Jacobian.Values[1][j] = deltaPosition.y;
			Jacobian.Values[2][j] = deltaPosition.z;
			//Jacobian[3,j] = deltaRotation.x / Differential;
			//Jacobian[4,j] = deltaRotation.y / Differential;
			//Jacobian[5,j] = deltaRotation.z / Differential;
			//Jacobian[6,j] = deltaRotation.w / Differential;
		}

		//Gradient Vector
		Vector3 gradientPosition = Step * (GoalPosition - tipPosition);

		//Quaternion gradientRotation = Quaternion.Inverse(tipRotation) * Goal.rotation;

		Gradient.Values[0][0] = gradientPosition.x;
		Gradient.Values[1][0] = gradientPosition.y;
		Gradient.Values[2][0] = gradientPosition.z;
		//Gradient[3] = gradientRotation.x;
		//Gradient[4] = gradientRotation.y;
		//Gradient[5] = gradientRotation.z;
		//Gradient[6] = gradientRotation.w;

		//Jacobian Damped-Least-Squares
		Matrix DLS = DampedLeastSquares();
		for(int m=0; m<DOF; m++) {
			for(int n=0; n<Dimensions; n++) {
				variables[m] += DLS.Values[m][n] * Gradient.Values[n][0];
			}
		}
	}

	private Matrix DampedLeastSquares() {
		Matrix transpose = new Matrix(DOF, Dimensions);
		for(int m=0; m<Dimensions; m++) {
			for(int n=0; n<DOF; n++) {
				transpose.Values[n][m] = Jacobian.Values[m][n];
			}
		}
		Matrix jTj = transpose * Jacobian;
		for(int i=0; i<DOF; i++) {
			jTj.Values[i][i] += Damping*Damping;
		}
		Matrix dls = jTj.GetInverse() * transpose;
		return dls;
  	}

}