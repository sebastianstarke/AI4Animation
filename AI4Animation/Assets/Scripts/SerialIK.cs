using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SerialIK : MonoBehaviour {

	public Transform Goal;
	public Transform[] Transforms;

	public enum JacobianMethod{Transpose, DampedLeastSquares};
	public JacobianMethod Method = JacobianMethod.DampedLeastSquares;
	[Range(0f, 1f)] public float Step = 1.0f;
	[Range(0f, 1f)] public float Damping = 0.1f;

	private int DoF;
	private int Entries;
	private float[][] Jacobian;
	private float[] Gradient;

	private float Differential = 0.001f;

	void Reset() {
		Transforms = new Transform[1] {transform};
	}

	public void UpdateGoal() {
		Goal.position = GetTipPosition();
		Goal.rotation = GetTipRotation();
	}

	public void ProcessIK() {
		if(Goal == null || Transforms.Length == 0) {
			return;
		}

		if(RequireProcessing()) {
			Matrix4x4[] posture = GetPosture();
			float[] solution = new float[3*Transforms.Length];
			DoF = Transforms.Length * 3;
			Entries = 3;
			Jacobian = new float[Entries][];
			for(int i=0; i<Entries; i++) {
				Jacobian[i] = new float[DoF];
			}
			Gradient = new float[Entries];
			for(int i=0; i<10; i++) {
				Iterate(posture, solution);
			}
			FK(posture, solution);
		}
	}

	private bool RequireProcessing() {
		float height = Utility.GetHeight(Goal.position, LayerMask.GetMask("Ground"));
		//if(height > Goal.position.y - transform.root.position.y) {
			Goal.position = new Vector3(Goal.position.x, height + (Goal.position.y - transform.root.position.y), Goal.position.z);
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
		for(int j=0; j<DoF; j++) {
			variables[j] += Differential;
			FK(posture, variables);
			variables[j] -= Differential;

			Vector3 deltaPosition = (GetTipPosition() - tipPosition) / Differential;

			//Quaternion deltaRotation = Quaternion.Inverse(tipRotation) * GetTipRotation();
	
			Jacobian[0][j] = deltaPosition.x;
			Jacobian[1][j] = deltaPosition.y;
			Jacobian[2][j] = deltaPosition.z;
			//Jacobian[3,j] = deltaRotation.x / Differential;
			//Jacobian[4,j] = deltaRotation.y / Differential;
			//Jacobian[5,j] = deltaRotation.z / Differential;
			//Jacobian[6,j] = deltaRotation.w / Differential;
		}

		//Gradient Vector
		Vector3 gradientPosition = Step * (Goal.position - tipPosition);

		//Quaternion gradientRotation = Quaternion.Inverse(tipRotation) * Goal.rotation;

		Gradient[0] = gradientPosition.x;
		Gradient[1] = gradientPosition.y;
		Gradient[2] = gradientPosition.z;
		//Gradient[3] = gradientRotation.x;
		//Gradient[4] = gradientRotation.y;
		//Gradient[5] = gradientRotation.z;
		//Gradient[6] = gradientRotation.w;

		//Jacobian Transpose
		if(Method == JacobianMethod.Transpose) {
			for(int m=0; m<DoF; m++) {
				for(int n=0; n<Entries; n++) {
					variables[m] += Jacobian[n][m] * Gradient[n];
				}
			}
		}

		//Jacobian Damped-Least-Squares
		if(Method == JacobianMethod.DampedLeastSquares) {
			float[][] DLS = DampedLeastSquares();
			for(int m=0; m<DoF; m++) {
				for(int n=0; n<Entries; n++) {
					variables[m] += DLS[m][n] * Gradient[n];
				}
			}
		}
	}

	private float[][] DampedLeastSquares() {
		float[][] transpose = Matrix.MatrixCreate(DoF, Entries);
		for(int m=0; m<Entries; m++) {
			for(int n=0; n<DoF; n++) {
				transpose[n][m] = Jacobian[m][n];
			}
		}
		float[][] jTj = Matrix.MatrixProduct(transpose, Jacobian);
		for(int i=0; i<DoF; i++) {
			jTj[i][i] += Damping*Damping;
		}
		float[][] dls = Matrix.MatrixProduct(Matrix.MatrixInverse(jTj), transpose);
		return dls;
  	}

}