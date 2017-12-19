using System;
using System.Collections;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using UnityEngine;

public class BodyIK : MonoBehaviour {

	public Transform[] Bones;
	public Objective[] Objectives;

	public enum JacobianMethod{Transpose, Pseudoinverse, DampedLeastSquares};
	public JacobianMethod Method = JacobianMethod.DampedLeastSquares;
	[Range(0f, 1f)] public double Step = 1.0;
	[Range(0f, 1f)] public double Damping = 0.1;

	private int DoF;
	private int Entries;
	private double[][] Jacobian;
	private double[] Gradient;
	private double Differential = 0.001;

	[Serializable]
	public class Objective {
		public Transform Tip;
		public Transform Goal;

		public bool IsConverged() {
			//Foot IK
			Goal.position = Tip.position;
			Goal.rotation = Tip.rotation;
			float height = Utility.GetHeight(Goal.position, LayerMask.GetMask("Ground"));
			if(height > Goal.position.y) {
				Goal.position = new Vector3(Goal.position.x, height, Goal.position.z);
				return true;
			}
			return false;
		}
	}

	void Reset() {
		Bones = new Transform[1] {transform};
		Objectives = new Objective[0];
	}

	void Start() {

	}

	void LateUpdate() {
		Process();
	}

	public void Process() {
		if(Objectives.Length == 0 || Bones.Length == 0) {
			return;
		}
		for(int i=0; i<Objectives.Length; i++) {
			if(Objectives[i].Tip == null || Objectives[i].Goal == null) {
				Debug.Log("Missing objective information at objective " + i + ".");
				return;
			}
		}

		if(RequireProcessing()) {
			Matrix4x4[] posture = GetPosture();
			double[] solution = new double[3*Bones.Length];
			DoF = Bones.Length * 3;
			Entries = Objectives.Length * 3;
			Jacobian = new double[Entries][];
			for(int i=0; i<Entries; i++) {
				Jacobian[i] = new double[DoF];
			}
			Gradient = new double[Entries];
			for(int i=0; i<10; i++) {
				Iterate(posture, solution);
			}
			FK(posture, solution);
		}
	}

	private bool RequireProcessing() {
		bool converged = true;
		for(int i=0; i<Objectives.Length; i++) {
			if(!Objectives[i].IsConverged()) {
				converged = false;
			}
		}
		return converged;
	}
	
	private void FK(Matrix4x4[] posture, double[] variables) {
		for(int i=0; i<Bones.Length; i++) {
			Quaternion update = Quaternion.AngleAxis(Mathf.Rad2Deg*(float)variables[i*3+0], Vector3.forward) * Quaternion.AngleAxis(Mathf.Rad2Deg*(float)variables[i*3+1], Vector3.right) * Quaternion.AngleAxis(Mathf.Rad2Deg*(float)variables[i*3+2], Vector3.up);
			Bones[i].localPosition = posture[i].GetPosition();
			Bones[i].localRotation = posture[i].GetRotation() * update;
		}
	}

	private Matrix4x4[] GetPosture() {
		Matrix4x4[] posture = new Matrix4x4[Bones.Length];
		for(int i=0; i<posture.Length; i++) {
			posture[i] = Bones[i].GetLocalMatrix();
		}
		return posture;
	}

	private void Iterate(Matrix4x4[] posture, double[] variables) {
		FK(posture, variables);
		Vector3[] tipPositions = new Vector3[Objectives.Length];
		for(int i=0; i<Objectives.Length; i++) {
			tipPositions[i] = Objectives[i].Tip.position;
		}

		int index = 0;

		//Jacobian
		for(int j=0; j<DoF; j++) {
			variables[j] += Differential;
			FK(posture, variables);
			variables[j] -= Differential;

			index = 0;
			for(int i=0; i<Objectives.Length; i++) {
				Vector3 deltaPosition = (Objectives[i].Tip.position - tipPositions[i]) / (float)Differential;
				//Quaternion deltaRotation = Quaternion.Inverse(tipRotation) * GetTipRotation();
		
				Jacobian[index][j] = deltaPosition.x; index += 1;
				Jacobian[index][j] = deltaPosition.y; index += 1;
				Jacobian[index][j] = deltaPosition.z; index += 1;
				//Jacobian[3,j] = deltaRotation.x / Differential;
				//Jacobian[4,j] = deltaRotation.y / Differential;
				//Jacobian[5,j] = deltaRotation.z / Differential;
				//Jacobian[6,j] = deltaRotation.w / Differential;
			}
		}

		//Gradient Vector
		index = 0;
		for(int i=0; i<Objectives.Length; i++) {
			Vector3 gradientPosition = (float)Step * (Objectives[i].Goal.position - tipPositions[i]);
			//Quaternion gradientRotation = Quaternion.Inverse(tipRotation) * Goal.rotation;

			Gradient[index] = gradientPosition.x; index += 1;
			Gradient[index] = gradientPosition.y; index += 1;
			Gradient[index] = gradientPosition.z; index += 1;
			//Gradient[3] = gradientRotation.x;
			//Gradient[4] = gradientRotation.y;
			//Gradient[5] = gradientRotation.z;
			//Gradient[6] = gradientRotation.w;
		}

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
			double[][] DLS = DampedLeastSquares();
			for(int m=0; m<DoF; m++) {
				for(int n=0; n<Entries; n++) {
					variables[m] += DLS[m][n] * Gradient[n];
				}
			}
		}
	}

	private double[][] DampedLeastSquares() {
		double[][] transpose = Matrix.MatrixCreate(DoF, Entries);
		for(int m=0; m<Entries; m++) {
			for(int n=0; n<DoF; n++) {
				transpose[n][m] = Jacobian[m][n];
			}
		}
		double[][] jTj = Matrix.MatrixProduct(transpose, Jacobian);
		for(int i=0; i<DoF; i++) {
			jTj[i][i] += Damping*Damping;
		}
		double[][] dls = Matrix.MatrixProduct(Matrix.MatrixInverse(jTj), transpose);
		return dls;
  	}

}