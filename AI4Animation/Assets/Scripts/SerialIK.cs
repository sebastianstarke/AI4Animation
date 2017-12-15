using System;
using System.Collections;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using UnityEngine;

public class SerialIK : MonoBehaviour {

	public Transform Goal;
	public Transform[] Transforms;

	public enum JacobianMethod{Transpose, Pseudoinverse, DampedLeastSquares};
	public JacobianMethod Method = JacobianMethod.DampedLeastSquares;
	public double Step = 1.0;
	public double Damping = 0.1;

	private int DoF;
	private int Entries;
	private Matrix<double> Jacobian;
	private Vector<double> Gradient;

	private double Differential = 0.001;

	void Reset() {
		Transforms = new Transform[1] {transform};
	}

	void Start() {

	}

	void LateUpdate() {
		Process();
	}

	public void Process() {
		if(Goal == null || Transforms.Length == 0) {
			return;
		}

		if(RequireProcessing()) {
			Matrix4x4[] posture = GetPosture();
			double[] solution = new double[3*Transforms.Length];
			DoF = Transforms.Length * 3;
			//Entries = 7;
			Entries = 3;
			Jacobian = Matrix<double>.Build.Dense(Entries, DoF);
			Gradient = Vector<double>.Build.Dense(Entries);
			for(int i=0; i<10; i++) {
				Iterate(posture, solution);
			}
			FK(posture, solution);
		}
	}

	private bool RequireProcessing() {
		Goal.position = GetTipPosition();
		Goal.rotation = GetTipRotation();
		//FOOT IK
		Vector3 tipPosition = GetTipPosition();
		//Vector3 groundPosition = Utility.ProjectGround(tipPosition, LayerMask.GetMask("Ground"));
		float height = Utility.GetHeight(Goal.position, LayerMask.GetMask("Ground"));
		if(height > Goal.position.y) {
			Goal.position = new Vector3(Goal.position.x, height, Goal.position.z);
			return true;
		}
		return false;
	}
	
	private void FK(Matrix4x4[] posture, double[] variables) {
		for(int i=0; i<Transforms.Length; i++) {
			Quaternion update = Quaternion.AngleAxis(Mathf.Rad2Deg*(float)variables[i*3+0], Vector3.forward) * Quaternion.AngleAxis(Mathf.Rad2Deg*(float)variables[i*3+1], Vector3.right) * Quaternion.AngleAxis(Mathf.Rad2Deg*(float)variables[i*3+2], Vector3.up);
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

	private void Iterate(Matrix4x4[] posture, double[] variables) {
		FK(posture, variables);
		Vector3 tipPosition = GetTipPosition();
		//Quaternion tipRotation = GetTipRotation();

		//Jacobian
		for(int j=0; j<DoF; j++) {
			variables[j] += Differential;
			FK(posture, variables);
			variables[j] -= Differential;

			Vector3 deltaPosition = (GetTipPosition() - tipPosition) / (float)Differential;

			//Quaternion deltaRotation = Quaternion.Inverse(tipRotation) * GetTipRotation();
	
			Jacobian[0,j] = deltaPosition.x;
			Jacobian[1,j] = deltaPosition.y;
			Jacobian[2,j] = deltaPosition.z;
			//Jacobian[3,j] = deltaRotation.x / Differential;
			//Jacobian[4,j] = deltaRotation.y / Differential;
			//Jacobian[5,j] = deltaRotation.z / Differential;
			//Jacobian[6,j] = deltaRotation.w / Differential;
		}

		//Gradient Vector
		Vector3 gradientPosition = (float)Step * (Goal.position - tipPosition);

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
			Vector<double> update = Jacobian.Transpose() * Gradient;
			for(int i=0; i<variables.Length; i++) {
				variables[i] += update[i];
			}
		}

		//Jacobian Pseudoinverse
		if(Method == JacobianMethod.Pseudoinverse) {
			Vector<double> update = Jacobian.Transpose() * (Jacobian * Jacobian.Transpose()).Inverse() * Gradient;
			for(int i=0; i<variables.Length; i++) {
				variables[i] += update[i];
			}
		}

		//Jacobian Damped-Least-Squares
		if(Method == JacobianMethod.DampedLeastSquares) {
			Matrix<double> transpose = Jacobian.Transpose();
			Matrix<double> dls = (transpose * Jacobian);
			for(int i=0; i<DoF; i++) {
				dls[i,i] += Damping*Damping;
			}
			dls = dls.Inverse() * transpose;
			Vector<double> update = dls * Gradient;
			for(int i=0; i<variables.Length; i++) {
				variables[i] += update[i];
			}
		}
	}

}