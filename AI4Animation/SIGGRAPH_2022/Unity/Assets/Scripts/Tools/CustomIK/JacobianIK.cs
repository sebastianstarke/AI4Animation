using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JacobianIK : MonoBehaviour {

	public Vector3 Offset;

	public Transform[] Bones;
	public Objective[] Objectives;

	public int Iterations = 10;
	[Range(0f, 1f)] public float Step = 1f;
	[Range(0f, 1f)] public float Differential = 0.01f;

	public bool SeedZeroPose = false;

	private Matrix4x4[] ZeroPose = null;

	private int DoF;
	private int Entries;
	private float[][] Jacobian;
	private float[] Gradient;

	[Serializable]
	public class Objective {
		public Transform Tip;
		public Transform Goal;
	}

	void Reset() {
		Bones = new Transform[1] {transform};
		Objectives = new Objective[0];
	}

	void Start() {
		ZeroPose = GetPosture();
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

		Matrix4x4[] posture = SeedZeroPose ? ZeroPose : GetPosture();
		float[] solution = new float[3*Bones.Length];
		DoF = Bones.Length * 3;
		Entries = Objectives.Length * 3;
		Jacobian = new float[Entries][];
		for(int i=0; i<Entries; i++) {
			Jacobian[i] = new float[DoF];
		}
		Gradient = new float[Entries];
		for(int i=0; i<Iterations; i++) {
			Iterate(posture, solution);
		}
		FK(posture, solution);

		for(int i=0; i<Objectives.Length; i++) {
			Objectives[i].Tip.rotation = Objectives[i].Goal.rotation;
		}
	}

	private void FK(Matrix4x4[] posture, float[] variables) {
		for(int i=0; i<Bones.Length; i++) {
			Quaternion update = Quaternion.Euler(Mathf.Rad2Deg*variables[i*3+0], Mathf.Rad2Deg*variables[i*3+1], Mathf.Rad2Deg*variables[i*3+2]);
			Bones[i].rotation = posture[i].GetRotation() * update;
		}
	}

	private Matrix4x4[] GetPosture() {
		Matrix4x4[] posture = new Matrix4x4[Bones.Length];
		for(int i=0; i<posture.Length; i++) {
			posture[i] = Bones[i].GetWorldMatrix();
		}
		return posture;
	}

	private void Iterate(Matrix4x4[] posture, float[] variables) {
		FK(posture, variables);
		Vector3[] tipPositions = new Vector3[Objectives.Length];
		Quaternion[] tipRotations = new Quaternion[Objectives.Length];
		for(int i=0; i<Objectives.Length; i++) {
			tipPositions[i] = Objectives[i].Tip.position;
			tipRotations[i] = Objectives[i].Tip.rotation;
		}

		int index = 0;

		//Jacobian
		for(int j=0; j<DoF; j++) {
			variables[j] += Differential;
			FK(posture, variables);
			variables[j] -= Differential;

			index = 0;
			for(int i=0; i<Objectives.Length; i++) {
				Vector3 deltaPosition = (Objectives[i].Tip.position - tipPositions[i]) / Differential;
		
				Jacobian[index][j] = deltaPosition.x; index += 1;
				Jacobian[index][j] = deltaPosition.y; index += 1;
				Jacobian[index][j] = deltaPosition.z; index += 1;
			}
		}

		//Gradient Vector
		index = 0;
		for(int i=0; i<Objectives.Length; i++) {
			Vector3 gradientPosition = Step * (Objectives[i].Goal.position + Offset - tipPositions[i]);

			Gradient[index] = gradientPosition.x; index += 1;
			Gradient[index] = gradientPosition.y; index += 1;
			Gradient[index] = gradientPosition.z; index += 1;
		}

		//Jacobian Transpose
		for(int m=0; m<DoF; m++) {
			for(int n=0; n<Entries; n++) {
				variables[m] += Jacobian[n][m] * Gradient[n];
			}
		}
		

		// //Jacobian Damped-Least-Squares
		// if(Method == JacobianMethod.DampedLeastSquares) {
		// 	float[][] DLS = DampedLeastSquares();
		// 	for(int m=0; m<DoF; m++) {
		// 		for(int n=0; n<Entries; n++) {
		// 			variables[m] += DLS[m][n] * Gradient[n];
		// 		}
		// 	}
		// }
	}

	// private float[][] DampedLeastSquares() {
	// 	float[][] transpose = Matrix.MatrixCreate(DoF, Entries);
	// 	for(int m=0; m<Entries; m++) {
	// 		for(int n=0; n<DoF; n++) {
	// 			transpose[n][m] = Jacobian[m][n];
	// 		}
	// 	}
	// 	float[][] jTj = Matrix.MatrixProduct(transpose, Jacobian);
	// 	for(int i=0; i<DoF; i++) {
	// 		jTj[i][i] += Damping*Damping;
	// 	}
	// 	float[][] dls = Matrix.MatrixProduct(Matrix.MatrixInverse(jTj), transpose);
	// 	return dls;
  	// }

}