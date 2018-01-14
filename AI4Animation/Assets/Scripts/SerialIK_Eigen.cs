using System;
using System.Runtime.InteropServices;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SerialIK_Eigen : MonoBehaviour {

	public Vector3 GoalPosition;
	public Transform[] Transforms;

	[Range(0f, 1f)] public float Step = 1.0f;
	[Range(0f, 1f)] public float Damping = 0.1f;
	public int Iterations = 10;

	private IntPtr Solver;

    [DllImport("SerialIK")]
    private static extern IntPtr Create();
    [DllImport("SerialIK")]
    private static extern IntPtr Delete(IntPtr obj);
    [DllImport("SerialIK")]
    private static extern void Initialise(IntPtr obj, int bones, int dof, int dimensions);
    [DllImport("SerialIK")]
    private static extern void Process(IntPtr obj, int iterations);
    [DllImport("SerialIK")]
    private static extern void SetValue(IntPtr obj, int matrix, int row, int col, float value);
    [DllImport("SerialIK")]
    private static extern void SetVariable(IntPtr obj, int index, float value);
    [DllImport("SerialIK")]
    private static extern float GetVariable(IntPtr obj, int index);

	void Awake() {
		Solver = Create();
	}

	void OnDestroy() {
		Delete(Solver);
	}

	void Reset() {
		Transforms = new Transform[1] {transform};
	}

	public void UpdateGoal() {
		if(Transforms.Length == 0) {
			return;
		}

		GoalPosition = Transforms[Transforms.Length-1].position;
	}

	public void ProcessIK() {
		if(Transforms.Length == 0) {
			return;
		}

		Initialise(Solver, Transforms.Length, 3*Transforms.Length, 3);

		float height = Utility.GetHeight(GoalPosition, LayerMask.GetMask("Ground"));
		GoalPosition.y = height + (GoalPosition.y - transform.root.position.y);

		Matrix4x4[] sequence = GetSequence();
		for(int i=0; i<sequence.Length; i++) {
			for(int x=0; x<4; x++) {
				for(int y=0; y<4; y++) {
					SetValue(Solver, i, x, y, sequence[i][x,y]);
				}
			}
		}

		for(int i=0; i<3*Transforms.Length; i++) {
			SetVariable(Solver, i, 0f);
		}

		Process(Solver, Iterations);
		
		float[] solution = new float[3*Transforms.Length];
		for(int i=0; i<3*Transforms.Length; i++) {
			solution[i] = GetVariable(Solver, i);
		}

		Assign(sequence, solution);
	}

	private void Assign(Matrix4x4[] sequence, float[] variables) {
		Matrix4x4 result = Matrix4x4.identity;
		for(int i=0; i<sequence.Length; i++) {
			Matrix4x4 update = Matrix4x4.TRS(Vector3.zero, Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+0], Vector3.forward) * Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+1], Vector3.right) * Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+2], Vector3.up), Vector3.one);
			if(i == 0) {
				result = sequence[i] * update;
				Transforms[i].position = result.GetPosition();
				Transforms[i].rotation = result.GetRotation();
			} else {
				result = result * sequence[i] * update;
				Transforms[i].localPosition = result.GetPosition();
				Transforms[i].localRotation = result.GetRotation();
			}
		}
	}
	
	private Matrix4x4[] GetSequence() {
		Matrix4x4[] sequence = new Matrix4x4[Transforms.Length];
		sequence[0] = Transforms[0].GetWorldMatrix();
		for(int i=1; i<sequence.Length; i++) {
			sequence[i] = Transforms[i].GetLocalMatrix();
		}
		return sequence;
	}

	private void Iterate(Matrix4x4[] sequence, float[] variables) {
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

	private Matrix DampedLeastSquares() {
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
		return new Matrix(1, 1);
  	}

}