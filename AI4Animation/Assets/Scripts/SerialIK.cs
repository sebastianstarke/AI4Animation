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
	[Range(0f, 1f)] public double Step = 1.0;
	[Range(0f, 1f)] public double Damping = 0.1;

	private int DoF;
	private int Entries;
	private double[][] Jacobian;
	private double[] Gradient;

	private double Differential = 0.001;

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
			double[] solution = new double[3*Transforms.Length];
			DoF = Transforms.Length * 3;
			Entries = 3;
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
		float height = Utility.GetHeight(Goal.position, LayerMask.GetMask("Ground"));
		//if(height > Goal.position.y - transform.root.position.y) {
			Goal.position = new Vector3(Goal.position.x, height + (Goal.position.y - transform.root.position.y), Goal.position.z);
			return true;
		//}
		//return false;
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
	
			Jacobian[0][j] = deltaPosition.x;
			Jacobian[1][j] = deltaPosition.y;
			Jacobian[2][j] = deltaPosition.z;
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


	class Matrix {
		public static double[][] MatrixInverse(double[][] matrix)
		{
		// assumes determinant is not 0
		// that is, the matrix does have an inverse
		int n = matrix.Length;
		double[][] result = MatrixCreate(n, n); // make a copy of matrix
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
			result[i][j] = matrix[i][j];

		double[][] lum; // combined lower & upper
		int[] perm;
		MatrixDecompose(matrix, out lum, out perm);

		double[] b = new double[n];
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			if (i == perm[j])
				b[j] = 1.0;
			else
				b[j] = 0.0;
	
			double[] x = Helper(lum, b); // 
			for (int j = 0; j < n; ++j)
			result[j][i] = x[j];
		}
		return result;
		} // MatrixInverse

		public static int MatrixDecompose(double[][] m, out double[][] lum, out int[] perm)
		{
		// Crout's LU decomposition for matrix determinant and inverse
		// stores combined lower & upper in lum[][]
		// stores row permuations into perm[]
		// returns +1 or -1 according to even or odd number of row permutations
		// lower gets dummy 1.0s on diagonal (0.0s above)
		// upper gets lum values on diagonal (0.0s below)

		int toggle = +1; // even (+1) or odd (-1) row permutatuions
		int n = m.Length;

		// make a copy of m[][] into result lu[][]
		lum = MatrixCreate(n, n);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
			lum[i][j] = m[i][j];


		// make perm[]
		perm = new int[n];
		for (int i = 0; i < n; ++i)
			perm[i] = i;

		for (int j = 0; j < n - 1; ++j) // process by column. note n-1 
		{
			double max = Math.Abs(lum[j][j]);
			int piv = j;

			for (int i = j + 1; i < n; ++i) // find pivot index
			{
			double xij = Math.Abs(lum[i][j]);
			if (xij > max)
			{
				max = xij;
				piv = i;
			}
			} // i

			if (piv != j)
			{
			double[] tmp = lum[piv]; // swap rows j, piv
			lum[piv] = lum[j];
			lum[j] = tmp;

			int t = perm[piv]; // swap perm elements
			perm[piv] = perm[j];
			perm[j] = t;

			toggle = -toggle;
			}

			double xjj = lum[j][j];
			if (xjj != 0.0)
			{
			for (int i = j + 1; i < n; ++i)
			{
				double xij = lum[i][j] / xjj;
				lum[i][j] = xij;
				for (int k = j + 1; k < n; ++k)
				lum[i][k] -= xij * lum[j][k];
			}
			}

		} // j

		return toggle;
		} // MatrixDecompose

		public static double[] Helper(double[][] luMatrix, double[] b) // helper
		{
		int n = luMatrix.Length;
		double[] x = new double[n];
		b.CopyTo(x, 0);

		for (int i = 1; i < n; ++i)
		{
			double sum = x[i];
			for (int j = 0; j < i; ++j)
			sum -= luMatrix[i][j] * x[j];
			x[i] = sum;
		}

		x[n - 1] /= luMatrix[n - 1][n - 1];
		for (int i = n - 2; i >= 0; --i)
		{
			double sum = x[i];
			for (int j = i + 1; j < n; ++j)
			sum -= luMatrix[i][j] * x[j];
			x[i] = sum / luMatrix[i][i];
		}

		return x;
		} // Helper

		public static double MatrixDeterminant(double[][] matrix)
		{
		double[][] lum;
		int[] perm;
		int toggle = MatrixDecompose(matrix, out lum, out perm);
		double result = toggle;
		for (int i = 0; i < lum.Length; ++i)
			result *= lum[i][i];
		return result;
		}

		// ----------------------------------------------------------------

		public static double[][] MatrixCreate(int rows, int cols)
		{
		double[][] result = new double[rows][];
		for (int i = 0; i < rows; ++i)
			result[i] = new double[cols];
		return result;
		}

		public static double[][] MatrixProduct(double[][] matrixA,
		double[][] matrixB)
		{
		int aRows = matrixA.Length;
		int aCols = matrixA[0].Length;
		int bRows = matrixB.Length;
		int bCols = matrixB[0].Length;
		if (aCols != bRows)
			throw new Exception("Non-conformable matrices");

		double[][] result = MatrixCreate(aRows, bCols);

		for (int i = 0; i < aRows; ++i) // each row of A
			for (int j = 0; j < bCols; ++j) // each col of B
			for (int k = 0; k < aCols; ++k) // could use k < bRows
				result[i][j] += matrixA[i][k] * matrixB[k][j];

		return result;
		}

		public static string MatrixAsString(double[][] matrix)
		{
		string s = "";
		for (int i = 0; i < matrix.Length; ++i)
		{
			for (int j = 0; j < matrix[i].Length; ++j)
			s += matrix[i][j].ToString("F3").PadLeft(8) + " ";
			s += Environment.NewLine;
		}
		return s;
		}

		public static double[][] ExtractLower(double[][] lum)
		{
		// lower part of an LU Doolittle decomposition (dummy 1.0s on diagonal, 0.0s above)
		int n = lum.Length;
		double[][] result = MatrixCreate(n, n);
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
			if (i == j)
				result[i][j] = 1.0;
			else if (i > j)
				result[i][j] = lum[i][j];
			}
		}
		return result;
		}

		public static double[][] ExtractUpper(double[][] lum)
		{
		// upper part of an LU (lu values on diagional and above, 0.0s below)
		int n = lum.Length;
		double[][] result = MatrixCreate(n, n);
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
			if (i <= j)
				result[i][j] = lum[i][j];
			}
		}
		return result;
		}
	}

}