using System;

public class Vector {
	public float[] Values;

	public Vector(int dim) {
		Values = new float[dim];
	}

	public Vector(float[] values) {
		Values = values;
	}
}

public class Matrix {
	public float[][] Values;

	public Matrix(int rows, int cols) {
		Values = new float[rows][];
		for(int i=0; i<rows; i++) {
			Values[i] = new float[cols];
		}
	}

	public Matrix(float[][] values) {
		Values = values;
	}

	public int GetRows() {
		return Values.Length;
	}
	
	public int GetColumns() {
		return Values[0].Length;
	}

	public static Matrix operator+ (Matrix a, Matrix b) {
		Matrix result = new Matrix(a.Values.Length, a.Values[0].Length);
		for(int i=0; i<result.Values.Length; i++) {
			for(int j=0; j<result.Values[i].Length; j++) {
				result.Values[i][j] = a.Values[i][j] + b.Values[i][j];
			}
		}
		return result;
	}

	public static Matrix operator- (Matrix a, Matrix b) {
		Matrix result = new Matrix(a.Values.Length, a.Values[0].Length);
		for(int i=0; i<result.Values.Length; i++) {
			for(int j=0; j<result.Values[i].Length; j++) {
				result.Values[i][j] = a.Values[i][j] - b.Values[i][j];
			}
		}
		return result;
	}

	public static Matrix operator* (float value, Matrix m) {
		Matrix result = new Matrix(m.Values.Length, m.Values[0].Length);
		for(int i=0; i<result.Values.Length; i++) {
			for(int j=0; j<result.Values[i].Length; j++) {
				result.Values[i][j] = value * m.Values[i][j];
			}
		}
		return result;
	}

	public static Matrix operator* (Matrix m, float value) {
		Matrix result = new Matrix(m.Values.Length, m.Values[0].Length);
		for(int i=0; i<result.Values.Length; i++) {
			for(int j=0; j<result.Values[i].Length; j++) {
				result.Values[i][j] = m.Values[i][j] * value;
			}
		}
		return result;
	}

	public static Matrix operator* (Matrix a, Matrix b) {
		return new Matrix(MatrixUtility.MatrixProduct(a.Values, b.Values));
	}

	public Matrix GetInverse() {
		return new Matrix(MatrixUtility.MatrixInverse(Values));
	}

	public Matrix PointwiseMultiply(Matrix m) {
		Matrix result = new Matrix(Values.Length, Values[0].Length);
		for(int i=0; i<result.Values.Length; i++) {
			for(int j=0; j<result.Values[i].Length; j++) {
				result.Values[i][j] = Values[i][j] * m.Values[i][j];
			}
		}
		return result;
	}

	public Matrix PointwiseDivide(Matrix m) {
		Matrix result = new Matrix(Values.Length, Values[0].Length);
		for(int i=0; i<result.Values.Length; i++) {
			for(int j=0; j<result.Values[i].Length; j++) {
				result.Values[i][j] = Values[i][j] / m.Values[i][j];
			}
		}
		return result;
	}
}

public static class MatrixUtility {

	public static float[][] MatrixInverse(float[][] matrix)
	{
	// assumes determinant is not 0
	// that is, the matrix does have an inverse
	int n = matrix.Length;
	float[][] result = MatrixCreate(n, n); // make a copy of matrix
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
		result[i][j] = matrix[i][j];

	float[][] lum; // combined lower & upper
	int[] perm;
	MatrixDecompose(matrix, out lum, out perm);

	float[] b = new float[n];
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		if (i == perm[j])
			b[j] = 1f;
		else
			b[j] = 0f;

		float[] x = Helper(lum, b); // 
		for (int j = 0; j < n; ++j)
		result[j][i] = x[j];
	}
	return result;
	} // MatrixInverse

	public static int MatrixDecompose(float[][] m, out float[][] lum, out int[] perm)
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
		float max = Math.Abs(lum[j][j]);
		int piv = j;

		for (int i = j + 1; i < n; ++i) // find pivot index
		{
		float xij = Math.Abs(lum[i][j]);
		if (xij > max)
		{
			max = xij;
			piv = i;
		}
		} // i

		if (piv != j)
		{
		float[] tmp = lum[piv]; // swap rows j, piv
		lum[piv] = lum[j];
		lum[j] = tmp;

		int t = perm[piv]; // swap perm elements
		perm[piv] = perm[j];
		perm[j] = t;

		toggle = -toggle;
		}

		float xjj = lum[j][j];
		if (xjj != 0.0)
		{
		for (int i = j + 1; i < n; ++i)
		{
			float xij = lum[i][j] / xjj;
			lum[i][j] = xij;
			for (int k = j + 1; k < n; ++k)
			lum[i][k] -= xij * lum[j][k];
		}
		}

	} // j

	return toggle;
	} // MatrixDecompose

	public static float[] Helper(float[][] luMatrix, float[] b) // helper
	{
	int n = luMatrix.Length;
	float[] x = new float[n];
	b.CopyTo(x, 0);

	for (int i = 1; i < n; ++i)
	{
		float sum = x[i];
		for (int j = 0; j < i; ++j)
		sum -= luMatrix[i][j] * x[j];
		x[i] = sum;
	}

	x[n - 1] /= luMatrix[n - 1][n - 1];
	for (int i = n - 2; i >= 0; --i)
	{
		float sum = x[i];
		for (int j = i + 1; j < n; ++j)
		sum -= luMatrix[i][j] * x[j];
		x[i] = sum / luMatrix[i][i];
	}

	return x;
	} // Helper

	public static float MatrixDeterminant(float[][] matrix)
	{
	float[][] lum;
	int[] perm;
	int toggle = MatrixDecompose(matrix, out lum, out perm);
	float result = toggle;
	for (int i = 0; i < lum.Length; ++i)
		result *= lum[i][i];
	return result;
	}

	// ----------------------------------------------------------------

	public static float[][] MatrixCreate(int rows, int cols)
	{
	float[][] result = new float[rows][];
	for (int i = 0; i < rows; ++i)
		result[i] = new float[cols];
	return result;
	}

	public static float[][] MatrixProduct(float[][] matrixA,
	float[][] matrixB)
	{
	int aRows = matrixA.Length;
	int aCols = matrixA[0].Length;
	int bRows = matrixB.Length;
	int bCols = matrixB[0].Length;
	if (aCols != bRows)
		throw new Exception("Non-conformable matrices");

	float[][] result = MatrixCreate(aRows, bCols);

	for (int i = 0; i < aRows; ++i) // each row of A
		for (int j = 0; j < bCols; ++j) // each col of B
		for (int k = 0; k < aCols; ++k) // could use k < bRows
			result[i][j] += matrixA[i][k] * matrixB[k][j];

	return result;
	}

	public static string MatrixAsString(float[][] matrix)
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

	public static float[][] ExtractLower(float[][] lum)
	{
	// lower part of an LU Doolittle decomposition (dummy 1.0s on diagonal, 0.0s above)
	int n = lum.Length;
	float[][] result = MatrixCreate(n, n);
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
		if (i == j)
			result[i][j] = 1f;
		else if (i > j)
			result[i][j] = lum[i][j];
		}
	}
	return result;
	}

	public static float[][] ExtractUpper(float[][] lum)
	{
	// upper part of an LU (lu values on diagional and above, 0.0s below)
	int n = lum.Length;
	float[][] result = MatrixCreate(n, n);
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