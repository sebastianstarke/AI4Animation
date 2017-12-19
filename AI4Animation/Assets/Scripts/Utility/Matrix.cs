using System;

public class Matrix {
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