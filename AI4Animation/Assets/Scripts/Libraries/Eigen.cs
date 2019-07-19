using System;
using System.Runtime.InteropServices;

public static class Eigen {
    //Default
    [DllImport("Eigen")]
    public static extern IntPtr Create(int rows, int cols);
    [DllImport("Eigen")]
    public static extern IntPtr Delete(IntPtr T);

    //Setters and Getters
    [DllImport("Eigen")]
    public static extern int GetRows(IntPtr T);
    [DllImport("Eigen")]
    public static extern int GetCols(IntPtr T);
    [DllImport("Eigen")]
    public static extern void SetZero(IntPtr T);
    [DllImport("Eigen")]
    public static extern void SetSize(IntPtr T, int rows, int cols);
    [DllImport("Eigen")]
    public static extern void SetValue(IntPtr T, int row, int col, float value);
    [DllImport("Eigen")]
    public static extern float GetValue(IntPtr T, int row, int col);

    //Arithmetics
    [DllImport("Eigen")]
    public static extern void Add(IntPtr lhs, IntPtr rhs, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void Subtract(IntPtr lhs, IntPtr rhs, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void Product(IntPtr lhs, IntPtr rhs, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void Scale(IntPtr lhs, float value, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void PointwiseProduct(IntPtr lhs, IntPtr rhs, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void PointwiseQuotient(IntPtr lhs, IntPtr rhs, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void PointwiseAbsolute(IntPtr IN, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern float RowSum(IntPtr T, int row);
    [DllImport("Eigen")]
    public static extern float ColSum(IntPtr T, int col);
    [DllImport("Eigen")]
    public static extern float RowMean(IntPtr T, int row);
    [DllImport("Eigen")]
    public static extern float ColMean(IntPtr T, int col);
    [DllImport("Eigen")]
    public static extern float RowStd(IntPtr T, int row);
    [DllImport("Eigen")]
    public static extern float ColStd(IntPtr T, int col);

    //Deep Learning Functions
    [DllImport("Eigen")]
    public static extern void Normalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void Renormalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void Layer(IntPtr IN, IntPtr W, IntPtr b, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void Blend(IntPtr T, IntPtr W, float w);
    [DllImport("Eigen")]
    public static extern void ELU(IntPtr T);
    [DllImport("Eigen")]
    public static extern void Sigmoid(IntPtr T);
    [DllImport("Eigen")]
    public static extern void TanH(IntPtr T);
    [DllImport("Eigen")]
    public static extern void SoftMax(IntPtr T);
    [DllImport("Eigen")]
    public static extern void LogSoftMax(IntPtr T);
    [DllImport("Eigen")]
    public static extern void SoftSign(IntPtr T);
    [DllImport("Eigen")]
    public static extern void Exp(IntPtr T);
}
