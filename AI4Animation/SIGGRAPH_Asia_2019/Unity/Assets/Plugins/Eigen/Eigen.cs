using System;
using System.Runtime.InteropServices;

//Eigen Plugin
public static class Eigen {
    //Default
    [DllImport("Eigen")]
    public static extern IntPtr Create(int rows, int cols);
    [DllImport("Eigen")]
    public static extern IntPtr Delete(IntPtr ptr);

    //Setters and Getters
    [DllImport("Eigen")]
    public static extern int GetRows(IntPtr ptr);
    [DllImport("Eigen")]
    public static extern int GetCols(IntPtr ptr);
    [DllImport("Eigen")]
    public static extern void SetZero(IntPtr ptr);
    [DllImport("Eigen")]
    public static extern void SetSize(IntPtr ptr, int rows, int cols);
    [DllImport("Eigen")]
    public static extern void SetValue(IntPtr ptr, int row, int col, float value);
    [DllImport("Eigen")]
    public static extern float GetValue(IntPtr ptr, int row, int col);

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
    public static extern float RowSum(IntPtr ptr, int row);
    [DllImport("Eigen")]
    public static extern float ColSum(IntPtr ptr, int col);
    [DllImport("Eigen")]
    public static extern float RowMean(IntPtr ptr, int row);
    [DllImport("Eigen")]
    public static extern float ColMean(IntPtr ptr, int col);
    [DllImport("Eigen")]
    public static extern float RowStd(IntPtr ptr, int row);
    [DllImport("Eigen")]
    public static extern float ColStd(IntPtr ptr, int col);

    //Deep Learning Functions
    [DllImport("Eigen")]
    public static extern void Normalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void Renormalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void Layer(IntPtr IN, IntPtr W, IntPtr b, IntPtr OUT);
    [DllImport("Eigen")]
    public static extern void Blend(IntPtr ptr, IntPtr W, float w);
    [DllImport("Eigen")]
    public static extern void BlendAll(IntPtr ptr, IntPtr[] W, float[] w, int length);
    [DllImport("Eigen")]
    public static extern void ELU(IntPtr ptr);
    [DllImport("Eigen")]
    public static extern void Sigmoid(IntPtr ptr);
    [DllImport("Eigen")]
    public static extern void TanH(IntPtr ptr);
    [DllImport("Eigen")]
    public static extern void SoftMax(IntPtr ptr);
    [DllImport("Eigen")]
    public static extern void LogSoftMax(IntPtr ptr);
    [DllImport("Eigen")]
    public static extern void SoftSign(IntPtr ptr);
    [DllImport("Eigen")]
    public static extern void Exp(IntPtr ptr);
}
