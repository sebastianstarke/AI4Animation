using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class EigenNN {

    [DllImport("EigenNN")]
    private static extern IntPtr Create(int rows, int cols);
    [DllImport("EigenNN")]
    private static extern IntPtr Delete(IntPtr m);
	[DllImport("EigenNN")]
    private static extern void SetZero(IntPtr m);
    [DllImport("EigenNN")]
    private static extern void Add(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void Subtract(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void Product(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void Scale(IntPtr lhs, float value, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void PointwiseProduct(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void PointwiseQuotient(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void SetValue(IntPtr m, int row, int col, float value);
    [DllImport("EigenNN")]
    private static extern float GetValue(IntPtr m, int row, int col);
	[DllImport("EigenNN")]
    private static extern void Layer(IntPtr x, IntPtr y, IntPtr W, IntPtr b);
	[DllImport("EigenNN")]
    private static extern void Blend(IntPtr m, IntPtr W, float w, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void ELU(IntPtr m);
    [DllImport("EigenNN")]
    private static extern void Sigmoid(IntPtr m);
    [DllImport("EigenNN")]
    private static extern void TanH(IntPtr m);
    [DllImport("EigenNN")]
    private static extern void SoftMax(IntPtr m);

    public static void Add(Tensor lhs, Tensor rhs, Tensor result) {
        Add(lhs.Ptr, rhs.Ptr, result.Ptr);
    }

    public static void Subtract(Tensor lhs, Tensor rhs, Tensor result) {
        Subtract(lhs.Ptr, rhs.Ptr, result.Ptr);
    }

    public static void Product(Tensor lhs, Tensor rhs, Tensor result) {
        Product(lhs.Ptr, rhs.Ptr, result.Ptr);
    }

    public static void Scale(Tensor lhs, float value, Tensor result) {
        Scale(lhs.Ptr, value, result.Ptr);
    }

    public class Tensor {

        public IntPtr Ptr;
        
        public Tensor(int rows, int cols) {
            Ptr = EigenNN.Create(rows, cols);
        }

        ~Tensor() {
            EigenNN.Delete(Ptr);
        }

        public void SetValue(int row, int col, float value) {
            EigenNN.SetValue(Ptr, row, col, value);
        }

        public float GetValue(int row, int col) {
            return EigenNN.GetValue(Ptr, row, col);
        }

    }

}
