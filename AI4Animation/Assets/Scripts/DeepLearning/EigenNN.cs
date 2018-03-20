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
    private static extern int GetRows(IntPtr m);
    [DllImport("EigenNN")]
    private static extern int GetCols(IntPtr m);
	[DllImport("EigenNN")]
    private static extern void SetZero(IntPtr m);
    [DllImport("EigenNN")]
    private static extern void SetValue(IntPtr m, int row, int col, float value);
    [DllImport("EigenNN")]
    private static extern float GetValue(IntPtr m, int row, int col);

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
    private static extern void Normalise(IntPtr m, IntPtr mean, IntPtr std, IntPtr result);
	[DllImport("EigenNN")]
    private static extern void Renormalise(IntPtr m, IntPtr mean, IntPtr std, IntPtr result);
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

    public static Tensor Add(Tensor lhs, Tensor rhs, Tensor result) {
        Add(lhs.Ptr, rhs.Ptr, result.Ptr);
        return result;
    }

    public static Tensor Subtract(Tensor lhs, Tensor rhs, Tensor result) {
        Subtract(lhs.Ptr, rhs.Ptr, result.Ptr);
        return result;
    }

    public static Tensor Product(Tensor lhs, Tensor rhs, Tensor result) {
        Product(lhs.Ptr, rhs.Ptr, result.Ptr);
        return result;
    }

    public static Tensor Scale(Tensor lhs, float value, Tensor result) {
        Scale(lhs.Ptr, value, result.Ptr);
        return result;
    }

    public static Tensor PointwiseProduct(Tensor lhs, Tensor rhs, Tensor result) {
        PointwiseProduct(lhs.Ptr, rhs.Ptr, result.Ptr);
        return result;
    }

    public static Tensor PointwiseQuotient(Tensor lhs, Tensor rhs, Tensor result) {
        PointwiseQuotient(lhs.Ptr, rhs.Ptr, result.Ptr);
        return result;
    }

    public class Tensor {

        public IntPtr Ptr;
        
        public Tensor(int rows, int cols) {
            System.GC.Collect();
            Ptr = EigenNN.Create(rows, cols);
        }

        ~Tensor() {
            EigenNN.Delete(Ptr);
        }

        public int GetRows() {
            return EigenNN.GetRows(Ptr);
        }

        public int GetCols() {
            return EigenNN.GetCols(Ptr);
        }

        public void SetZero() {
            EigenNN.SetZero(Ptr);
        }

        public void SetValue(int row, int col, float value) {
            if(row >= GetRows() || col >= GetCols()) {
                Debug.Log("Accessing out of bounds.");
                return;
            }
            EigenNN.SetValue(Ptr, row, col, value);
        }

        public float GetValue(int row, int col) {
            if(row >= GetRows() || col >= GetCols()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return EigenNN.GetValue(Ptr, row, col);
        }

    }

}
