using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class EigenNN {

    //Default
    [DllImport("EigenNN")]
    private static extern IntPtr Create(int rows, int cols);
    [DllImport("EigenNN")]
    private static extern IntPtr Delete(IntPtr T);

    //Setters and Getters
    [DllImport("EigenNN")]
    private static extern int GetRows(IntPtr T);
    [DllImport("EigenNN")]
    private static extern int GetCols(IntPtr T);
	[DllImport("EigenNN")]
    private static extern void SetZero(IntPtr T);
    [DllImport("EigenNN")]
    private static extern void SetValue(IntPtr T, int row, int col, float value);
    [DllImport("EigenNN")]
    private static extern float GetValue(IntPtr T, int row, int col);

    //Arithmetics
    [DllImport("EigenNN")]
    private static extern void Add(IntPtr lhs, IntPtr rhs, IntPtr OUT);
    [DllImport("EigenNN")]
    private static extern void Subtract(IntPtr lhs, IntPtr rhs, IntPtr OUT);
    [DllImport("EigenNN")]
    private static extern void Product(IntPtr lhs, IntPtr rhs, IntPtr OUT);
    [DllImport("EigenNN")]
    private static extern void Scale(IntPtr lhs, float value, IntPtr OUT);
    [DllImport("EigenNN")]
    private static extern void PointwiseProduct(IntPtr lhs, IntPtr rhs, IntPtr OUT);
    [DllImport("EigenNN")]
    private static extern void PointwiseQuotient(IntPtr lhs, IntPtr rhs, IntPtr OUT);

    //NN Functions
    [DllImport("EigenNN")]
    private static extern void Normalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
	[DllImport("EigenNN")]
    private static extern void Renormalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
	[DllImport("EigenNN")]
    private static extern void Layer(IntPtr IN, IntPtr W, IntPtr b, IntPtr OUT);
	[DllImport("EigenNN")]
    private static extern void Blend(IntPtr T, IntPtr W, float w);

    //NN Activations
    [DllImport("EigenNN")]
    private static extern void ELU(IntPtr T);
    [DllImport("EigenNN")]
    private static extern void Sigmoid(IntPtr T);
    [DllImport("EigenNN")]
    private static extern void TanH(IntPtr T);
    [DllImport("EigenNN")]
    private static extern void SoftMax(IntPtr T);

    public static Tensor Add(Tensor lhs, Tensor rhs, Tensor OUT) {
        Add(lhs.Ptr, rhs.Ptr, OUT.Ptr);
        return OUT;
    }

    public static Tensor Subtract(Tensor lhs, Tensor rhs, Tensor OUT) {
        Subtract(lhs.Ptr, rhs.Ptr, OUT.Ptr);
        return OUT;
    }

    public static Tensor Product(Tensor lhs, Tensor rhs, Tensor OUT) {
        Product(lhs.Ptr, rhs.Ptr, OUT.Ptr);
        return OUT;
    }

    public static Tensor Scale(Tensor lhs, float value, Tensor OUT) {
        Scale(lhs.Ptr, value, OUT.Ptr);
        return OUT;
    }

    public static Tensor PointwiseProduct(Tensor lhs, Tensor rhs, Tensor OUT) {
        PointwiseProduct(lhs.Ptr, rhs.Ptr, OUT.Ptr);
        return OUT;
    }

    public static Tensor PointwiseQuotient(Tensor lhs, Tensor rhs, Tensor OUT) {
        PointwiseQuotient(lhs.Ptr, rhs.Ptr, OUT.Ptr);
        return OUT;
    }

    public static Tensor Normalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
        Normalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
        return OUT;
    }
    
    public static Tensor Renormalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
        Renormalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
        return OUT;
    }

    public static Tensor Layer(Tensor IN, Tensor W, Tensor b, Tensor OUT) {
        Layer(IN.Ptr, W.Ptr, b.Ptr, OUT.Ptr);
        return OUT;
    }

    public static Tensor Blend(Tensor T, Tensor W, float w) {
        Blend(T.Ptr, W.Ptr, w);
        return T;
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

        public void ELU() {
            EigenNN.ELU(Ptr);
        }

        public void Sigmoid() {
            EigenNN.Sigmoid(Ptr);
        }

        public void TanH() {
            EigenNN.TanH(Ptr);
        }

        public void SoftMax() {
            EigenNN.SoftMax(Ptr);
        }

    }

}
