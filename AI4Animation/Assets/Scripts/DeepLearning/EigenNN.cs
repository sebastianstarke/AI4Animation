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
    private static extern void Sub(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void Multiply(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void Scale(IntPtr lhs, float value, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void PointwiseMultiply(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void PointwiseDivide(IntPtr lhs, IntPtr rhs, IntPtr result);
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

	private List<IntPtr> Ptrs;

	public EigenNN() {
		Ptrs = new List<IntPtr>();
	}

	~EigenNN() {
		for(int i=0; i<Ptrs.Count; i++) {
			Delete(Ptrs[i]);
		}
	}

	public IntPtr CreateMatrix(int rows, int cols) {
		IntPtr m = Create(rows, cols);
		Ptrs.Add(m);
		return m;
	}

	public void DeleteMatrix(IntPtr m) {
		Ptrs.Remove(m);
		Delete(m);
	}

}
