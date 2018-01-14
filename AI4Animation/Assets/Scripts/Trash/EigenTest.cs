using System.Runtime.InteropServices;
using UnityEngine;
using System;

public class EigenTest : MonoBehaviour {

    void Start() {
        
        int dim = 100;
        int iter = 1;

        System.DateTime t1 = Utility.GetTimestamp();
        IntPtr A = Create(dim, dim);
        IntPtr B = Create(dim, dim);
        IntPtr C = Create(dim, dim);
        for(int i=0; i<iter; i++) {
            Multiply(A, B, C);
        }
        Debug.Log("Time Eigen C# / C++: " + Utility.GetElapsedTime(t1) + "s");

        System.DateTime t2 = Utility.GetTimestamp();
        Matrix a = new Matrix(dim, dim);
        Matrix b = new Matrix(dim, dim);
        for(int i=0; i<iter; i++) {
            Matrix c = a * b;
        }
        Debug.Log("Time Matrix C#: " + Utility.GetElapsedTime(t2) + "s");

        System.DateTime t3 = Utility.GetTimestamp();
        Performance(dim, dim, iter);
        Debug.Log("Time Eigen C++: " + Utility.GetElapsedTime(t3) + "s");
        
        /*
        System.DateTime t1 = Utility.GetTimestamp();
        IntPtr a = Create(10000, 10000);
        SoftMax(a);
        Debug.Log("Eigen: " + Utility.GetElapsedTime(t1));

        System.DateTime t2 = Utility.GetTimestamp();
        Matrix b = new Matrix(10000, 10000);
        SoftMax(ref b);
        Debug.Log("Unity: " + Utility.GetElapsedTime(t2));
        */
    }

	private void ELU(ref Matrix m) {
		for(int x=0; x<m.Values.Length; x++) {
			m.Values[x][0] = System.Math.Max(m.Values[x][0], 0f) + (float)System.Math.Exp(System.Math.Min(m.Values[x][0], 0f)) - 1f;
		}
	}

	private void SoftMax(ref Matrix m) {
        float lower = 0f;
		for(int x=0; x<m.Values.Length; x++) {
			m.Values[x][0] = (float)System.Math.Exp(m.Values[x][0]);
            lower += m.Values[x][0];
		}
		for(int x=0; x<m.Values.Length; x++) {
			m.Values[x][0] /= lower;
		}
	}

    [DllImport("Eigen")]
    private static extern IntPtr Create(int rows, int cols);
    [DllImport("Eigen")]
    private static extern IntPtr Delete(IntPtr m);
    [DllImport("Eigen")]
    private static extern void Add(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("Eigen")]
    private static extern void Multiply(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("Eigen")]
    private static extern void SetValue(IntPtr m, int row, int col, float value);
    [DllImport("Eigen")]
    private static extern float GetValue(IntPtr m, int row, int col);
    [DllImport("Eigen")]
    private static extern void ELU(IntPtr m);
    [DllImport("Eigen")]
    private static extern void SoftMax(IntPtr m);

    [DllImport("Eigen")]
    private static extern void Performance(int rows, int cols, int iterations);

}





/*
public class Test {
    private IntPtr Ptr = IntPtr.Zero;
    public Test() {
        Ptr = Create();
    }

    ~Test() {
        Destroy();
    }
    
    public void Destroy() {
        Destroy(Ptr);
    }
    
    public void Update(int value) {
        UpdateValue(Ptr, value);
    }

    public int Get() {
        return GetValue(Ptr);
    }

    [DllImport("Eigen")]
    private static extern IntPtr Create();
    [DllImport("Eigen")]
    private static extern void Destroy(IntPtr obj);
    [DllImport("Eigen")]
    private static extern void UpdateValue(IntPtr obj, int value);
    [DllImport("Eigen")]
    private static extern int GetValue(IntPtr obj);
}
*/

//Test test = new Test();
/*
IntPtr A = GetMatrix(4,4);
SetMatrixValue(A, 0, 0, UnityEngine.Random.value);
IntPtr B = GetMatrix(4,4);
SetMatrixValue(B, 0, 0, UnityEngine.Random.value);
IntPtr C = Multiply(A,B);
Debug.Log(GetMatrixValue(C, 0, 0));
*/
