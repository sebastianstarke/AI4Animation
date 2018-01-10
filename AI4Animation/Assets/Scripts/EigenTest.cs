using System.Runtime.InteropServices;
using UnityEngine;
using System;

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

public class EigenTest : MonoBehaviour {
    void Start() {
        Test test = new Test();
        Debug.Log(test.Get());
        test.Update(10);
        Debug.Log(test.Get());
    }
}