using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace DeepLearning {

    public class Tensor {
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

        //Activations
        [DllImport("EigenNN")]
        private static extern void ELU(IntPtr T);
        [DllImport("EigenNN")]
        private static extern void Sigmoid(IntPtr T);
        [DllImport("EigenNN")]
        private static extern void TanH(IntPtr T);
        [DllImport("EigenNN")]
        private static extern void SoftMax(IntPtr T);

        public IntPtr Ptr;
        
        public Tensor(int rows, int cols) {
            System.GC.Collect();
            Ptr = Create(rows, cols);
        }

        ~Tensor() {
            Delete(Ptr);
        }

        public int GetRows() {
            return GetRows(Ptr);
        }

        public int GetCols() {
            return GetCols(Ptr);
        }

        public void SetZero() {
            SetZero(Ptr);
        }

        public void SetValue(int row, int col, float value) {
            if(row >= GetRows() || col >= GetCols()) {
                Debug.Log("Accessing out of bounds.");
                return;
            }
            SetValue(Ptr, row, col, value);
        }

        public float GetValue(int row, int col) {
            if(row >= GetRows() || col >= GetCols()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return GetValue(Ptr, row, col);
        }

        public void ELU() {
            ELU(Ptr);
        }

        public void Sigmoid() {
            Sigmoid(Ptr);
        }

        public void TanH() {
            TanH(Ptr);
        }

        public void SoftMax() {
            SoftMax(Ptr);
        }
    }

}
