using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace DeepLearning {

    public class Tensor {
        //Default
        [DllImport("DeepLearning")]
        private static extern IntPtr Create(int rows, int cols);
        [DllImport("DeepLearning")]
        private static extern IntPtr Delete(IntPtr T);

        //Setters and Getters
        [DllImport("DeepLearning")]
        private static extern int GetRows(IntPtr T);
        [DllImport("DeepLearning")]
        private static extern int GetCols(IntPtr T);
        [DllImport("DeepLearning")]
        private static extern void SetZero(IntPtr T);
        [DllImport("DeepLearning")]
        private static extern void SetValue(IntPtr T, int row, int col, float value);
        [DllImport("DeepLearning")]
        private static extern float GetValue(IntPtr T, int row, int col);

        //Arithmetics
        [DllImport("DeepLearning")]
        private static extern void Add(IntPtr lhs, IntPtr rhs, IntPtr OUT);
        [DllImport("DeepLearning")]
        private static extern void Subtract(IntPtr lhs, IntPtr rhs, IntPtr OUT);
        [DllImport("DeepLearning")]
        private static extern void Product(IntPtr lhs, IntPtr rhs, IntPtr OUT);
        [DllImport("DeepLearning")]
        private static extern void Scale(IntPtr lhs, float value, IntPtr OUT);
        [DllImport("DeepLearning")]
        private static extern void PointwiseProduct(IntPtr lhs, IntPtr rhs, IntPtr OUT);
        [DllImport("DeepLearning")]
        private static extern void PointwiseQuotient(IntPtr lhs, IntPtr rhs, IntPtr OUT);

        public IntPtr Ptr;

        private bool Deleted;
        
        public Tensor(int rows, int cols) {
            Ptr = Create(rows, cols);
            Deleted = false;
        }

        ~Tensor() {
            Delete();
        }

        public void Delete() {
            if(!Deleted) {
                Delete(Ptr);
                Deleted = true;
            }
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

        public static Tensor Add(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                Add(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor Subtract(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                Subtract(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor Product(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.GetCols() != rhs.GetRows()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else { 
                Product(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor Scale(Tensor lhs, float value, Tensor OUT) {
            Scale(lhs.Ptr, value, OUT.Ptr);
            return OUT;
        }

        public static Tensor PointwiseProduct(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                PointwiseProduct(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor PointwiseQuotient(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                PointwiseQuotient(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }
    }

}
