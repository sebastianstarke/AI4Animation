using System;
using UnityEngine;

namespace DeepLearning {

    public class Matrix {

        public IntPtr Ptr;
        public string ID;

        private bool Deleted;
        
        public Matrix(int rows, int cols, string id = "") {
            Ptr = Eigen.Create(rows, cols);
            ID = id;
            Deleted = false;
        }

        ~Matrix() {
            Delete();
        }

        public void Delete() {
            if(!Deleted) {
                Eigen.Delete(Ptr);
                Deleted = true;
            }
        }

        public int GetRows() {
            return Eigen.GetRows(Ptr);
        }

        public int GetCols() {
            return Eigen.GetCols(Ptr);
        }

        public void SetZero() {
            Eigen.SetZero(Ptr);
        }

        public void SetSize(int rows, int cols) {
            Eigen.SetSize(Ptr, rows, cols);
        }

        public void SetValue(int row, int col, float value) {
            if(row >= GetRows() || col >= GetCols()) {
                Debug.Log("Setting out of bounds at [" + row + ", " + col + "] in matrix " + ID + ".");
                return;
            }
            Eigen.SetValue(Ptr, row, col, value);
        }

        public float GetValue(int row, int col) {
            if(row >= GetRows() || col >= GetCols()) {
                Debug.Log("Getting out of bounds at [" + row + ", " + col + "] in matrix " + ID + ".");
                return 0f;
            }
            return Eigen.GetValue(Ptr, row, col);
        }

        public static Matrix Add(Matrix lhs, Matrix rhs, Matrix OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible Matrix dimensions.");
            } else {
                Eigen.Add(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Matrix Subtract(Matrix lhs, Matrix rhs, Matrix OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible Matrix dimensions.");
            } else {
                Eigen.Subtract(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Matrix Product(Matrix lhs, Matrix rhs, Matrix OUT) {
            if(lhs.GetCols() != rhs.GetRows()) {
                Debug.Log("Incompatible Matrix dimensions.");
            } else { 
                Eigen.Product(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Matrix Scale(Matrix lhs, float value, Matrix OUT) {
            Eigen.Scale(lhs.Ptr, value, OUT.Ptr);
            return OUT;
        }

        public static Matrix PointwiseProduct(Matrix lhs, Matrix rhs, Matrix OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible Matrix dimensions.");
            } else {
                Eigen.PointwiseProduct(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Matrix PointwiseQuotient(Matrix lhs, Matrix rhs, Matrix OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible Matrix dimensions.");
            } else {
                Eigen.PointwiseQuotient(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Matrix PointwiseAbsolute(Matrix IN, Matrix OUT) {
            Eigen.PointwiseAbsolute(IN.Ptr, OUT.Ptr);
            return OUT;
        }

        public float RowMean(int row) {
            if(row >= GetRows()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.RowMean(Ptr, row);
        }

        public float ColMean(int col) {
            if(col >= GetCols()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.ColMean(Ptr, col);
        }

        public float RowStd(int row) {
            if(row >= GetRows()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.RowStd(Ptr, row);
        }

        public float ColStd(int col) {
            if(col >= GetCols()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.ColStd(Ptr, col);
        }

        public float RowSum(int row) {
            if(row >= GetRows()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.RowSum(Ptr, row);
        }

        public float ColSum(int col) {
            if(col >= GetCols()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.ColSum(Ptr, col);
        }

        public float[] Flatten() {
            int rows = GetRows();
            int cols = GetCols();
            float[] array = new float[rows * cols];
            for(int i=0; i<rows; i++) {
                for(int j=0; j<cols; j++) {
                    array[i*cols + j] = GetValue(i, j);
                }
            }
            return array;
        }

        public void Unit() {
            int rows = GetRows();
            int cols = GetCols();
            float magnitude = 0f;
            for(int i=0; i<rows; i++) {
                for(int j=0; j<cols; j++) {
                    float value = GetValue(i, j);
                    magnitude += value*value;
                }
            }
            magnitude = Mathf.Sqrt(magnitude);
            if(magnitude != 0f) {
                Eigen.Scale(Ptr, 1f/magnitude, Ptr);
            }
        }

        public void ELU() {
            Eigen.ELU(Ptr);
        }

        public void Sigmoid() {
            Eigen.Sigmoid(Ptr);
        }

        public void TanH() {
            Eigen.TanH(Ptr);
        }

        public void SoftMax() {
            Eigen.SoftMax(Ptr);
        }

        public void LogSoftMax() {
            Eigen.LogSoftMax(Ptr);
        }

        public void SoftSign() {
            Eigen.SoftSign(Ptr);
        }

        public void Exp() {
            Eigen.Exp(Ptr);
        }

        public void Print() {
            string output = string.Empty;
            for(int i=0; i<GetRows(); i++) {
                for(int j=0; j<GetCols(); j++) {
                    output += GetValue(i, j) + " "; 
                }
                output += "\n";
            }
            Debug.Log(output);
        }
    }

}
