using System;
using UnityEngine;

namespace DeepLearning {

    public class Tensor {

        public IntPtr Ptr;
        public string ID;

        private bool Deleted;
        
        public Tensor(int rows, int cols, string id = "") {
            Ptr = Eigen.Create(rows, cols);
            ID = id;
            Deleted = false;
        }

        ~Tensor() {
            Delete();
        }

        public void Delete() {
            if(!Deleted) {
                Eigen.Delete(Ptr);
                Deleted = true;
            }
        }

        public int Rows() {
            return Eigen.Rows(Ptr);
        }

        public int Cols() {
            return Eigen.Cols(Ptr);
        }

        public void SetZero() {
            Eigen.SetZero(Ptr);
        }

        public void SetSize(int rows, int cols) {
            Eigen.SetSize(Ptr, rows, cols);
        }

        public void SetValue(int row, int col, float value) {
            if(row >= Rows() || col >= Cols()) {
                Debug.Log("Setting out of bounds at [" + row + ", " + col + "].");
                return;
            }
            Eigen.SetValue(Ptr, row, col, value);
        }

        public float GetValue(int row, int col) {
            if(row >= Rows() || col >= Cols()) {
                Debug.Log("Getting out of bounds at [" + row + ", " + col + "].");
                return 0f;
            }
            return Eigen.GetValue(Ptr, row, col);
        }

        public static Tensor Add(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.Rows() != rhs.Rows() || lhs.Cols() != rhs.Cols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                Eigen.Add(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor Subtract(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.Rows() != rhs.Rows() || lhs.Cols() != rhs.Cols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                Eigen.Subtract(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor Product(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.Cols() != rhs.Rows()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else { 
                Eigen.Product(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor Scale(Tensor lhs, float value, Tensor OUT) {
            Eigen.Scale(lhs.Ptr, value, OUT.Ptr);
            return OUT;
        }

        public static Tensor PointwiseProduct(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.Rows() != rhs.Rows() || lhs.Cols() != rhs.Cols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                Eigen.PointwiseProduct(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor PointwiseQuotient(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.Rows() != rhs.Rows() || lhs.Cols() != rhs.Cols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                Eigen.PointwiseQuotient(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor PointwiseAbsolute(Tensor IN, Tensor OUT) {
            Eigen.PointwiseAbsolute(IN.Ptr, OUT.Ptr);
            return OUT;
        }

        public float RowMean(int row) {
            if(row >= Rows()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.RowMean(Ptr, row);
        }

        public float ColMean(int col) {
            if(col >= Cols()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.ColMean(Ptr, col);
        }

        public float RowStd(int row) {
            if(row >= Rows()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.RowStd(Ptr, row);
        }

        public float ColStd(int col) {
            if(col >= Cols()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.ColStd(Ptr, col);
        }

        public float RowSum(int row) {
            if(row >= Rows()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.RowSum(Ptr, row);
        }

        public float ColSum(int col) {
            if(col >= Cols()) {
                Debug.Log("Accessing out of bounds.");
                return 0f;
            }
            return Eigen.ColSum(Ptr, col);
        }

        public void Print() {
            string output = string.Empty;
            for(int i=0; i<Rows(); i++) {
                for(int j=0; j<Cols(); j++) {
                    output += GetValue(i, j) + " "; 
                }
                output += "\n";
            }
            Debug.Log(output);
        }
    }

}
