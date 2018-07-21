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

        public void Resize(int rows, int cols) {
            Eigen.Resize(Ptr, rows, cols);
        }

        public void ConservativeResize(int rows, int cols) {
            Eigen.ConservativeResize(Ptr, rows, cols);
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

        public void SetValue(int row, int col, float value) {
            if(row >= GetRows() || col >= GetCols()) {
                Debug.Log("Setting out of bounds at [" + row + ", " + col + "].");
                return;
            }
            Eigen.SetValue(Ptr, row, col, value);
        }

        public float GetValue(int row, int col) {
            if(row >= GetRows() || col >= GetCols()) {
                Debug.Log("Getting out of bounds at [" + row + ", " + col + "].");
                return 0f;
            }
            return Eigen.GetValue(Ptr, row, col);
        }

        public static Tensor Add(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                Eigen.Add(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor Subtract(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                Eigen.Subtract(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor Product(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.GetCols() != rhs.GetRows()) {
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
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
                Debug.Log("Incompatible tensor dimensions.");
            } else {
                Eigen.PointwiseProduct(lhs.Ptr, rhs.Ptr, OUT.Ptr);
            }
            return OUT;
        }

        public static Tensor PointwiseQuotient(Tensor lhs, Tensor rhs, Tensor OUT) {
            if(lhs.GetRows() != rhs.GetRows() || lhs.GetCols() != rhs.GetCols()) {
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
    }

}
