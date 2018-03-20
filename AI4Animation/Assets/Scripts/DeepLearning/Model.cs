using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace DeepLearning {
    
    public class Model {
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

        //Functions
        [DllImport("EigenNN")]
        private static extern void Normalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
        [DllImport("EigenNN")]
        private static extern void Renormalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
        [DllImport("EigenNN")]
        private static extern void Layer(IntPtr IN, IntPtr W, IntPtr b, IntPtr OUT);
        [DllImport("EigenNN")]
        private static extern void Blend(IntPtr T, IntPtr W, float w);

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
    }
    
}