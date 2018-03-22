using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace DeepLearning {
    
    public class Model {
        //Functions
        [DllImport("DeepLearning")]
        private static extern void Normalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
        [DllImport("DeepLearning")]
        private static extern void Renormalise(IntPtr IN, IntPtr mean, IntPtr std, IntPtr OUT);
        [DllImport("DeepLearning")]
        private static extern void Layer(IntPtr IN, IntPtr W, IntPtr b, IntPtr OUT);
        [DllImport("DeepLearning")]
        private static extern void Blend(IntPtr T, IntPtr W, float w);

        //Activations
        [DllImport("DeepLearning")]
        private static extern void ELU(IntPtr T);
        [DllImport("DeepLearning")]
        private static extern void Sigmoid(IntPtr T);
        [DllImport("DeepLearning")]
        private static extern void TanH(IntPtr T);
        [DllImport("DeepLearning")]
        private static extern void SoftMax(IntPtr T);

        private List<Tensor> Tensors = new List<Tensor>();
        private List<string> Identifiers = new List<string>();

        public Tensor CreateTensor(int rows, int cols, string id) {
            if(Identifiers.Contains(id)) {
                //Debug.Log("Tensor with ID " + id + " already contained.");
                return null;
            }
            Tensor T = new Tensor(rows, cols);
            Tensors.Add(T);
            Identifiers.Add(id);
            return T;
        }

        public Tensor CreateTensor(Parameters.FloatMatrix matrix, string id) {
            if(Identifiers.Contains(id)) {
                //Debug.Log("Tensor with ID " + id + " already contained.");
                return null;
            }
            Tensor T = new Tensor(matrix.Rows, matrix.Cols);
            for(int x=0; x<matrix.Rows; x++) {
                for(int y=0; y<matrix.Cols; y++) {
                    T.SetValue(x, y, matrix.Values[x].Values[y]);
                }
            }
            Tensors.Add(T);
            Identifiers.Add(id);
            return T;
        }

        public void DeleteTensor(Tensor T) {
            int index = Tensors.IndexOf(T);
            if(index == -1) {
               // Debug.Log("Tensor not found.");
                return;
            }
            Tensors.RemoveAt(index);
            Identifiers.RemoveAt(index);
            T.Delete();
        }

        public Tensor GetTensor(string id) {
            int index = Identifiers.IndexOf(id);
            if(index == -1) {
                //Debug.Log("ID not found.");
                return null;
            }
            return Tensors[index];
        }

        public string GetID(Tensor T) {
            int index = Tensors.IndexOf(T);
            if(index == -1) {
                //Debug.Log("Tensor not found.");
                return null;
            }
            return Identifiers[index];
        }

        public Tensor Normalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            Normalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
            return OUT;
        }
        
        public Tensor Renormalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            Renormalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Layer(Tensor IN, Tensor W, Tensor b, Tensor OUT) {
            Layer(IN.Ptr, W.Ptr, b.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Blend(Tensor T, Tensor W, float w) {
            Blend(T.Ptr, W.Ptr, w);
            return T;
        }

        public Tensor ELU(Tensor T) {
            ELU(T.Ptr);
            return T;
        }

        public Tensor Sigmoid(Tensor T) {
            Sigmoid(T.Ptr);
            return T;
        }

        public Tensor TanH(Tensor T) {
            TanH(T.Ptr);
            return T;
        }

        public Tensor SoftMax(Tensor T) {
            SoftMax(T.Ptr);
            return T;
        }
    }
    
}