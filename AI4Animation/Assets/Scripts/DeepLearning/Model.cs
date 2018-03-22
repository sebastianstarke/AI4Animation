using System.Collections.Generic;
using UnityEngine;

namespace DeepLearning {
    
    public abstract class Model {

        public Parameters Parameters;

        private List<Tensor> Tensors = new List<Tensor>();
        private List<string> Identifiers = new List<string>();

        public Tensor CreateTensor(int rows, int cols, string id) {
            if(Identifiers.Contains(id)) {
                Debug.Log("Tensor with ID " + id + " already contained.");
                return null;
            }
            Tensor T = new Tensor(rows, cols);
            Tensors.Add(T);
            Identifiers.Add(id);
            return T;
        }

        public Tensor CreateTensor(Parameters.FloatMatrix matrix, string id) {
            if(Identifiers.Contains(id)) {
                Debug.Log("Tensor with ID " + id + " already contained.");
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
                Debug.Log("Tensor not found.");
                return;
            }
            Tensors.RemoveAt(index);
            Identifiers.RemoveAt(index);
            T.Delete();
        }

        public Tensor GetTensor(string id) {
            int index = Identifiers.IndexOf(id);
            if(index == -1) {
                return null;
            }
            return Tensors[index];
        }

        public string GetID(Tensor T) {
            int index = Tensors.IndexOf(T);
            if(index == -1) {
                return null;
            }
            return Identifiers[index];
        }

        public Tensor Normalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            Eigen.Normalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
            return OUT;
        }
        
        public Tensor Renormalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            Eigen.Renormalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Layer(Tensor IN, Tensor W, Tensor b, Tensor OUT) {
            Eigen.Layer(IN.Ptr, W.Ptr, b.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Blend(Tensor T, Tensor W, float w) {
            Eigen.Blend(T.Ptr, W.Ptr, w);
            return T;
        }

        public Tensor ELU(Tensor T) {
            Eigen.ELU(T.Ptr);
            return T;
        }

        public Tensor Sigmoid(Tensor T) {
            Eigen.Sigmoid(T.Ptr);
            return T;
        }

        public Tensor TanH(Tensor T) {
            Eigen.TanH(T.Ptr);
            return T;
        }

        public Tensor SoftMax(Tensor T) {
            Eigen.SoftMax(T.Ptr);
            return T;
        }

    }
    
}