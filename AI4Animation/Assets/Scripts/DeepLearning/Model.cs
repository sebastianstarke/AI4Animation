using System;
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