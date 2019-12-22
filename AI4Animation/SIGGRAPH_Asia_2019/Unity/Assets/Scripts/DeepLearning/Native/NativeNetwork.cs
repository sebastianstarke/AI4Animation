using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace DeepLearning {

    public abstract class NativeNetwork : NeuralNetwork {

        public Parameters Parameters = null;
		public string Folder = "";
        protected abstract void LoadDerived();
        protected abstract void UnloadDerived();

        protected override bool SetupDerived() {
            if(Setup) {
                return true;
            }
            LoadDerived();
            Setup = true;
            for(int i=0; i<Matrices.Count; i++) {
                if(Matrices[i] == null) {
                    Setup = false;
                    for(int j=0; j<Matrices.Count; j++) {
                        if(Matrices[j] != null) {
                            Matrices[j].Delete();
                        }
                    }
                    Matrices.Clear();
                    Debug.Log("Building network " + name + " failed.");
                    return false;
                }
            }
            return true;
        }

        protected override bool ShutdownDerived() {
            if(Setup) {
                UnloadDerived();
                DeleteMatrices();
                ResetPredictionTime();
                ResetPivot();
            }
            return false;
        }

        public Matrix CreateMatrix(int rows, int cols, string id, string binary) {
            if(Matrices.Exists(x => x != null && x.ID == id)) {
                Debug.Log("Matrix with ID " + id + " already contained.");
                return GetMatrix(id);
            }
            float[] buffer = null;
            if(Parameters != null) {
                Parameters.Buffer b = Parameters.Load(id);
                if(b != null) {
                    buffer = b.Values;
                }
            }
            if(buffer == null) {
                buffer = ReadBinary(binary, rows*cols);
            }
            if(buffer != null) {
                Matrix M = new Matrix(rows, cols, id);
                for(int row=0; row<rows; row++) {
                    for(int col=0; col<cols; col++) {
                        M.SetValue(row, col, buffer[row*cols + col]);
                    }
                }
                Matrices.Add(M);
                return M;
            } else {
                Matrices.Add(null);
                Debug.Log("Creating matrix with ID " + id + " failed.");
                return null;
            }
        }

        public Matrix Normalise(Matrix IN, Matrix mean, Matrix std, Matrix OUT) {
            if(IN.GetRows() != mean.GetRows() || IN.GetRows() != std.GetRows() || IN.GetCols() != mean.GetCols() || IN.GetCols() != std.GetCols()) {
                Debug.Log("Incompatible dimensions for normalisation.");
                return IN;
            } else {
                Eigen.Normalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
                return OUT;
            }
        }
        
        public Matrix Renormalise(Matrix IN, Matrix mean, Matrix std, Matrix OUT) {
            if(IN.GetRows() != mean.GetRows() || IN.GetRows() != std.GetRows() || IN.GetCols() != mean.GetCols() || IN.GetCols() != std.GetCols()) {
                Debug.Log("Incompatible dimensions for renormalisation.");
                return IN;
            } else {
                Eigen.Renormalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
                return OUT;
            }
        }

        public Matrix Layer(Matrix IN, Matrix W, Matrix b, Matrix OUT) {
            if(IN.GetRows() != W.GetCols() || W.GetRows() != b.GetRows() || IN.GetCols() != b.GetCols()) {
                Debug.Log("Incompatible dimensions for layer feed-forward.");
                return IN;
            } else {
                Eigen.Layer(IN.Ptr, W.Ptr, b.Ptr, OUT.Ptr);
                return OUT;
            }
        }

        public Matrix Blend(Matrix M, Matrix W, float w) {
            if(M.GetRows() != W.GetRows() || M.GetCols() != W.GetCols()) {
                Debug.Log("Incompatible dimensions for blending.");
                return M;
            } else {
                Eigen.Blend(M.Ptr, W.Ptr, w);
                return M;
            }
        }

        public Matrix BlendAll(Matrix M, Matrix[] W, float[] w, int length) {
            System.IntPtr[] ptr = new System.IntPtr[length];
            for(int i=0; i<length; i++) {
                ptr[i] = W[i].Ptr;
            }
            Eigen.BlendAll(M.Ptr, ptr, w, length);
            return M;
        }

        private float[] ReadBinary(string fn, int size) {
            if(File.Exists(fn)) {
                float[] buffer = new float[size];
                BinaryReader reader = new BinaryReader(File.Open(fn, FileMode.Open));
                for(int i=0; i<size; i++) {
                    try {
                        buffer[i] = reader.ReadSingle();
                    } catch {
                        Debug.Log("There were errors reading file at path " + fn + ".");
                        reader.Close();
                        return null;
                    }
                }
                reader.Close();
                return buffer;
            } else {
                Debug.Log("File at path " + fn + " does not exist.");
                return null;
            }
        }

	}

}
