using System.IO;
using UnityEngine;

namespace DeepLearning {

    public class Parameters : ScriptableObject {
        public Matrix[] Matrices = new Matrix[0];

        public void Store(string fn, int rows, int cols, string id) {
            for(int i=0; i<Matrices.Length; i++) {
                if(Matrices[i] != null) {
                    if(Matrices[i].ID == id) {
                        Debug.Log("Matrix with ID " + id + " already contained.");
                        return;
                    }
                }
            }
            ArrayExtensions.Add(ref Matrices, ReadBinary(fn, rows, cols, id));
        }

        public Matrix Load(string id) {
            Matrix matrix = System.Array.Find(Matrices, x => x.ID == id);
            if(matrix == null) {
                Debug.Log("Matrix with ID " + id + " not found.");
            }
            return matrix;
        }

        public void Clear() {
            ArrayExtensions.Resize(ref Matrices, 0);
        }

        private Matrix ReadBinary(string fn, int rows, int cols, string id) {
            if(File.Exists(fn)) {
                Matrix matrix = new Matrix(rows, cols, id);
                BinaryReader reader = new BinaryReader(File.Open(fn, FileMode.Open));
                int errors = 0;
                for(int x=0; x<rows; x++) {
                    for(int y=0; y<cols; y++) {
                        try {
                            matrix.Values[x].Values[y] = reader.ReadSingle();
                        } catch {
                            errors += 1;
                        }
                    }
                }
                reader.Close();
                if(errors > 0) {
                    Debug.Log("There were " + errors + " errors reading file at path " + fn + ".");
                    return null;
                } else {
                    return matrix;
                }
            } else {
                Debug.Log("File at path " + fn + " does not exist.");
                return null;
            }
        }

        public bool Validate() {
            for(int i=0; i<Matrices.Length; i++) {
                if(Matrices[i] == null) {
                    return false;
                }
            }
            return true;
        }

        [System.Serializable]
        public class Vector {
            public float[] Values;

            public Vector(int size) {
                Values = new float[size];
            }
        }

        [System.Serializable]
        public class Matrix {
            public Vector[] Values;
            public int Rows, Cols;
            public string ID;

            public Matrix(int rows, int cols, string id) {
                Rows = rows;
                Cols = cols;
                ID = id;
                Values = new Vector[rows];
                for(int i=0; i<rows; i++) {
                    Values[i] = new Vector(cols);
                }
            }
        }
    }
	
}