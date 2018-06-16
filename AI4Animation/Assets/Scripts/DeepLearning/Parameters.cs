using System.IO;
using UnityEngine;

namespace DeepLearning {

    public class Parameters : ScriptableObject {
        public Matrix[] Matrices = new Matrix[0];

        public void Store(string fn, int rows, int cols, string id) {
            ArrayExtensions.Add(ref Matrices, ReadBinary(fn, rows, cols, id));
        }

        public Matrix Load(string id) {
            return System.Array.Find(Matrices, x => x.ID == id);
        }

        private Matrix ReadBinary(string fn, int rows, int cols, string id) {
            Matrix matrix = new Matrix(rows, cols, id);
            try {
                BinaryReader reader = new BinaryReader(File.Open(fn, FileMode.Open));
                for(int x=0; x<rows; x++) {
                    for(int y=0; y<cols; y++) {
                        matrix.Values[x].Values[y] = reader.ReadSingle();
                    }
                }
                reader.Close();
            } catch (System.Exception e) {
                Debug.Log(e.Message);
                return null;
            }
            return matrix;
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