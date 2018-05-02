using System.IO;
using UnityEngine;

namespace DeepLearning {

    public class Parameters : ScriptableObject {
        public FloatMatrix[] Matrices = new FloatMatrix[0];

        public void Store(string fn, int rows, int cols) {
            ArrayExtensions.Add(ref Matrices, ReadBinary(fn, rows, cols));
        }

        public FloatMatrix Load(int index) {
            return Matrices[index];
        }

        private FloatMatrix ReadBinary(string fn, int rows, int cols) {
            FloatMatrix matrix = new FloatMatrix(rows, cols);
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

        [System.Serializable]
        public class FloatVector {
            public float[] Values;

            public FloatVector(int size) {
                Values = new float[size];
            }
        }

        [System.Serializable]
        public class FloatMatrix {
            public FloatVector[] Values;
            public int Rows, Cols;

            public FloatMatrix(int rows, int cols) {
                Rows = rows;
                Cols = cols;
                Values = new FloatVector[rows];
                for(int i=0; i<rows; i++) {
                    Values[i] = new FloatVector(cols);
                }
            }
        }
    }
	
}