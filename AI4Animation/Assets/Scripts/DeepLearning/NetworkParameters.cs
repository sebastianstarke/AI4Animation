using UnityEngine;
using System.IO;

public class NetworkParameters : ScriptableObject {

	//public FloatVector[] Vectors = new FloatVector[0];
	public FloatMatrix[] Matrices = new FloatMatrix[0];

	/*
	public void StoreVector(string fn, int dim) {
		Utility.Add(ref Vectors, LoadVector(fn, dim));
	}
	*/

	public void StoreMatrix(string fn, int rows, int cols) {
		Arrays.Add(ref Matrices, LoadMatrix(fn, rows, cols));
	}

	/*
	public FloatVector GetVector(int index) {
		return Vectors[index];
	}
	*/

	public FloatMatrix GetMatrix(int index) {
		return Matrices[index];
	}

	/*
	private FloatVector LoadVector(string fn, int dim) {
		FloatVector vector = new FloatVector(dim);
		try {
			BinaryReader reader = new BinaryReader(File.Open(fn, FileMode.Open));
			for(int x=0; x<dim; x++) {
				vector.Values[x] = reader.ReadSingle();
			}
		} catch (System.Exception e) {
        	Debug.Log(e.Message);
			return null;
        }
		return vector;
	}
	*/

	private FloatMatrix LoadMatrix(string fn, int rows, int cols) {
		FloatMatrix matrix = new FloatMatrix(rows, cols);
		try {
			BinaryReader reader = new BinaryReader(File.Open(fn, FileMode.Open));
			for(int x=0; x<rows; x++) {
				for(int y=0; y<cols; y++) {
					matrix.Values[x].Values[y] = reader.ReadSingle();
				}
			}
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

		public Matrix Build() {
			Matrix matrix = new Matrix(Rows, Cols);
			for(int i=0; i<Rows; i++) {
				for(int j=0; j<Cols; j++) {
					matrix.Values[i][j] = Values[i].Values[j];
				}
			}
			return matrix;
		}
	}

}