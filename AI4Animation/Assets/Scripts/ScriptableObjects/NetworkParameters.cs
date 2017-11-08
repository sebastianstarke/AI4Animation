using UnityEngine;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

public class NetworkParameters : ScriptableObject {

	public FloatMatrix Xmean, Xstd, Ymean, Ystd;
	public FloatMatrix[] W0, W1, W2, b0, b1, b2;

	public void Load(int xDim, int yDim, int hDim) {
		Xmean = LoadWeights("Assets/Animation/PFNN/Xmean.bin", xDim, 1);
		Xstd = LoadWeights("Assets/Animation/PFNN/Xstd.bin", xDim, 1);
		Ymean = LoadWeights("Assets/Animation/PFNN/Ymean.bin", yDim, 1);
		Ystd = LoadWeights("Assets/Animation/PFNN/Ystd.bin", yDim, 1);
		
		W0 = new FloatMatrix[50];
		W1 = new FloatMatrix[50];
		W2 = new FloatMatrix[50];
		b0 = new FloatMatrix[50];
		b1 = new FloatMatrix[50];
		b2 = new FloatMatrix[50];
		for(int i=0; i<50; i++) {
			W0[i] = LoadWeights("Assets/Animation/PFNN/W0_"+i.ToString("D3")+".bin", hDim, xDim);
			W1[i] = LoadWeights("Assets/Animation/PFNN/W1_"+i.ToString("D3")+".bin", hDim, hDim);
			W2[i] = LoadWeights("Assets/Animation/PFNN/W2_"+i.ToString("D3")+".bin", yDim, hDim);
			b0[i] = LoadWeights("Assets/Animation/PFNN/b0_"+i.ToString("D3")+".bin", hDim, 1);
			b1[i] = LoadWeights("Assets/Animation/PFNN/b1_"+i.ToString("D3")+".bin", hDim, 1);
			b2[i] = LoadWeights("Assets/Animation/PFNN/b2_"+i.ToString("D3")+".bin", yDim, 1);
		}

		/*
		switch(Mode) {
			case MODE.CONSTANT:
			W0 = new Matrix<float>[50];
			W1 = new Matrix<float>[50];
			W2 = new Matrix<float>[50];
			b0 = new Matrix<float>[50];
			b1 = new Matrix<float>[50];
			b2 = new Matrix<float>[50];
			for(int i=0; i<50; i++) {
				LoadWeights(ref W0[i], HDim, XDim, "Assets/Animation/PFNN/W0_"+i.ToString("D3")+".bin");
				LoadWeights(ref W1[i], HDim, HDim, "Assets/Animation/PFNN/W1_"+i.ToString("D3")+".bin");
				LoadWeights(ref W2[i], YDim, HDim, "Assets/Animation/PFNN/W2_"+i.ToString("D3")+".bin");
				LoadWeights(ref b0[i], HDim, 1, "Assets/Animation/PFNN/b0_"+i.ToString("D3")+".bin");
				LoadWeights(ref b1[i], HDim, 1, "Assets/Animation/PFNN/b1_"+i.ToString("D3")+".bin");
				LoadWeights(ref b2[i], YDim, 1, "Assets/Animation/PFNN/b2_"+i.ToString("D3")+".bin");
			}	
			break;
			
			case MODE.LINEAR:
			W0 = new Matrix<float>[10];
			W1 = new Matrix<float>[10];
			W2 = new Matrix<float>[10];
			b0 = new Matrix<float>[10];
			b1 = new Matrix<float>[10];
			b2 = new Matrix<float>[10];
			for(int i=0; i<10; i++) {
				LoadWeights(ref W0[i], HDim, XDim, "Assets/Animation/PFNN/W0_"+(i*5).ToString("D3")+".bin");
				LoadWeights(ref W1[i], HDim, HDim, "Assets/Animation/PFNN/W1_"+(i*5).ToString("D3")+".bin");
				LoadWeights(ref W2[i], YDim, HDim, "Assets/Animation/PFNN/W2_"+(i*5).ToString("D3")+".bin");
				LoadWeights(ref b0[i], HDim, 1, "Assets/Animation/PFNN/b0_"+(i*5).ToString("D3")+".bin");
				LoadWeights(ref b1[i], HDim, 1, "Assets/Animation/PFNN/b1_"+(i*5).ToString("D3")+".bin");
				LoadWeights(ref b2[i], YDim, 1, "Assets/Animation/PFNN/b2_"+(i*5).ToString("D3")+".bin");
			}	
			break;

			case MODE.CUBIC:
			W0 = new Matrix<float>[4];
			W1 = new Matrix<float>[4];
			W2 = new Matrix<float>[4];
			b0 = new Matrix<float>[4];
			b1 = new Matrix<float>[4];
			b2 = new Matrix<float>[4];
			for(int i=0; i<4; i++) {
				LoadWeights(ref W0[i], HDim, XDim, "Assets/Animation/PFNN/W0_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(ref W1[i], HDim, HDim, "Assets/Animation/PFNN/W1_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(ref W2[i], YDim, HDim, "Assets/Animation/PFNN/W2_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(ref b0[i], HDim, 1, "Assets/Animation/PFNN/b0_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(ref b1[i], HDim, 1, "Assets/Animation/PFNN/b1_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(ref b2[i], YDim, 1, "Assets/Animation/PFNN/b2_"+(i*12.5).ToString("D3")+".bin");
			}	
			break;
		}
		*/
	}

	private FloatMatrix LoadWeights(string fn, int rows, int cols) {
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
        }
		return matrix;
	}

	[System.Serializable]
	public class FloatArray {
		public float[] Values;

		public FloatArray(int size) {
			Values = new float[size];
		}
	}

	[System.Serializable]
	public class FloatMatrix {
		public FloatArray[] Values;
		public int Rows, Cols;

		public FloatMatrix(int rows, int cols) {
			Rows = rows;
			Cols = cols;
			Values = new FloatArray[rows];
			for(int i=0; i<rows; i++) {
				Values[i] = new FloatArray(cols);
			}
		}

		public Matrix<float> Build() {
			Matrix<float> matrix = Matrix<float>.Build.Dense(Rows, Cols);
			for(int i=0; i<Rows; i++) {
				matrix.SetRow(i, Values[i].Values);
			}
			return matrix;
		}
	}

}