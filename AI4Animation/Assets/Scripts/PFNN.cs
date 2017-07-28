using UnityEngine;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

public class PFNN {

	public enum MODE { CONSTANT, LINEAR, CUBIC };

	private MODE Mode;

	private int XDim = 342;
	private int YDim = 311;
	private int HDim = 512;

	private Matrix<float> Xmean, Xstd;
	private Matrix<float> Ymean, Ystd;
	private List<Matrix<float>> W0, W1, W2;
	private List<Matrix<float>> b0, b1, b2;

	private Matrix<float> Xp, Yp;
	private Matrix<float> H0, H1;

	private Matrix<float> W0p, W1p, W2p;
	private Matrix<float> b0p, b1p, b2p;

	public PFNN(MODE mode) {
		Xp = Matrix<float>.Build.Dense(1, XDim);
		Yp = Matrix<float>.Build.Dense(1, YDim);

		H0 = Matrix<float>.Build.Dense(1, HDim);
		H1 = Matrix<float>.Build.Dense(1, HDim);

		W0p = Matrix<float>.Build.Dense(HDim, XDim);
		W1p = Matrix<float>.Build.Dense(HDim, HDim);
		W2p = Matrix<float>.Build.Dense(YDim, HDim);

		b0p = Matrix<float>.Build.Dense(1, HDim);
		b1p = Matrix<float>.Build.Dense(1, HDim);
		b2p = Matrix<float>.Build.Dense(1, YDim);
	}

	public void Load() {
		LoadWeights(Xmean, XDim, "./network/pfnn/Xmean.bin");
		LoadWeights(Xstd,  XDim, "./network/pfnn/Xstd.bin");
		LoadWeights(Ymean, YDim, "./network/pfnn/Ymean.bin");
		LoadWeights(Ystd,  YDim, "./network/pfnn/Ystd.bin");
    
		switch(Mode) {
			case MODE.CONSTANT:
			W0 = new List<Matrix<float>>(50);
			W1 = new List<Matrix<float>>(50);
			W2 = new List<Matrix<float>>(50);
			b0 = new List<Matrix<float>>(50);
			b1 = new List<Matrix<float>>(50);
			b2 = new List<Matrix<float>>(50);
			for(int i=0; i<50; i++) {            
				LoadWeights(W0[i], HDim, XDim, "./network/pfnn/W0_%03i.bin", i);
				LoadWeights(W1[i], HDim, HDim, "./network/pfnn/W1_%03i.bin", i);
				LoadWeights(W2[i], YDim, HDim, "./network/pfnn/W2_%03i.bin", i);
				LoadWeights(b0[i], HDim, "./network/pfnn/b0_%03i.bin", i);
				LoadWeights(b1[i], HDim, "./network/pfnn/b1_%03i.bin", i);
				LoadWeights(b2[i], YDim, "./network/pfnn/b2_%03i.bin", i);            
			}	
			break;
			
			case MODE.LINEAR:
			//TODO
			break;

			case MODE.CUBIC:
			//TODO
			break;
		}
	}

	public void LoadWeights(Matrix<float> A, int rows, int cols, string fn, params int[] values) {

	}

	public void LoadWeights(Matrix<float> A, int items, string fn, params int[] values) {
		
	}

	public void Test(int rows, int cols) {
		Debug.Log("Nothing to do!");
	}

}
