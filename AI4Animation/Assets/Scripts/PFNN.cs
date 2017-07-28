using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.IO;  

public class PFNN {

	public enum MODE { CONSTANT, LINEAR, CUBIC };

	private MODE Mode;

	private int XDim = 342;
	private int YDim = 311;
	private int HDim = 512;

	private Matrix<float> Xmean, Xstd;
	private Matrix<float> Ymean, Ystd;
	private Matrix<float>[] W0, W1, W2;
	private Matrix<float>[] b0, b1, b2;

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
		LoadWeights(Xmean, 1, XDim, "../PFNN/demo/network/pfnn/Xmean.bin");
		LoadWeights(Xstd,  1, XDim, "../PFNN/demo/network/pfnn/Xstd.bin");
		LoadWeights(Ymean, 1, YDim, "../PFNN/demo/network/pfnn/Ymean.bin");
		LoadWeights(Ystd,  1, YDim, "../PFNN/demo/network/pfnn/Ystd.bin");
    
		switch(Mode) {
			case MODE.CONSTANT:
			W0 = new Matrix<float>[50];
			W1 = new Matrix<float>[50];
			W2 = new Matrix<float>[50];
			b0 = new Matrix<float>[50];
			b1 = new Matrix<float>[50];
			b2 = new Matrix<float>[50];
			for(int i=0; i<50; i++) {
				LoadWeights(W0[i], HDim, XDim, "../PFNN/demo/network/pfnn/W0_"+i.ToString("D3")+".bin");
				LoadWeights(W1[i], HDim, HDim, "../PFNN/demo/network/pfnn/W1_"+i.ToString("D3")+".bin");
				LoadWeights(W2[i], YDim, HDim, "../PFNN/demo/network/pfnn/W2_"+i.ToString("D3")+".bin");
				LoadWeights(b0[i], 1, HDim, "../PFNN/demo/network/pfnn/b0_"+i.ToString("D3")+".bin");
				LoadWeights(b1[i], 1, HDim, "../PFNN/demo/network/pfnn/b1_"+i.ToString("D3")+".bin");
				LoadWeights(b2[i], 1, YDim, "../PFNN/demo/network/pfnn/b2_"+i.ToString("D3")+".bin");
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
				LoadWeights(W0[i], HDim, XDim, "../PFNN/demo/network/pfnn/W0_"+(i*5).ToString("D3")+".bin");
				LoadWeights(W1[i], HDim, HDim, "../PFNN/demo/network/pfnn/W1_"+(i*5).ToString("D3")+".bin");
				LoadWeights(W2[i], YDim, HDim, "../PFNN/demo/network/pfnn/W2_"+(i*5).ToString("D3")+".bin");
				LoadWeights(b0[i], 1, HDim, "../PFNN/demo/network/pfnn/b0_"+(i*5).ToString("D3")+".bin");
				LoadWeights(b1[i], 1, HDim, "../PFNN/demo/network/pfnn/b1_"+(i*5).ToString("D3")+".bin");
				LoadWeights(b2[i], 1, YDim, "../PFNN/demo/network/pfnn/b2_"+(i*5).ToString("D3")+".bin");
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
				LoadWeights(W0[i], HDim, XDim, "../PFNN/demo/network/pfnn/W0_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(W1[i], HDim, HDim, "../PFNN/demo/network/pfnn/W1_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(W2[i], YDim, HDim, "../PFNN/demo/network/pfnn/W2_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(b0[i], 1, HDim, "../PFNN/demo/network/pfnn/b0_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(b1[i], 1, HDim, "../PFNN/demo/network/pfnn/b1_"+(i*12.5).ToString("D3")+".bin");
				LoadWeights(b2[i], 1, YDim, "../PFNN/demo/network/pfnn/b2_"+(i*12.5).ToString("D3")+".bin");
			}	
			break;
		}
	}

	private void LoadWeights(Matrix<float> m, int rows, int cols, string fn) {
		try {
			BinaryReader reader = new BinaryReader(File.Open(fn, FileMode.Open));
			m = Matrix<float>.Build.Dense(rows, cols);
			int elements = 0;
			for(int x=0; x<rows; x++) {
				for(int y=0; y<cols; y++) {
					elements += 1;
					m[x,y] = reader.ReadSingle();
				}
			}
		} catch (System.Exception e) {
        	Debug.Log(e.Message);
        }
	}
	
	public void ELU(Matrix<float> m) {
		for(int x=0; x<m.RowCount; x++) {
			for(int y=0; y<m.ColumnCount; y++) {
				m[x,y] = System.Math.Max(m[x,y], 0f) + (float)System.Math.Exp(System.Math.Min(m[x,y], 0f)) - 1f;
			}
		}
	}

	public void Linear(Matrix<float> o, Matrix<float> y0, Matrix<float> y1, float mu) {
		o = (1.0f-mu) * y0 + (mu) * y1;
	}

	public void Cubic(Matrix<float> o, Matrix<float> y0, Matrix<float> y1, Matrix<float> y2, Matrix<float> y3, float mu) {
		o = (
		(-0.5f*y0+1.5f*y1-1.5f*y2+0.5f*y3)*mu*mu*mu + 
		(y0-2.5f*y1+2.0f*y2-0.5f*y3)*mu*mu + 
		(-0.5f*y0+0.5f*y2)*mu + 
		(y1));
	}
}
