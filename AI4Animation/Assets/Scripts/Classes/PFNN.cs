using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class PFNN {

	public bool Inspect = false;

	public enum MODE { Constant, Linear, Cubic };

	public MODE Mode = MODE.Constant;

	public string Folder = string.Empty;
	public int XDim = 504;
	public int HDim = 512;
	public int YDim = 352;

	public NetworkParameters Parameters;

	private Matrix Xmean, Xstd, Ymean, Ystd;

	private Matrix[] W0, W1, W2, b0, b1, b2;

	private Matrix X, Y;

	private const float M_PI = 3.14159265358979323846f;

	public PFNN() {
		
	}

	public void Initialise() {
		if(Parameters == null) {
			Debug.Log("Building PFNN failed because no parameters were loaded.");
			return;
		}

		Xmean = Parameters.GetMatrix(0).Build();
		Xstd = Parameters.GetMatrix(1).Build();
		Ymean = Parameters.GetMatrix(2).Build();
		Ystd = Parameters.GetMatrix(3).Build();

		switch(Mode) {
			case MODE.Constant:
			W0 = new Matrix[50];
			W1 = new Matrix[50];
			W2 = new Matrix[50];
			b0 = new Matrix[50];
			b1 = new Matrix[50];
			b2 = new Matrix[50];
			for(int i=0; i<50; i++) {
				W0[i] = Parameters.GetMatrix(4 + i*6 + 0).Build();
				W1[i] = Parameters.GetMatrix(4 + i*6 + 1).Build();
				W2[i] = Parameters.GetMatrix(4 + i*6 + 2).Build();
				b0[i] = Parameters.GetMatrix(4 + i*6 + 3).Build();
				b1[i] = Parameters.GetMatrix(4 + i*6 + 4).Build();
				b2[i] = Parameters.GetMatrix(4 + i*6 + 5).Build();
			}
			break;

			case MODE.Linear:
			W0 = new Matrix[10];
			W1 = new Matrix[10];
			W2 = new Matrix[10];
			b0 = new Matrix[10];
			b1 = new Matrix[10];
			b2 = new Matrix[10];
			for(int i=0; i<50; i+=5) {
				W0[i] = Parameters.GetMatrix(4 + i*6 + 0).Build();
				W1[i] = Parameters.GetMatrix(4 + i*6 + 1).Build();
				W2[i] = Parameters.GetMatrix(4 + i*6 + 2).Build();
				b0[i] = Parameters.GetMatrix(4 + i*6 + 3).Build();
				b1[i] = Parameters.GetMatrix(4 + i*6 + 4).Build();
				b2[i] = Parameters.GetMatrix(4 + i*6 + 5).Build();
			}
			break;

			case MODE.Cubic:
			W0 = new Matrix[4];
			W1 = new Matrix[4];
			W2 = new Matrix[4];
			b0 = new Matrix[4];
			b1 = new Matrix[4];
			b2 = new Matrix[4];
			for(float i=0; i<50f; i+=12.5f) {
				int index = Mathf.RoundToInt(i);
				W0[index] = Parameters.GetMatrix(4 + index*6 + 0).Build();
				W1[index] = Parameters.GetMatrix(4 + index*6 + 1).Build();
				W2[index] = Parameters.GetMatrix(4 + index*6 + 2).Build();
				b0[index] = Parameters.GetMatrix(4 + index*6 + 3).Build();
				b1[index] = Parameters.GetMatrix(4 + index*6 + 4).Build();
				b2[index] = Parameters.GetMatrix(4 + index*6 + 5).Build();
			}
			break;

			default:
			break;
		}

		X = new Matrix(XDim, 1);
		Y = new Matrix(YDim, 1);
	}

	public void LoadParameters() {
		Parameters = ScriptableObject.CreateInstance<NetworkParameters>();
		Parameters.StoreMatrix(Folder+"/Xmean.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Xstd.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Ymean.bin", YDim, 1);
		Parameters.StoreMatrix(Folder+"/Ystd.bin", YDim, 1);
		for(int i=0; i<50; i++) {
			Parameters.StoreMatrix(Folder+"/W0_"+i.ToString("D3")+".bin", HDim, XDim);
			Parameters.StoreMatrix(Folder+"/W1_"+i.ToString("D3")+".bin", HDim, HDim);
			Parameters.StoreMatrix(Folder+"/W2_"+i.ToString("D3")+".bin", YDim, HDim);
			Parameters.StoreMatrix(Folder+"/b0_"+i.ToString("D3")+".bin", HDim, 1);
			Parameters.StoreMatrix(Folder+"/b1_"+i.ToString("D3")+".bin", HDim, 1);
			Parameters.StoreMatrix(Folder+"/b2_"+i.ToString("D3")+".bin", YDim, 1);
		}
	}

	public void SetMode(MODE mode) {
		if(Mode != mode) {
			Mode = mode;
			Initialise();
		}
	}

	public void SetInput(int i, float value) {
		X.Values[i][0] = value;
	}

	public float GetOutput(int i) {
		return Y.Values[i][0];
	}

	public void Output() {
		Debug.Log("====================INPUT====================");
		for(int i=0; i<XDim; i++) {
			Debug.Log(i + ": " + X.Values[i][0]);
		}
		Debug.Log("====================OUTPUT====================");
		for(int i=0; i<YDim; i++) {
			Debug.Log(i + ": " + Y.Values[i][0]);
		}
	}

	public void Predict(float phase) {
		float pamount;
		int pindex_0;
		int pindex_1;
		int pindex_2;
		int pindex_3;

		Matrix _X = (X - Xmean).PointwiseDivide(Xstd);
		
		switch(Mode) {
			case MODE.Constant:
			pindex_1 = (int)((phase / (2*M_PI)) * 50);
			Matrix H0 = (W0[pindex_1] * _X) + b0[pindex_1]; ELU(ref H0);
			Matrix H1 = (W1[pindex_1] * H0) + b1[pindex_1]; ELU(ref H1);
			Y = (W2[pindex_1] * H1) + b2[pindex_1];
			break;
		
			case MODE.Linear:
			//NOT YET WORKING
			//TODO: make fmod faster
			pamount = Mathf.Repeat((phase / (2*M_PI)) * 10, 1.0f);
			pindex_1 = (int)((phase / (2*M_PI)) * 10);
			pindex_2 = ((pindex_1+1) % 10);
			Matrix W0l = Linear(ref W0[pindex_1], ref W0[pindex_2], pamount);
			Matrix W1l = Linear(ref W1[pindex_1], ref W1[pindex_2], pamount);
			Matrix W2l = Linear(ref W2[pindex_1], ref W2[pindex_2], pamount);
			Matrix b0l = Linear(ref b0[pindex_1], ref b0[pindex_2], pamount);
			Matrix b1l = Linear(ref b1[pindex_1], ref b1[pindex_2], pamount);
			Matrix b2l = Linear(ref b2[pindex_1], ref b2[pindex_2], pamount);
			H0 = (W0l * _X) + b0l; ELU(ref H0);
			H1 = (W1l * H0) + b1l; ELU(ref H1);
			Y = (W2l * H1) + b2l;
			break;
			
			case MODE.Cubic:
			//NOT YET WORKING
			//TODO: make fmod faster
			pamount = Mathf.Repeat((phase / (2*M_PI)) * 4, 1.0f);
			pindex_1 = (int)((phase / (2*M_PI)) * 4);
			pindex_0 = ((pindex_1+3) % 4);
			pindex_2 = ((pindex_1+1) % 4);
			pindex_3 = ((pindex_1+2) % 4);
			Matrix W0c = Cubic(ref W0[pindex_0], ref W0[pindex_1], ref W0[pindex_2], ref W0[pindex_3], pamount);
			Matrix W1c = Cubic(ref W1[pindex_0], ref W1[pindex_1], ref W1[pindex_2], ref W1[pindex_3], pamount);
			Matrix W2c = Cubic(ref W2[pindex_0], ref W2[pindex_1], ref W2[pindex_2], ref W2[pindex_3], pamount);
			Matrix b0c = Cubic(ref b0[pindex_0], ref b0[pindex_1], ref b0[pindex_2], ref b0[pindex_3], pamount);
			Matrix b1c = Cubic(ref b1[pindex_0], ref b1[pindex_1], ref b1[pindex_2], ref b1[pindex_3], pamount);
			Matrix b2c = Cubic(ref b2[pindex_0], ref b2[pindex_1], ref b2[pindex_2], ref b2[pindex_3], pamount);
			H0 = (W0c * _X) + b0c; ELU(ref H0);
			H1 = (W1c * H0) + b1c; ELU(ref H1);
			Y = (W2c * H1) + b2c;
			break;
			
			default:
			break;
		}
		
		Y = (Y.PointwiseMultiply(Ystd)) + Ymean;
	}

	private void ELU(ref Matrix m) {
		for(int x=0; x<m.Values.Length; x++) {
			m.Values[x][0] = System.Math.Max(m.Values[x][0], 0f) + (float)System.Math.Exp(System.Math.Min(m.Values[x][0], 0f)) - 1f;
		}
	}

	private void SoftMax(ref Matrix m) {
		float lower = 0f;
		for(int x=0; x<m.Values.Length; x++) {
			lower += (float)System.Math.Exp(m.Values[x][0]);
		}
		for(int x=0; x<m.Values.Length; x++) {
			m.Values[x][0] = (float)System.Math.Exp(m.Values[x][0]) / lower;
		}
	}

	private Matrix Linear(ref Matrix y0, ref Matrix y1, float mu) {
		return (1.0f-mu) * y0 + (mu) * y1;
	}

	private Matrix Cubic(ref Matrix y0, ref Matrix y1, ref Matrix y2, ref Matrix y3, float mu) {
		return
		(-0.5f*y0 + 1.5f*y1 - 1.5f*y2 + 0.5f*y3)*mu*mu*mu + 
		(y0 - 2.5f*y1 + 2.0f*y2 - 0.5f*y3)*mu*mu + 
		(-0.5f*y0 + 0.5f*y2)*mu + 
		(y1);
	}

	#if UNITY_EDITOR
	public void Inspector() {
		Utility.SetGUIColor(Color.grey);
		using(new GUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Utility.GUIButton("PFNN", Utility.DarkGrey, Utility.White)) {
				Inspect = !Inspect;
			}

			if(Inspect) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Folder = EditorGUILayout.TextField("Folder", Folder);
					Mode = (MODE)EditorGUILayout.EnumPopup("Mode", Mode);
					XDim = EditorGUILayout.IntField("XDim", XDim);
					HDim = EditorGUILayout.IntField("HDim", HDim);
					YDim = EditorGUILayout.IntField("YDim", YDim);
					EditorGUILayout.BeginHorizontal();
					if(GUILayout.Button("Load Parameters")) {
						LoadParameters();
					}
					Parameters = (NetworkParameters)EditorGUILayout.ObjectField(Parameters, typeof(NetworkParameters), true);
					EditorGUILayout.EndHorizontal();
				}
			}
		}
	}
	#endif

}