using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class PFNN {

	public bool Inspect = false;

	//public enum MODE { CONSTANT, LINEAR, CUBIC };

	//public MODE Mode = MODE.CONSTANT;

	public string Folder = string.Empty;
	public int XDim = 504;
	public int HDim = 512;
	public int YDim = 352;

	public NetworkParameters Parameters;

	//private Vector<float> Xmean, Xstd;
	//private Vector<float> Ymean, Ystd;

	//private Matrix<float>[] W0, W1, W2;
	//private Vector<float>[] b0, b1, b2;

	//private Vector<float> X, Y;
	//private Matrix<float> W0p, W1p, W2p;
	//private Vector<float> b0p, b1p, b2p;

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

		X = new Matrix(XDim, 1);
		Y = new Matrix(YDim, 1);

		/*
		W0p = Matrix<float>.Build.Dense(HDim, XDim);
		W1p = Matrix<float>.Build.Dense(HDim, HDim);
		W2p = Matrix<float>.Build.Dense(YDim, HDim);

		b0p = Vector<float>.Build.Dense(HDim);
		b1p = Vector<float>.Build.Dense(HDim);
		b2p = Vector<float>.Build.Dense(YDim);
		*/
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
		//float pamount;
		//int pindex_0;
		int pindex_1;
		//int pindex_2;
		//int pindex_3;

		Matrix _X = (X - Xmean).PointwiseDivide(Xstd);
		
		//switch(Mode) {
		//	case MODE.CONSTANT:
				pindex_1 = (int)((phase / (2*M_PI)) * 50);
				Matrix H0 = (W0[pindex_1] * _X) + b0[pindex_1]; ELU(ref H0);
				Matrix H1 = (W1[pindex_1] * H0) + b1[pindex_1]; ELU(ref H1);
				Y = (W2[pindex_1] * H1) + b2[pindex_1];
		/*	break;
			
			case MODE.LINEAR:
				//TODO: make fmod faster
				pamount = Mathf.Repeat((phase / (2*M_PI)) * 10, 1.0f);
				pindex_1 = (int)((phase / (2*M_PI)) * 10);
				pindex_2 = ((pindex_1+1) % 10);
				Linear(ref W0p, ref W0[pindex_1], ref W0[pindex_2], pamount);
				Linear(ref W1p, ref W1[pindex_1], ref W1[pindex_2], pamount);
				Linear(ref W2p, ref W2[pindex_1], ref W2[pindex_2], pamount);
				Linear(ref b0p, ref b0[pindex_1], ref b0[pindex_2], pamount);
				Linear(ref b1p, ref b1[pindex_1], ref b1[pindex_2], pamount);
				Linear(ref b2p, ref b2[pindex_1], ref b2[pindex_2], pamount);
				H0 = (W0p * _Xp) + b0p; ELU(ref H0);
				H1 = (W1p * H0) + b1p; ELU(ref H1);
				Yp = (W2p * H1) + b2p;
			break;
			
			case MODE.CUBIC:
				//TODO: make fmod faster
				pamount = Mathf.Repeat((phase / (2*M_PI)) * 4, 1.0f);
				pindex_1 = (int)((phase / (2*M_PI)) * 4);
				pindex_0 = ((pindex_1+3) % 4);
				pindex_2 = ((pindex_1+1) % 4);
				pindex_3 = ((pindex_1+2) % 4);
				Cubic(ref W0p, ref W0[pindex_0], ref W0[pindex_1], ref W0[pindex_2], ref W0[pindex_3], pamount);
				Cubic(ref W1p, ref W1[pindex_0], ref W1[pindex_1], ref W1[pindex_2], ref W1[pindex_3], pamount);
				Cubic(ref W2p, ref W2[pindex_0], ref W2[pindex_1], ref W2[pindex_2], ref W2[pindex_3], pamount);
				Cubic(ref b0p, ref b0[pindex_0], ref b0[pindex_1], ref b0[pindex_2], ref b0[pindex_3], pamount);
				Cubic(ref b1p, ref b1[pindex_0], ref b1[pindex_1], ref b1[pindex_2], ref b1[pindex_3], pamount);
				Cubic(ref b2p, ref b2[pindex_0], ref b2[pindex_1], ref b2[pindex_2], ref b2[pindex_3], pamount);
				H0 = (W0p * _Xp) + b0p; ELU(ref H0);
				H1 = (W1p * H0) + b1p; ELU(ref H1);
				Yp = (W2p * H1) + b2p;
			break;
			
			default:
			break;
		}
		*/
		
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

	private void Linear(ref Matrix o, ref Matrix y0, ref Matrix y1, float mu) {
		o = (1.0f-mu) * y0 + (mu) * y1;
	}

	private void Cubic(ref Matrix o, ref Matrix y0, ref Matrix y1, ref Matrix y2, ref Matrix y3, float mu) {
		o = (
		(-0.5f*y0 + 1.5f*y1 - 1.5f*y2 + 0.5f*y3)*mu*mu*mu + 
		(y0 - 2.5f*y1 + 2.0f*y2 - 0.5f*y3)*mu*mu + 
		(-0.5f*y0 + 0.5f*y2)*mu + 
		(y1));
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