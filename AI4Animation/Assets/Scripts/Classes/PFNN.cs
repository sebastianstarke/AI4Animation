using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
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

	public Matrix<float> Xmean, Xstd;
	public Matrix<float> Ymean, Ystd;
	public Matrix<float>[] W0, W1, W2;
	public Matrix<float>[] b0, b1, b2;

	public Matrix<float> Xp, Yp;
	public Matrix<float> H0, H1;

	public Matrix<float> W0p, W1p, W2p;
	public Matrix<float> b0p, b1p, b2p;

	private const float M_PI = 3.14159265358979323846f;

	public PFNN() {
		
	}

	public void Initialise() {
		if(Parameters == null) {
			Debug.Log("Building PFNN failed because no parameters were loaded.");
			return;
		}

		Xmean = Parameters.Xmean.Build();
		Xstd = Parameters.Xstd.Build();
		Ymean = Parameters.Ymean.Build();
		Ystd = Parameters.Ystd.Build();

		W0 = new Matrix<float>[Parameters.W0.Length];
		W1 = new Matrix<float>[Parameters.W1.Length];
		W2 = new Matrix<float>[Parameters.W2.Length];
		b0 = new Matrix<float>[Parameters.b0.Length];
		b1 = new Matrix<float>[Parameters.b1.Length];
		b2 = new Matrix<float>[Parameters.b2.Length];
		for(int i=0; i<W0.Length; i++) {
			W0[i] = Parameters.W0[i].Build();
		}
		for(int i=0; i<W0.Length; i++) {
			W1[i] = Parameters.W1[i].Build();
		}
		for(int i=0; i<W0.Length; i++) {
			W2[i] = Parameters.W2[i].Build();
		}
		for(int i=0; i<W0.Length; i++) {
			b0[i] = Parameters.b0[i].Build();
		}
		for(int i=0; i<W0.Length; i++) {
			b1[i] = Parameters.b1[i].Build();
		}
		for(int i=0; i<W0.Length; i++) {
			b2[i] = Parameters.b2[i].Build();
		}

		Xp = Matrix<float>.Build.Dense(XDim, 1);
		Yp = Matrix<float>.Build.Dense(YDim, 1);

		H0 = Matrix<float>.Build.Dense(HDim, 1);
		H1 = Matrix<float>.Build.Dense(HDim, 1);

		W0p = Matrix<float>.Build.Dense(HDim, XDim);
		W1p = Matrix<float>.Build.Dense(HDim, HDim);
		W2p = Matrix<float>.Build.Dense(YDim, HDim);

		b0p = Matrix<float>.Build.Dense(HDim, 1);
		b1p = Matrix<float>.Build.Dense(HDim, 1);
		b2p = Matrix<float>.Build.Dense(YDim, 1);
	}

	public void LoadParameters() {
		Parameters = ScriptableObject.CreateInstance<NetworkParameters>();
		if(!Parameters.Load(Folder, XDim, YDim, HDim)) {
			Parameters = null;
			Debug.Log("Failed loading parameters.");
			return;
		} else {
			Debug.Log("Parameters successfully loaded.");
		}
		if(Application.isPlaying) {
			Initialise();
		}
	}

	public void SetInput(int i, float value) {
		Xp[i, 0] = value;
	}

	public float GetOutput(int i) {
		return Yp[i, 0];
	}

	public void Output() {
		Debug.Log("====================INPUT====================");
		for(int i=0; i<XDim; i++) {
			Debug.Log(i + ": " + Xp[i, 0]);
		}
		Debug.Log("====================OUTPUT====================");
		for(int i=0; i<YDim; i++) {
			Debug.Log(i + ": " + Yp[i, 0]);
		}
	}

	public Matrix<float> Predict(float phase) {
		//float pamount;
		//int pindex_0;
		int pindex_1;
		//int pindex_2;
		//int pindex_3;

		Matrix<float> _Xp = Xp.Clone();
		
		_Xp = (_Xp - Xmean).PointwiseDivide(Xstd);
		
		//switch(Mode) {
		//	case MODE.CONSTANT:
				pindex_1 = (int)((phase / (2*M_PI)) * 50);
				H0 = (W0[pindex_1] * _Xp) + b0[pindex_1];
				ELU(ref H0);
				H1 = (W1[pindex_1] * H0) + b1[pindex_1];
				ELU(ref H1);
				Yp = (W2[pindex_1] * H1) + b2[pindex_1];
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
		
		Yp = (Yp.PointwiseMultiply(Ystd)) + Ymean;

		return Yp;
	}

	private void ELU(ref Matrix<float> m) {
		for(int x=0; x<m.RowCount; x++) {
			for(int y=0; y<m.ColumnCount; y++) {
				m[x,y] = System.Math.Max(m[x,y], 0f) + (float)System.Math.Exp(System.Math.Min(m[x,y], 0f)) - 1f;
			}
		}
	}

	private void Linear(ref Matrix<float> o, ref Matrix<float> y0, ref Matrix<float> y1, float mu) {
		o = (1.0f-mu) * y0 + (mu) * y1;
	}

	private void Cubic(ref Matrix<float> o, ref Matrix<float> y0, ref Matrix<float> y1, ref Matrix<float> y2, ref Matrix<float> y3, float mu) {
		o = (
		(-0.5f*y0+1.5f*y1-1.5f*y2+0.5f*y3)*mu*mu*mu + 
		(y0-2.5f*y1+2.0f*y2-0.5f*y3)*mu*mu + 
		(-0.5f*y0+0.5f*y2)*mu + 
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