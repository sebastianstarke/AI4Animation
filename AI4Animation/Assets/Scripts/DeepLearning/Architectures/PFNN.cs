using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class PFNN {

	public bool Inspect = false;

	public string Folder = string.Empty;
	public int XDim = 504;
	public int HDim = 512;
	public int YDim = 352;

	public NetworkParameters Parameters;

	private Matrix Xmean, Xstd, Ymean, Ystd;

	private Matrix[] W0, W1, W2, b0, b1, b2;

	private Matrix X, Y;

	private float Phase;

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

		Phase = 0f;
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

	public void SetPhase(float value) {
		Phase = value;
	}

	public float GetPhase() {
		return Phase;
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

	public void Predict() {
		int index = (int)((Phase / (2f*M_PI)) * 50f);
		Matrix _X = (X - Xmean).PointwiseDivide(Xstd);
		Matrix H0 = (W0[index] * _X) + b0[index]; ELU(ref H0);
		Matrix H1 = (W1[index] * H0) + b1[index]; ELU(ref H1);
		Y = (W2[index] * H1) + b2[index];
		Y = (Y.PointwiseMultiply(Ystd)) + Ymean;
	}

	private void ELU(ref Matrix m) {
		for(int x=0; x<m.Values.Length; x++) {
			m.Values[x][0] = System.Math.Max(m.Values[x][0], 0f) + (float)System.Math.Exp(System.Math.Min(m.Values[x][0], 0f)) - 1f;
		}
	}

	private void SoftMax(ref Matrix m) {
		for(int x=0; x<m.Values.Length; x++) {
			m.Values[x][0] = (float)System.Math.Exp(m.Values[x][0]);
		}
		float lower = 0f;
		for(int x=0; x<m.Values.Length; x++) {
			lower += m.Values[x][0];
		}
		for(int x=0; x<m.Values.Length; x++) {
			m.Values[x][0] /= lower;
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
			if(Utility.GUIButton("PFNN", UltiDraw.DarkGrey, UltiDraw.White)) {
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