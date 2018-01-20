using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class MLP {

	public bool Inspect = false;

	public string Folder = string.Empty;
	public int XDim = 0;
	public int HDim = 512;
	public int YDim = 0;

	public NetworkParameters Parameters;

	private Matrix Xmean, Xstd, Ymean, Ystd;

	private Matrix W0, W1, W2, b0, b1, b2;

	private Matrix X, Y;

	public MLP() {
		
	}

	public void Initialise() {
		if(Parameters == null) {
			Debug.Log("Building MLP failed because no parameters were loaded.");
			return;
		}

		Xmean = Parameters.GetMatrix(0).Build();
		Xstd = Parameters.GetMatrix(1).Build();
		Ymean = Parameters.GetMatrix(2).Build();
		Ystd = Parameters.GetMatrix(3).Build();

		W0 = Parameters.GetMatrix(4).Build();
		W1 = Parameters.GetMatrix(5).Build();
		W2 = Parameters.GetMatrix(6).Build();
		b0 = Parameters.GetMatrix(7).Build();
		b1 = Parameters.GetMatrix(8).Build();
		b2 = Parameters.GetMatrix(9).Build();

		X = new Matrix(XDim, 1);
		Y = new Matrix(YDim, 1);
	}

	public void LoadParameters() {
		Parameters = ScriptableObject.CreateInstance<NetworkParameters>();
		Parameters.StoreMatrix(Folder+"/Xmean.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Xstd.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Ymean.bin", YDim, 1);
		Parameters.StoreMatrix(Folder+"/Ystd.bin", YDim, 1);
		Parameters.StoreMatrix(Folder+"/W0_000.bin", HDim, XDim);
		Parameters.StoreMatrix(Folder+"/W1_000.bin", HDim, HDim);
		Parameters.StoreMatrix(Folder+"/W2_000.bin", YDim, HDim);
		Parameters.StoreMatrix(Folder+"/b0_000.bin", HDim, 1);
		Parameters.StoreMatrix(Folder+"/b1_000.bin", HDim, 1);
		Parameters.StoreMatrix(Folder+"/b2_000.bin", YDim, 1);
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
		Matrix _X = (X - Xmean).PointwiseDivide(Xstd);
		Matrix H0 = (W0 * _X) + b0; ELU(ref H0);
		Matrix H1 = (W1 * H0) + b1; ELU(ref H1);
		Y = (W2 * H1) + b2;
		Y = (Y.PointwiseMultiply(Ystd)) + Ymean;
	}

	private void ELU(ref Matrix m) {
		for(int x=0; x<m.Values.Length; x++) {
			m.Values[x][0] = System.Math.Max(m.Values[x][0], 0f) + (float)System.Math.Exp(System.Math.Min(m.Values[x][0], 0f)) - 1f;
		}
	}

	#if UNITY_EDITOR
	public void Inspector() {
		Utility.SetGUIColor(Color.grey);
		using(new GUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Utility.GUIButton("MLP", Utility.DarkGrey, Utility.White)) {
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