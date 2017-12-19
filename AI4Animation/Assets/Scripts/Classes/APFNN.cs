using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class APFNN {

	public bool Inspect = false;

	public string Folder = string.Empty;
	
	public int MLPDim = 48;
	public int XDim = 504;
	public int HDim = 512;
	public int YDim = 352;

	private Matrix MLPXmean;
	private Matrix MLPXstd;
	private Matrix MLPYmean;
	private Matrix MLPYstd;

	private Matrix PFNNXmean;
	private Matrix PFNNXstd;
	private Matrix PFNNYmean;
	private Matrix PFNNYstd;

	private Matrix MLPW0, MLPW1, MLPW2;
	private Matrix MLPb0, MLPb1, MLPb2;

	private Matrix[] CPa0, CPa1, CPa2;
	private Matrix[] CPb0, CPb1, CPb2;

	private Matrix MLPX;
	private Matrix MLPY;

	private Matrix PFNNX;
	private Matrix PFNNY;

	public NetworkParameters Parameters;

	public APFNN() {
		
	}

	public void Initialise() {
		if(Parameters == null) {
			Debug.Log("Building PFNN failed because no parameters were loaded.");
			return;
		}

		

	}

	public void LoadParameters() {
		Parameters = ScriptableObject.CreateInstance<NetworkParameters>();
		Parameters.StoreMatrix(Folder+"/Xmean.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Xstd.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Ymean.bin", YDim, 1);
		Parameters.StoreMatrix(Folder+"/Ystd.bin", YDim, 1);
		Parameters.StoreMatrix(Folder+"/Xmean_hands.bin", MLPDim, 1);
		Parameters.StoreMatrix(Folder+"/Xstd_hands.bin", MLPDim, 1);

		Parameters.StoreMatrix(Folder+"/wc0_w.bin", MLPDim, MLPDim);
		Parameters.StoreMatrix(Folder+"/wc0_b.bin", MLPDim, MLPDim);
		Parameters.StoreMatrix(Folder+"/wc1_w.bin", MLPDim, MLPDim);
		Parameters.StoreMatrix(Folder+"/wc1_b.bin", MLPDim, MLPDim);
		Parameters.StoreMatrix(Folder+"/wc2_w.bin", MLPDim, 4);
		Parameters.StoreMatrix(Folder+"/wc2_b.bin", MLPDim, 4);

		for(int i=0; i<4; i++) {
			Parameters.StoreMatrix(Folder+"/cp0_a"+i.ToString("D3")+".bin", XDim, HDim);
			Parameters.StoreMatrix(Folder+"/cp0_b"+i.ToString("D3")+".bin", 1, HDim);

			Parameters.StoreMatrix(Folder+"/cp1_a"+i.ToString("D3")+".bin", HDim, HDim);
			Parameters.StoreMatrix(Folder+"/cp1_b"+i.ToString("D3")+".bin", 1, HDim);

			Parameters.StoreMatrix(Folder+"/cp2_a"+i.ToString("D3")+".bin", HDim, YDim);
			Parameters.StoreMatrix(Folder+"/cp2_b"+i.ToString("D3")+".bin", 1, YDim);
		}
	}

	public void SetMLPInput(int index, float value) {
		MLPX.Values[index][0] = value; 
	}

	public float GetMLPOutput(int index) {
		return MLPY.Values[index][0];
	}

	public void SetPFNNInput(int index, float value) {
		PFNNX.Values[index][0] = value;
	}

	public float GetPFNNOutput(int index) {
		return PFNNY.Values[index][0];
	}

	public void Predict() {
		//Process MLP
		Matrix _MLPX = (MLPX - MLPXmean).PointwiseDivide(MLPXstd);
		Matrix H0 = (_MLPX * MLPW0) + MLPb0; ELU(ref H0);
		Matrix H1 = (H0 * MLPW1) + MLPb1; ELU(ref H1);
		MLPY = (H1 * MLPW2) + MLPb2; SoftMax(ref MLPY);

		//Control Points
		Matrix PFNNW0 = new Matrix(XDim, HDim);
		Matrix PFNNW1 = new Matrix(HDim, HDim);
		Matrix PFNNW2 = new Matrix(HDim, YDim);
		Matrix PFNNb0 = new Matrix(1, HDim);
		Matrix PFNNb1 = new Matrix(1, HDim);
		Matrix PFNNb2 = new Matrix(1, YDim);
		for(int i=0; i<4; i++) {
			PFNNW0 += CPa0[i] * MLPY.Values[i][0];
			PFNNW1 += CPa1[i] * MLPY.Values[i][0];
			PFNNW2 += CPa2[i] * MLPY.Values[i][0];
			PFNNb0 += CPb0[i] * MLPY.Values[i][0];
			PFNNb1 += CPb1[i] * MLPY.Values[i][0];
			PFNNb2 += CPb2[i] * MLPY.Values[i][0];
		}

		//Process PFNN
		Matrix _PFNNX = (PFNNX - PFNNXmean).PointwiseDivide(PFNNXstd);
		H0 = (_PFNNX * PFNNW0) + PFNNb0; ELU(ref H0);
		H1 = (H0 * PFNNW1) + PFNNb1; ELU(ref H1);
		PFNNY = (H1 * PFNNW2) + PFNNb2;
		PFNNY = (PFNNY.PointwiseMultiply(PFNNYstd)) + PFNNYmean;
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
					MLPDim = EditorGUILayout.IntField("MLPDim", MLPDim);
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