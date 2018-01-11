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

	private Matrix PFNNXmean;
	private Matrix PFNNXstd;
	private Matrix PFNNYmean;
	private Matrix PFNNYstd;

	private Matrix MLPXmean;
	private Matrix MLPXstd;

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

		PFNNXmean = Parameters.GetMatrix(0).Build();
		PFNNXstd = Parameters.GetMatrix(1).Build();
		PFNNYmean = Parameters.GetMatrix(2).Build();
		PFNNYstd = Parameters.GetMatrix(3).Build();
		MLPXmean = Parameters.GetMatrix(4).Build();
		MLPXstd = Parameters.GetMatrix(5).Build();

		MLPW0 = Parameters.GetMatrix(6).Build();
		MLPb0 = Parameters.GetMatrix(7).Build();
		MLPW1 = Parameters.GetMatrix(8).Build();
		MLPb1 = Parameters.GetMatrix(9).Build();
		MLPW2 = Parameters.GetMatrix(10).Build();
		MLPb2 = Parameters.GetMatrix(11).Build();

		CPa0 = new Matrix[4];
		CPb0 = new Matrix[4];
		CPa1 = new Matrix[4];
		CPb1 = new Matrix[4];
		CPa2 = new Matrix[4];
		CPb2 = new Matrix[4];
		for(int i=0; i<4; i++) {
			CPa0[i] = Parameters.GetMatrix(12 + i*6 + 0).Build();
			CPb0[i] = Parameters.GetMatrix(12 + i*6 + 1).Build();
			CPa1[i] = Parameters.GetMatrix(12 + i*6 + 2).Build();
			CPb1[i] = Parameters.GetMatrix(12 + i*6 + 3).Build();
			CPa2[i] = Parameters.GetMatrix(12 + i*6 + 4).Build();
			CPb2[i] = Parameters.GetMatrix(12 + i*6 + 5).Build();
		}

		MLPX = new Matrix(MLPDim, 1);
		MLPY = new Matrix(4, 1);

		PFNNX = new Matrix(XDim, 1);
		PFNNY = new Matrix(YDim, 1);
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
		Parameters.StoreMatrix(Folder+"/wc0_b.bin", MLPDim, 1);

		Parameters.StoreMatrix(Folder+"/wc1_w.bin", MLPDim, MLPDim);
		Parameters.StoreMatrix(Folder+"/wc1_b.bin", MLPDim, 1);
		
		Parameters.StoreMatrix(Folder+"/wc2_w.bin", 4, MLPDim);
		Parameters.StoreMatrix(Folder+"/wc2_b.bin", 4, 1);

		for(int i=0; i<4; i++) {
			Parameters.StoreMatrix(Folder+"/cp0_a"+i.ToString("D1")+".bin", HDim, XDim);
			Parameters.StoreMatrix(Folder+"/cp0_b"+i.ToString("D1")+".bin", HDim, 1);

			Parameters.StoreMatrix(Folder+"/cp1_a"+i.ToString("D1")+".bin", HDim, HDim);
			Parameters.StoreMatrix(Folder+"/cp1_b"+i.ToString("D1")+".bin", HDim, 1);

			Parameters.StoreMatrix(Folder+"/cp2_a"+i.ToString("D1")+".bin", YDim, HDim);
			Parameters.StoreMatrix(Folder+"/cp2_b"+i.ToString("D1")+".bin", YDim, 1);
		}
	}

	public void SetMLPInput(int index, float value) {
		MLPX.Values[index][0] = value; 
	}

	public float GetMLPInput(int index) {
		return MLPX.Values[index][0];
	}

	public float GetMLPOutput(int index) {
		return MLPY.Values[index][0];
	}

	public void SetPFNNInput(int index, float value) {
		PFNNX.Values[index][0] = value;
	}

	public float GetPFNNInput(int index) {
		return PFNNX.Values[index][0];
	}

	public float GetPFNNOutput(int index) {
		return PFNNY.Values[index][0];
	}

	public void Predict() {
		//Process MLP
		Matrix _MLPX = (MLPX - MLPXmean).PointwiseDivide(MLPXstd);
		Matrix H0 = (MLPW0 * _MLPX) + MLPb0; ELU(ref H0);
		Matrix H1 = (MLPW1 * H0) + MLPb1; ELU(ref H1);
		MLPY = (MLPW2 * H1) + MLPb2; SoftMax(ref MLPY);

		//Control Points
		Matrix PFNNW0 = new Matrix(HDim, XDim);
		Matrix PFNNW1 = new Matrix(HDim, HDim);
		Matrix PFNNW2 = new Matrix(YDim, HDim);
		Matrix PFNNb0 = new Matrix(HDim, 1);
		Matrix PFNNb1 = new Matrix(HDim, 1);
		Matrix PFNNb2 = new Matrix(YDim, 1);
		for(int i=0; i<4; i++) {
			PFNNW0 += CPa0[i] * MLPY.Values[i][0];
			PFNNW1 += CPa1[i] * MLPY.Values[i][0];
			PFNNW2 += CPa2[i] * MLPY.Values[i][0];
			PFNNb0 += CPb0[i] * MLPY.Values[i][0];
			PFNNb1 += CPb1[i] * MLPY.Values[i][0];
			PFNNb2 += CPb2[i] * MLPY.Values[i][0];
		}

		//Debug.Log(MLPY.Values[0][0].ToString("F5") + " " +  MLPY.Values[1][0].ToString("F5") + " " + MLPY.Values[2][0].ToString("F5") + " " + MLPY.Values[3][0].ToString("F5"));

		//Process PFNN
		Matrix _PFNNX = (PFNNX - PFNNXmean).PointwiseDivide(PFNNXstd);
		H0 = (PFNNW0 * _PFNNX) + PFNNb0; ELU(ref H0);
		H1 = (PFNNW1 * H0) + PFNNb1; ELU(ref H1);
		PFNNY = (PFNNW2 * H1) + PFNNb2;
		PFNNY = (PFNNY.PointwiseMultiply(PFNNYstd)) + PFNNYmean;
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