using UnityEngine;
using System;
using System.Runtime.InteropServices;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class APFNN_Eigen {

	public bool Inspect = false;

	public string Folder = string.Empty;
	
	public int MLPDim = 12;
	public int XDim = 504;
	public int HDim = 512;
	public int YDim = 352;

	private IntPtr PFNNXmean;
	private IntPtr PFNNXstd;
	private IntPtr PFNNYmean;
	private IntPtr PFNNYstd;

	private IntPtr MLPXmean;
	private IntPtr MLPXstd;

	private IntPtr MLPW0, MLPW1, MLPW2;
	private IntPtr MLPb0, MLPb1, MLPb2;

	private IntPtr[] CPa0, CPa1, CPa2;
	private IntPtr[] CPb0, CPb1, CPb2;

	private IntPtr MLPX;
	private IntPtr MLPY;

	private IntPtr PFNNX;
	private IntPtr PFNNY;

	//TMP
	private IntPtr _MLPX, _MLPH0, _MLPH1;
	private IntPtr _PFNNX, _PFNNH0, _PFNNH1;
	private IntPtr _PFNNW0, _PFNNW1, _PFNNW2, _PFNNb0, _PFNNb1, _PFNNb2;
	private IntPtr __PFNNW0, __PFNNW1, __PFNNW2, __PFNNb0, __PFNNb1, __PFNNb2;

	public NetworkParameters Parameters;

    [DllImport("Eigen")]
    private static extern IntPtr Create(int rows, int cols);
    [DllImport("Eigen")]
    private static extern IntPtr Delete(IntPtr m);
    [DllImport("Eigen")]
    private static extern void Add(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("Eigen")]
    private static extern void Sub(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("Eigen")]
    private static extern void Multiply(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("Eigen")]
    private static extern void Scale(IntPtr lhs, float value, IntPtr result);
    [DllImport("Eigen")]
    private static extern void PointwiseMultiply(IntPtr m, IntPtr value);
    [DllImport("Eigen")]
    private static extern void PointwiseDivide(IntPtr m, IntPtr value);
    [DllImport("Eigen")]
    private static extern void SetValue(IntPtr m, int row, int col, float value);
    [DllImport("Eigen")]
    private static extern float GetValue(IntPtr m, int row, int col);
    [DllImport("Eigen")]
    private static extern void ELU(IntPtr m);
    [DllImport("Eigen")]
    private static extern void SoftMax(IntPtr m);
	[DllImport("Eigen")]
    private static extern void Clear(IntPtr m);

	public APFNN_Eigen() {
		
	}

	public void Initialise() {
		if(Parameters == null) {
			Debug.Log("Building PFNN failed because no parameters were loaded.");
			return;
		}

		PFNNXmean = RetrieveMatrix(0);
		PFNNXstd = RetrieveMatrix(1);
		PFNNYmean = RetrieveMatrix(2);
		PFNNYstd = RetrieveMatrix(3);
		MLPXmean = RetrieveMatrix(4);
		MLPXstd = RetrieveMatrix(5);

		MLPW0 = RetrieveMatrix(6);
		MLPb0 = RetrieveMatrix(7);
		MLPW1 = RetrieveMatrix(8);
		MLPb1 = RetrieveMatrix(9);
		MLPW2 = RetrieveMatrix(10);
		MLPb2 = RetrieveMatrix(11);

		CPa0 = new IntPtr[4];
		CPb0 = new IntPtr[4];
		CPa1 = new IntPtr[4];
		CPb1 = new IntPtr[4];
		CPa2 = new IntPtr[4];
		CPb2 = new IntPtr[4];
		for(int i=0; i<4; i++) {
			CPa0[i] = RetrieveMatrix(12 + i*6 + 0);
			CPb0[i] = RetrieveMatrix(12 + i*6 + 1);
			CPa1[i] = RetrieveMatrix(12 + i*6 + 2);
			CPb1[i] = RetrieveMatrix(12 + i*6 + 3);
			CPa2[i] = RetrieveMatrix(12 + i*6 + 4);
			CPb2[i] = RetrieveMatrix(12 + i*6 + 5);
		}

		MLPX = Create(MLPDim, 1);
		MLPY = Create(4, 1);

		PFNNX = Create(XDim, 1);
		PFNNY = Create(YDim, 1);

		//TMP
		_MLPX = Create(MLPDim, 1);
		_MLPH0 = Create(Parameters.GetMatrix(7).Rows, Parameters.GetMatrix(7).Cols);
		_MLPH1 = Create(Parameters.GetMatrix(9).Rows, Parameters.GetMatrix(9).Cols);

		_PFNNX = Create(XDim, 1);
		_PFNNH0 = Create(HDim, 1);
		_PFNNH1 = Create(HDim, 1);

		_PFNNW0 = Create(HDim, XDim);
		_PFNNW1 = Create(HDim, HDim);
		_PFNNW2 = Create(YDim, HDim);
		_PFNNb0 = Create(HDim, 1);
		_PFNNb1 = Create(HDim, 1);
		_PFNNb2 = Create(YDim, 1);

		__PFNNW0 = Create(HDim, XDim);
		__PFNNW1 = Create(HDim, HDim);
		__PFNNW2 = Create(YDim, HDim);
		__PFNNb0 = Create(HDim, 1);
		__PFNNb1 = Create(HDim, 1);
		__PFNNb2 = Create(YDim, 1);
	}

	IntPtr RetrieveMatrix(int index) {
		NetworkParameters.FloatMatrix matrix = Parameters.GetMatrix(index);
		IntPtr ptr = Create(matrix.Rows, matrix.Cols);
		for(int i=0; i<matrix.Rows; i++) {
			for(int j=0; j<matrix.Cols; j++) {
				SetValue(ptr, i, j, matrix.Values[i].Values[j]);
			}
		}
		return ptr;
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
		SetValue(MLPX, index, 0, value);
	}

	public float GetMLPInput(int index) {
		return GetValue(MLPX, index, 0);
	}

	public float GetMLPOutput(int index) {
		return GetValue(MLPY, index, 0);
	}

	public void SetPFNNInput(int index, float value) {
		SetValue(PFNNX, index, 0, value);
	}

	public float GetPFNNInput(int index) {
		return GetValue(PFNNX, index, 0);
	}

	public float GetPFNNOutput(int index) {
		return GetValue(PFNNY, index, 0);
	}

	public void Predict() {
		IntPtr ptr = Create(10000, 10000);

		//Process MLP
		Sub(MLPX, MLPXmean, _MLPX); PointwiseDivide(_MLPX, MLPXstd);
		Multiply(MLPW0, _MLPX, _MLPH0); Add(_MLPH0, MLPb0, _MLPH0); ELU(_MLPH0);
		Multiply(MLPW1, _MLPH0, _MLPH1); Add(_MLPH1, MLPb1, _MLPH1); ELU(_MLPH1);
		Multiply(MLPW2, _MLPH1, MLPY); Add(MLPY, MLPb2, MLPY); SoftMax(MLPY);

		//Control Points
		Clear(_PFNNW0); Clear(_PFNNW1); Clear(_PFNNW2);
		Clear(_PFNNb0); Clear(_PFNNb1); Clear(_PFNNb2);
		for(int i=0; i<4; i++) {
			Scale(CPa0[i], GetValue(MLPY, i, 0), __PFNNW0); Add(_PFNNW0, __PFNNW0, _PFNNW0);
			Scale(CPa1[i], GetValue(MLPY, i, 0), __PFNNW1); Add(_PFNNW1, __PFNNW1, _PFNNW1);
			Scale(CPa2[i], GetValue(MLPY, i, 0), __PFNNW2); Add(_PFNNW2, __PFNNW2, _PFNNW2);
			Scale(CPb0[i], GetValue(MLPY, i, 0), __PFNNb0); Add(_PFNNb0, __PFNNb0, _PFNNb0);
			Scale(CPb1[i], GetValue(MLPY, i, 0), __PFNNb1); Add(_PFNNb1, __PFNNb1, _PFNNb1);
			Scale(CPb2[i], GetValue(MLPY, i, 0), __PFNNb2); Add(_PFNNb2, __PFNNb2, _PFNNb2);
		}

		//Process PFNN
		Sub(PFNNX, PFNNXmean, _PFNNX); PointwiseDivide(_PFNNX, PFNNXstd);
		Multiply(_PFNNW0, _PFNNX, _PFNNH0); Add(_PFNNH0, _PFNNb0, _PFNNH0); ELU(_PFNNH0);
		Multiply(_PFNNW1, _PFNNH0, _PFNNH1); Add(_PFNNH1, _PFNNb1, _PFNNH1); ELU(_PFNNH1);
		Multiply(_PFNNW2, _PFNNH1, PFNNY); Add(PFNNY, _PFNNb2, PFNNY);
		PointwiseMultiply(PFNNY, PFNNYstd); Add(PFNNY, PFNNYmean, PFNNY);
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