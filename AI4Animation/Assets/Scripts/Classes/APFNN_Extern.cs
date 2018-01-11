using UnityEngine;
using System;
using System.Runtime.InteropServices;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class APFNN_Extern {

	public bool Inspect = false;

	public string Folder = string.Empty;
	
	public int MLPDim = 12;
	public int XDim = 504;
	public int HDim = 512;
	public int YDim = 352;

	public NetworkParameters Parameters;
	
	private IntPtr Network;

    [DllImport("APFNN")]
    private static extern IntPtr Create();
    [DllImport("APFNN")]
    private static extern IntPtr Delete(IntPtr obj);
    [DllImport("APFNN")]
    private static extern void Initialise(IntPtr obj, int cDim, int xDim, int hDim, int yDim);

	public APFNN_Extern() {
		Network = Create();
	}

	~APFNN_Extern() {
		Delete(Network);
	}

	public void Initialise() {
		if(Parameters == null) {
			Debug.Log("Building PFNN failed because no parameters were loaded.");
			return;
		}
		Initialise(Network, 12, 200, 200, 200);

	}

	//IntPtr RetrieveMatrix(int index) {
		/*
		NetworkParameters.FloatMatrix matrix = Parameters.GetMatrix(index);
		IntPtr ptr = Create(matrix.Rows, matrix.Cols);
		for(int i=0; i<matrix.Rows; i++) {
			for(int j=0; j<matrix.Cols; j++) {
				SetValue(ptr, i, j, matrix.Values[i].Values[j]);
			}
		}
		return ptr;
		*/
	//}

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
		//SetValue(MLPX, index, 0, value);
	}

	public float GetMLPInput(int index) {
		return 0f;
		//return GetValue(MLPX, index, 0);
	}

	public float GetMLPOutput(int index) {
		return 0f;
		//return GetValue(MLPY, index, 0);
	}

	public void SetPFNNInput(int index, float value) {
		//SetValue(PFNNX, index, 0, value);
	}

	public float GetPFNNInput(int index) {
		return 0f;
		//return GetValue(PFNNX, index, 0);
	}

	public float GetPFNNOutput(int index) {
		return 0f;
		//return GetValue(PFNNY, index, 0);
	}

	public void Predict() {

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