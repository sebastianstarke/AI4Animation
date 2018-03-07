using UnityEngine;
using System;
using System.Runtime.InteropServices;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class MFNN {

	public bool Inspect = false;

	public string Folder = string.Empty;
	
	public int CDim = 12;
	public int XDim = 504;
	public int HDim = 512;
	public int YDim = 352;
	public int ControlWeights = 4;
	public int[] ControlNeurons = new int[0];

	public NetworkParameters Parameters;
	
	private IntPtr Network;

    [DllImport("MFNN")]
    private static extern IntPtr Create();
    [DllImport("MFNN")]
    private static extern IntPtr Delete(IntPtr obj);
    [DllImport("MFNN")]
    private static extern void Initialise(IntPtr obj, int cDim, int xDim, int hDim, int yDim, int controlWeights);
    [DllImport("MFNN")]
    private static extern void SetValue(IntPtr obj, int matrix, int row, int col, float value);
    [DllImport("MFNN")]
    private static extern float GetValue(IntPtr obj, int matrix, int row, int col);
    [DllImport("MFNN")]
    private static extern float AddControlNeuron(IntPtr obj, int index);
    [DllImport("MFNN")]
    private static extern float IgnoreControlNeuron(IntPtr obj, int index, bool value);
    [DllImport("MFNN")]
    private static extern void Predict(IntPtr obj);

	public MFNN() {
		Network = Create();
	}

	~MFNN() {
		Delete(Network);
	}

	public void LoadParameters() {
		Parameters = ScriptableObject.CreateInstance<NetworkParameters>();
		Parameters.StoreMatrix(Folder+"/Xmean.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Xstd.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Ymean.bin", YDim, 1);
		Parameters.StoreMatrix(Folder+"/Ystd.bin", YDim, 1);

		Parameters.StoreMatrix(Folder+"/wc0_w.bin", CDim, CDim);
		Parameters.StoreMatrix(Folder+"/wc0_b.bin", CDim, 1);

		Parameters.StoreMatrix(Folder+"/wc1_w.bin", CDim, CDim);
		Parameters.StoreMatrix(Folder+"/wc1_b.bin", CDim, 1);
		
		Parameters.StoreMatrix(Folder+"/wc2_w.bin", ControlWeights, CDim);
		Parameters.StoreMatrix(Folder+"/wc2_b.bin", ControlWeights, 1);

		for(int i=0; i<ControlWeights; i++) {
			Parameters.StoreMatrix(Folder+"/cp0_a"+i.ToString("D1")+".bin", HDim, XDim);
			Parameters.StoreMatrix(Folder+"/cp0_b"+i.ToString("D1")+".bin", HDim, 1);

			Parameters.StoreMatrix(Folder+"/cp1_a"+i.ToString("D1")+".bin", HDim, HDim);
			Parameters.StoreMatrix(Folder+"/cp1_b"+i.ToString("D1")+".bin", HDim, 1);

			Parameters.StoreMatrix(Folder+"/cp2_a"+i.ToString("D1")+".bin", YDim, HDim);
			Parameters.StoreMatrix(Folder+"/cp2_b"+i.ToString("D1")+".bin", YDim, 1);
		}
	}

	public void Initialise() {
		if(Parameters == null) {
			Debug.Log("Building MFNN failed because no parameters were loaded.");
			return;
		}
		Initialise(Network, CDim, XDim, HDim, YDim, ControlWeights);
		for(int i=0; i<ControlNeurons.Length; i++) {
			AddControlNeuron(ControlNeurons[i]);
		}
		for(int i=0; i<Parameters.Matrices.Length; i++) {
			SetupMatrix(i);
		}
	}

	private void SetupMatrix(int index) {
		NetworkParameters.FloatMatrix matrix = Parameters.GetMatrix(index);
		for(int i=0; i<matrix.Rows; i++) {
			for(int j=0; j<matrix.Cols; j++) {
				SetValue(Network, index, i, j, matrix.Values[i].Values[j]);
			}
		}	
	}

	public void SetInput(int index, float value) {
		if(Parameters == null) {
			return;
		}
		SetValue(Network, 10+ControlWeights*6, index, 0, value);
	}

	public float GetOutput(int index) {
		if(Parameters == null) {
			return 0f;
		}
		return GetValue(Network, 10+ControlWeights*6+1, index, 0);
	}

	public void AddControlNeuron(int index) {
		if(Parameters == null) {
			return;
		}
		AddControlNeuron(Network, index);
	}

	public void IgnoreControlNeuron(int index, bool value) {
		if(Parameters == null) {
			return;
		}
		IgnoreControlNeuron(Network, index, value);
	}

	public float GetControlPoint(int index) {
		if(Parameters == null) {
			return 0f;
		}
		return GetValue(Network, 10+ControlWeights*6+2, index, 0);
	}

	public void Predict() {
		if(Parameters == null) {
			return;
		}
		Predict(Network);
	}

	#if UNITY_EDITOR
	public void Inspector() {
		Utility.SetGUIColor(Color.grey);
		using(new GUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Utility.GUIButton("MFNN", UltiDraw.DarkGrey, UltiDraw.White)) {
				Inspect = !Inspect;
			}

			if(Inspect) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Folder = EditorGUILayout.TextField("Folder", Folder);
					CDim = EditorGUILayout.IntField("CDim", CDim);
					XDim = EditorGUILayout.IntField("XDim", XDim);
					HDim = EditorGUILayout.IntField("HDim", HDim);
					YDim = EditorGUILayout.IntField("YDim", YDim);
					ControlWeights = EditorGUILayout.IntField("Control Weights", ControlWeights);
					Array.Resize(ref ControlNeurons, EditorGUILayout.IntField("Control Neurons", ControlNeurons.Length));
					for(int i=0; i<ControlNeurons.Length; i++) {
						ControlNeurons[i] = EditorGUILayout.IntField("Neuron " + (i+1), ControlNeurons[i]);
					}
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