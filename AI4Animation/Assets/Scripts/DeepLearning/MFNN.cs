using UnityEngine;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class MFNN {

	public bool Inspect = false;

	public string Folder = string.Empty;
	
	public int XDimBlend = 12;
	public int HDimBlend = 12;
	public int YDimBlend = 4;
	public int XDim = 504;
	public int HDim = 512;
	public int YDim = 352;
	public int[] ControlNeurons = new int[0];

	public NetworkParameters Parameters;

	private IntPtr Xmean, Xstd, Ymean, Ystd;
	private IntPtr BW0, BW1, BW2, Bb0, Bb1, Bb2;
	private IntPtr[] M;
	private IntPtr X, Y;

	private IntPtr CN, CP;
	private IntPtr NNW0, NNW1, NNW2, NNb0, NNb1, NNb2;
	private IntPtr Temp;

	private List<IntPtr> Ptrs;

    [DllImport("EigenNN")]
    private static extern IntPtr Create(int rows, int cols);
    [DllImport("EigenNN")]
    private static extern IntPtr Delete(IntPtr m);
	[DllImport("EigenNN")]
    private static extern void SetZero(IntPtr m);
    [DllImport("EigenNN")]
    private static extern void Add(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void Sub(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void Multiply(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void Scale(IntPtr lhs, float value, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void PointwiseMultiply(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void PointwiseDivide(IntPtr lhs, IntPtr rhs, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void SetValue(IntPtr m, int row, int col, float value);
    [DllImport("EigenNN")]
    private static extern float GetValue(IntPtr m, int row, int col);
	[DllImport("EigenNN")]
    private static extern void Layer(IntPtr x, IntPtr y, IntPtr W, IntPtr b);
	[DllImport("EigenNN")]
    private static extern void Blend(IntPtr m, IntPtr W, float w, IntPtr result);
    [DllImport("EigenNN")]
    private static extern void ELU(IntPtr m);
    [DllImport("EigenNN")]
    private static extern void SoftMax(IntPtr m);

	public MFNN() {
		Ptrs = new List<IntPtr>();
	}

	~MFNN() {
		for(int i=0; i<Ptrs.Count; i++) {
			Delete(Ptrs[i]);
		}
	}

	public void LoadParameters() {
		Parameters = ScriptableObject.CreateInstance<NetworkParameters>();
		Parameters.StoreMatrix(Folder+"/Xmean.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Xstd.bin", XDim, 1);
		Parameters.StoreMatrix(Folder+"/Ymean.bin", YDim, 1);
		Parameters.StoreMatrix(Folder+"/Ystd.bin", YDim, 1);

		Parameters.StoreMatrix(Folder+"/wc0_w.bin", HDimBlend, XDimBlend);
		Parameters.StoreMatrix(Folder+"/wc0_b.bin", HDimBlend, 1);

		Parameters.StoreMatrix(Folder+"/wc1_w.bin", HDimBlend, HDimBlend);
		Parameters.StoreMatrix(Folder+"/wc1_b.bin", HDimBlend, 1);
		
		Parameters.StoreMatrix(Folder+"/wc2_w.bin", YDimBlend, HDimBlend);
		Parameters.StoreMatrix(Folder+"/wc2_b.bin", YDimBlend, 1);

		for(int i=0; i<YDimBlend; i++) {
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
		Xmean = Generate(Parameters.GetMatrix(0));
		Xstd = Generate(Parameters.GetMatrix(1));
		Ymean = Generate(Parameters.GetMatrix(2));
		Ystd = Generate(Parameters.GetMatrix(3));

		BW0 = Generate(Parameters.GetMatrix(4));
		Bb0 = Generate(Parameters.GetMatrix(5));
		BW1 = Generate(Parameters.GetMatrix(6));
		Bb1 = Generate(Parameters.GetMatrix(7));
		BW2 = Generate(Parameters.GetMatrix(8));
		Bb2 = Generate(Parameters.GetMatrix(9));

		M = new IntPtr[YDimBlend*6];
		for(int i=0; i<YDimBlend*6; i++) {
			M[i] = Generate(Parameters.GetMatrix(10+i));
		}
		
		X = Create(XDim, 1);
		Ptrs.Add(X);
		Y = Create(YDim, 1);
		Ptrs.Add(Y);

		CN = Create(ControlNeurons.Length, 1); Ptrs.Add(CN);
		CP = Create(YDimBlend, 1); Ptrs.Add(CP);
		NNW0 = Create(HDim, XDim); Ptrs.Add(NNW0);
		NNW1 = Create(HDim, HDim); Ptrs.Add(NNW1);
		NNW2 = Create(YDim, HDim); Ptrs.Add(NNW2);
		NNb0 = Create(HDim, 1); Ptrs.Add(NNb0);
		NNb1 = Create(HDim, 1); Ptrs.Add(NNb1);
		NNb2 = Create(YDim, 1); Ptrs.Add(NNb2);
		Temp = Create(1, 1); Ptrs.Add(Temp);
	}

	private IntPtr Generate(NetworkParameters.FloatMatrix matrix) {
		IntPtr ptr = Create(matrix.Rows, matrix.Cols);
		for(int x=0; x<matrix.Rows; x++) {
			for(int y=0; y<matrix.Cols; y++) {
				SetValue(ptr, x, y, matrix.Values[x].Values[y]);
			}
		}
		Ptrs.Add(ptr);
		return ptr;
	}

	public void SetInput(int index, float value) {
		SetValue(X, index, 0, value);
	}

	public float GetOutput(int index) {
		return GetValue(Y, index, 0);
	}

	public float GetControlPoint(int index) {
		return 0f;
	}

	public void Predict() {
        //Normalise input
		Sub(X, Xmean, Y);
		PointwiseDivide(Y, Xstd, Y);
		
        //Process Blending Network
        for(int i=0; i<ControlNeurons.Length; i++) {
            SetValue(CN, i, 0, GetValue(Y, ControlNeurons[i], 0));
        }
		Layer(CN, CP, BW0, Bb0); ELU(CP);
		Layer(CP, CP, BW1, Bb1); ELU(CP);
		Layer(CP, CP, BW2, Bb2); SoftMax(CP);

        //Control Points
		SetZero(NNW0); SetZero(NNW1); SetZero(NNW2);
		SetZero(NNb0); SetZero(NNb1); SetZero(NNb2);
		for(int i=0; i<YDimBlend; i++) {
			Blend(NNW0, M[6*i + 0], GetValue(CP, i, 0), NNW0);
			Blend(NNb0, M[6*i + 1], GetValue(CP, i, 0), NNb0);
			Blend(NNW1, M[6*i + 2], GetValue(CP, i, 0), NNW1);
			Blend(NNb1, M[6*i + 3], GetValue(CP, i, 0), NNb1);
			Blend(NNW2, M[6*i + 4], GetValue(CP, i, 0), NNW2);
			Blend(NNb2, M[6*i + 5], GetValue(CP, i, 0), NNb2);
		}

        //Process Mode-Functioned Network
		Layer(Y, Y, NNW0, NNb0); ELU(Y);
		Layer(Y, Y, NNW1, NNb1); ELU(Y);
		Layer(Y, Y, NNW2, NNb2);

        //Renormalise output
		PointwiseMultiply(Y, Ystd, Y);
		Add(Y, Ymean, Y);
	}
	
	/*
	private IntPtr Network;

    [DllImport("MFNN")]
    private static extern IntPtr Create();
    [DllImport("MFNN")]
    private static extern IntPtr Delete(IntPtr obj);
    [DllImport("MFNN")]
    private static extern void Initialise(IntPtr obj, int xDimBlend, int hDimBlend, int yDimBlend, int xDim, int hDim, int yDim);
    [DllImport("MFNN")]
    private static extern void SetValue(IntPtr obj, int matrix, int row, int col, float value);
    [DllImport("MFNN")]
    private static extern float GetValue(IntPtr obj, int matrix, int row, int col);
    [DllImport("MFNN")]
    private static extern float AddControlNeuron(IntPtr obj, int index);
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

		Parameters.StoreMatrix(Folder+"/wc0_w.bin", HDimBlend, XDimBlend);
		Parameters.StoreMatrix(Folder+"/wc0_b.bin", HDimBlend, 1);

		Parameters.StoreMatrix(Folder+"/wc1_w.bin", HDimBlend, HDimBlend);
		Parameters.StoreMatrix(Folder+"/wc1_b.bin", HDimBlend, 1);
		
		Parameters.StoreMatrix(Folder+"/wc2_w.bin", YDimBlend, HDimBlend);
		Parameters.StoreMatrix(Folder+"/wc2_b.bin", YDimBlend, 1);

		for(int i=0; i<YDimBlend; i++) {
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
		Initialise(Network, XDimBlend, HDimBlend, YDimBlend, XDim, HDim, YDim);
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
		if(index >= XDim) {
			Debug.Log("Setting out of bounds " + index + ".");
			return;
		}
		SetValue(Network, 10+YDimBlend*6, index, 0, value);
	}

	public float GetOutput(int index) {
		if(Parameters == null) {
			return 0f;
		}
		if(index >= YDim) {
			Debug.Log("Returning out of bounds " + index + ".");
			return 0f;
		}
		return GetValue(Network, 10+YDimBlend*6+1, index, 0);
	}

	public void AddControlNeuron(int index) {
		if(Parameters == null) {
			return;
		}
		AddControlNeuron(Network, index);
	}

	public float GetControlPoint(int index) {
		if(Parameters == null) {
			return 0f;
		}
		return GetValue(Network, 10+YDimBlend*6+2, index, 0);
	}

	public void Predict() {
		if(Parameters == null) {
			return;
		}
		Predict(Network);
	}
	*/

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
					XDimBlend = EditorGUILayout.IntField("XDimBlend", XDimBlend);
					HDimBlend = EditorGUILayout.IntField("HDimBlend", HDimBlend);
					YDimBlend = EditorGUILayout.IntField("YDimBlend", YDimBlend);
					XDim = EditorGUILayout.IntField("XDim", XDim);
					HDim = EditorGUILayout.IntField("HDim", HDim);
					YDim = EditorGUILayout.IntField("YDim", YDim);
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