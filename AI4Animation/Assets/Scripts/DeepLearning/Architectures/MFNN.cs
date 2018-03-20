using System;
using System.Runtime.InteropServices;
using UnityEngine;
using DeepLearning;
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

	public Parameters Parameters;

	private Tensor Xmean, Xstd, Ymean, Ystd;
	private Tensor X, Y;
	private Tensor BX, BY;
	private Tensor BW0, BW1, BW2, Bb0, Bb1, Bb2;
	private Tensor[] M;
	private Tensor NNW0, NNW1, NNW2, NNb0, NNb1, NNb2;

	public void LoadParameters() {
		Parameters = ScriptableObject.CreateInstance<Parameters>();
		Parameters.StoreParameters(Folder+"/Xmean.bin", XDim, 1);
		Parameters.StoreParameters(Folder+"/Xstd.bin", XDim, 1);
		Parameters.StoreParameters(Folder+"/Ymean.bin", YDim, 1);
		Parameters.StoreParameters(Folder+"/Ystd.bin", YDim, 1);

		Parameters.StoreParameters(Folder+"/wc0_w.bin", HDimBlend, XDimBlend);
		Parameters.StoreParameters(Folder+"/wc0_b.bin", HDimBlend, 1);

		Parameters.StoreParameters(Folder+"/wc1_w.bin", HDimBlend, HDimBlend);
		Parameters.StoreParameters(Folder+"/wc1_b.bin", HDimBlend, 1);
		
		Parameters.StoreParameters(Folder+"/wc2_w.bin", YDimBlend, HDimBlend);
		Parameters.StoreParameters(Folder+"/wc2_b.bin", YDimBlend, 1);

		for(int i=0; i<YDimBlend; i++) {
			Parameters.StoreParameters(Folder+"/cp0_a"+i.ToString("D1")+".bin", HDim, XDim);
			Parameters.StoreParameters(Folder+"/cp0_b"+i.ToString("D1")+".bin", HDim, 1);

			Parameters.StoreParameters(Folder+"/cp1_a"+i.ToString("D1")+".bin", HDim, HDim);
			Parameters.StoreParameters(Folder+"/cp1_b"+i.ToString("D1")+".bin", HDim, 1);

			Parameters.StoreParameters(Folder+"/cp2_a"+i.ToString("D1")+".bin", YDim, HDim);
			Parameters.StoreParameters(Folder+"/cp2_b"+i.ToString("D1")+".bin", YDim, 1);
		}
	}

	public void Initialise() {
		if(Parameters == null) {
			Debug.Log("Building MFNN failed because no parameters were loaded.");
			return;
		}
		Xmean = Parameters.GetParameters(0).MakeTensor();
		Xstd = Parameters.GetParameters(1).MakeTensor();
		Ymean = Parameters.GetParameters(2).MakeTensor();
		Ystd = Parameters.GetParameters(3).MakeTensor();

		BW0 = Parameters.GetParameters(4).MakeTensor();
		Bb0 = Parameters.GetParameters(5).MakeTensor();
		BW1 = Parameters.GetParameters(6).MakeTensor();
		Bb1 = Parameters.GetParameters(7).MakeTensor();
		BW2 = Parameters.GetParameters(8).MakeTensor();
		Bb2 = Parameters.GetParameters(9).MakeTensor();

		M = new Tensor[YDimBlend*6];
		for(int i=0; i<YDimBlend*6; i++) {
			M[i] = Parameters.GetParameters(10+i).MakeTensor();
		}
		
		X = new Tensor(XDim, 1);
		Y = new Tensor(YDim, 1);

		BX = new Tensor(ControlNeurons.Length, 1);
		BY = new Tensor(YDimBlend, 1);
		NNW0 = new Tensor(HDim, XDim);
		NNW1 = new Tensor(HDim, HDim);
		NNW2 = new Tensor(YDim, HDim);
		NNb0 = new Tensor(HDim, 1);
		NNb1 = new Tensor(HDim, 1);
		NNb2 = new Tensor(YDim, 1);
	}

	public void SetInput(int index, float value) {
		X.SetValue(index, 0, value);
	}

	public float GetOutput(int index) {
		return Y.GetValue(index, 0);
	}

	public float GetControlPoint(int index) {
		if(BY == null) {
			return 0f;
		}
		return BY.GetValue(index, 0);
	}

	public void Predict() {
	    //Normalise Input
		Model.Normalise(X, Xmean, Xstd, Y);

        //Process Blending Network
        for(int i=0; i<ControlNeurons.Length; i++) {
            BX.SetValue(i, 0, Y.GetValue(ControlNeurons[i], 0));
        }
		Model.ELU(Model.Layer(BX, BW0, Bb0, BY));
		Model.ELU(Model.Layer(BY, BW1, Bb1, BY));
		Model.SoftMax(Model.Layer(BY, BW2, Bb2, BY));

        //Generate Network Weights
		NNW0.SetZero();	NNb0.SetZero();
		NNW1.SetZero();	NNb1.SetZero();
		NNW2.SetZero();	NNb2.SetZero();
		for(int i=0; i<YDimBlend; i++) {
			Model.Blend(NNW0, M[6*i + 0], BY.GetValue(i, 0));
			Model.Blend(NNb0, M[6*i + 1], BY.GetValue(i, 0));
			Model.Blend(NNW1, M[6*i + 2], BY.GetValue(i, 0));
			Model.Blend(NNb1, M[6*i + 3], BY.GetValue(i, 0));
			Model.Blend(NNW2, M[6*i + 4], BY.GetValue(i, 0));
			Model.Blend(NNb2, M[6*i + 5], BY.GetValue(i, 0));
		}

        //Process Mode-Functioned Network
		Model.ELU(Model.Layer(Y, NNW0, NNb0, Y));
		Model.ELU(Model.Layer(Y, NNW1, NNb1, Y));
		Model.Layer(Y, NNW2, NNb2, Y);

        //Renormalise Output
		Model.Renormalise(Y, Ymean, Ystd, Y);
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
					Parameters = (Parameters)EditorGUILayout.ObjectField(Parameters, typeof(Parameters), true);
					EditorGUILayout.EndHorizontal();
				}
			}
		}
	}
	#endif
	
}