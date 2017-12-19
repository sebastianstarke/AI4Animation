using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
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

	private Vector<float> MLPXmean;
	private Vector<float> MLPXstd;
	private Vector<float> MLPYmean;
	private Vector<float> MLPYstd;

	private Matrix<float> MLPW0, MLPW1, MLPW2;
	private Vector<float> MLPb0, MLPb1, MLPb2;

	private Vector<float> PFNNXmean;
	private Vector<float> PFNNXstd;
	private Vector<float> PFNNYmean;
	private Vector<float> PFNNYstd;

	private Vector<float> MLPX;
	private Vector<float> MLPY;

	private Vector<float> PFNNX;
	private Vector<float> PFNNY;

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
		Parameters.StoreVector("File1", 0);
		Parameters.StoreVector("File1", 0);
		Parameters.StoreVector("File1", 0);
		Parameters.StoreVector("File1", 0);
		Parameters.StoreVector("File1", 0);
		Parameters.StoreVector("File1", 0);
	}

	public void SetMLPInput(int index, float value) {
		MLPX[index] = value; 
	}

	public float GetMLPOutput(int index) {
		return MLPY[index];
	}

	public void SetPFNNInput(int index, float value) {
		PFNNX[index] = value;
	}

	public float GetPFNNOutput(int index) {
		return PFNNY[index];
	}

	public void Predict() {
		//Process MLP
		Vector<float> _MLPX = (MLPX - MLPXmean).PointwiseDivide(MLPXstd);
		Vector<float> H0 = (MLPW0 * _MLPX) + MLPb0; ELU(ref H0);
		Vector<float> H1 = (MLPW1 * H0) + MLPb1; ELU(ref H1);
		MLPY = (MLPW2 * H1) + MLPb2; SoftMax(ref MLPY);
		MLPY = (MLPY.PointwiseMultiply(MLPYstd)) + MLPYmean;

		//Control Points
		//TODO

		//Process PFNN
		//TODO
	}

	private void ELU(ref Vector<float> m) {
		for(int x=0; x<m.Count; x++) {
			m[x] = System.Math.Max(m[x], 0f) + (float)System.Math.Exp(System.Math.Min(m[x], 0f)) - 1f;
		}
	}

	private void SoftMax(ref Vector<float> m) {
		float lower = 0f;
		for(int x=0; x<m.Count; x++) {
			lower += (float)System.Math.Exp(m[x]);
		}
		for(int x=0; x<m.Count; x++) {
			m[x] = (float)System.Math.Exp(m[x]) / lower;
		}
	}

	private void Linear(ref Vector<float> o, ref Vector<float> y0, ref Vector<float> y1, float mu) {
		o = (1.0f-mu) * y0 + (mu) * y1;
	}

	private void Cubic(ref Vector<float> o, ref Vector<float> y0, ref Vector<float> y1, ref Vector<float> y2, ref Vector<float> y3, float mu) {
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