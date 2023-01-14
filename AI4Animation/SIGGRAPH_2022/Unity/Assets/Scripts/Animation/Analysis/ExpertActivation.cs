// using System.Collections.Generic;
// using AI4Animation;
// using UnityEngine;
// #if UNITY_EDITOR
// using UnityEditor;
// #endif

// public class ExpertActivation : MonoBehaviour {

// 	public bool Draw = true;

// 	public DeepLearning.GNN Model = null;

// 	public string ID = "";

// 	public enum MODE {Function, Graph, SliceCurves, SliceBars}

// 	public MODE Mode = MODE.Function;

// 	public int Frames = 100;

// 	public UltiDraw.GUIRect Rect;
// 	public UltiDraw.GUIRect Background;
// 	public float LabelOffset = 0f;
// 	public Color Color = Color.white;

// 	public UltiDraw.Dimension Dimension = UltiDraw.Dimension.Y;

// 	private float[] Weights;
// 	private Queue<float>[] Values;

// 	private bool Setup() {
// 		if(Model == null || !Model.Setup || ID == "") {
// 			return false;
// 		}
// 		Matrix matrix = Model.GetMatrix(ID);
// 		if(matrix == null) {
// 			return false;
// 		}
// 		if(Values == null || Values.Length != matrix.GetRows()) {
// 			Values = new Queue<float>[matrix.GetRows()];
// 			for(int i=0; i<Values.Length; i++) {
// 				Values[i] = new Queue<float>();
// 				for(int j=0; j<Frames; j++) {
// 					Values[i].Enqueue(0f);
// 				}
// 			}
// 			Weights = new float[Values.Length];
// 			for(int i=0; i<Weights.Length; i++) {
// 				Weights[i] = 1f;
// 			}
// 		}
// 		return true;
// 	}

// 	void FixedUpdate() {
// 		if(!Draw) {
// 			return;
// 		}

// 		if(!Setup()) {
// 			return;
// 		}

// 		switch(Mode) {
// 			case MODE.Function:
// 			AccumulateFunction();
// 			break;
// 			case MODE.Graph:
// 			AccumulateGraph();
// 			break;
// 			case MODE.SliceCurves:
// 			AccumulateFunction();
// 			break;
// 			case MODE.SliceBars:
// 			AccumulateFunction();
// 			break;
// 		}
// 	}

// 	void OnGUI() {
// 		if(Draw) {
// 			UltiDraw.Begin();
// 			UltiDraw.OnGUILabel(Rect.GetCenter() + new Vector2(0f, 0.175f + LabelOffset), new Vector2(0f, 0.2f), 0.02f, "Expert Activation", Color.black);
// 			UltiDraw.End();
// 		}
// 	}

// 	void OnRenderObject() {
// 		if(!Application.isPlaying) {
// 			return;
// 		}

// 		if(!Draw) {
// 			return;
// 		}

// 		if(!Setup()) {
// 			return;
// 		}

// 		if(Background.GetSize() != Vector2.zero) {
// 			UltiDraw.Begin();
// 			UltiDraw.GUIRectangle(Background.GetCenter(), Background.GetSize(), Color);
// 			UltiDraw.End();
// 		}

// 		switch(Mode) {
// 			case MODE.Function:
// 			DrawFunction();
// 			break;
// 			case MODE.Graph:
// 			DrawGraph();
// 			break;
// 			case MODE.SliceCurves:
// 			DrawSlices();
// 			break;
// 			case MODE.SliceBars:
// 			DrawSlices();
// 			break;
// 		}
// 	}

// 	private void AccumulateFunction() {
// 		#if UNITY_EDITOR
// 		if(EditorApplication.isPaused) {
// 			return;
// 		}
// 		#endif
// 		for(int i=0; i<Weights.Length; i++) {
// 			Weights[i] = Model.GetMatrix(ID).GetValue(i, 0);
// 			if(float.IsNaN(Weights[i])) {
// 				Weights[i] = 0f;
// 			}
// 		}
// 		for(int i=0; i<Values.Length; i++) {
// 			Values[i].Enqueue(Weights[i]);
// 			while(Values[i].Count > Frames) {
// 				Values[i].Dequeue();
// 			}
// 		}
// 	}

// 	private void AccumulateGraph() {
// 		#if UNITY_EDITOR
// 		if(EditorApplication.isPaused) {
// 			return;
// 		}
// 		#endif
// 		for(int i=0; i<Weights.Length; i++) {
// 			Weights[i] = Model.GetMatrix(ID).GetValue(i, 0);
// 		}

// 		for(int i=0; i<Weights.Length; i++) {
// 			Weights[i] = Mathf.Abs(Weights[i]);
// 			Weights[i] = Mathf.Pow(Weights[i], Mathf.Log(Weights.Length));
// 		}
// 		float sum = Weights.Sum();
// 		for(int i=0; i<Weights.Length; i++) {
// 			Weights[i] = Weights[i] / (sum == 0f ? 1f : sum);
// 		}
		
// 		for(int i=0; i<Values.Length; i++) {
// 			Values[i].Enqueue(Weights[i]);
// 			while(Values[i].Count > Frames) {
// 				Values[i].Dequeue();
// 			}
// 		}
// 	}

// 	private void DrawSlices() {
// 		UltiDraw.Begin();
// 		Color[] colors = UltiDraw.GetRainbowColors(Values.Length);
// 		for(int i=0; i<Weights.Length; i++) {
// 			List<float> function = new List<float>();
// 			for(int j=0; j<Frames; j++) {
// 				function.Add(Values[i].ToArray()[j]);
// 			}
// 			Vector3 center = Rect.GetCenter() + new Vector2(0f, 0.05f*i);
// 			Vector2 size = new Vector2(Rect.GetSize().x, 0.05f);
// 			if(Mode == MODE.SliceCurves) {
// 				UltiDraw.PlotFunction(center, size, function.ToArray(), yMin: 0f, yMax: 1f, thickness: 0.001f);
// 			}
// 			if(Mode == MODE.SliceBars) {
// 				UltiDraw.PlotBars(center, size, function.ToArray(), yMin: 0f, yMax: 1f, thickness: 0.001f);
// 			}
// 		}
// 		UltiDraw.End();
// 	}

// 	private void DrawFunction() {
// 		UltiDraw.Begin();
// 		Color[] colors = UltiDraw.GetRainbowColors(Values.Length);
// 		List<float[]> functions = new List<float[]>();
// 		for(int i=0; i<Values.Length; i++) {
// 			functions.Add(Values[i].ToArray());
// 		}
// 		UltiDraw.PlotFunctions(Rect.GetCenter(), Rect.GetSize(), functions.ToArray(), Dimension, yMin: 0f, yMax: 1f, thickness: 0.001f);
// 		UltiDraw.End();
// 	}

// 	private void DrawGraph() {
// 		UltiDraw.Begin();
// 		Color[] colors = UltiDraw.GetRainbowColors(Values.Length);
// 		Vector2 pivot = Rect.GetCenter();
// 		float radius = 0.2f * Rect.W;
// 		UltiDraw.GUICircle(pivot, Rect.W*1.05f, UltiDraw.Gold);
// 		UltiDraw.GUICircle(pivot, Rect.W, UltiDraw.White);
// 		Vector2[] anchors = new Vector2[Values.Length];
// 		for(int i=0; i<Values.Length; i++) {
// 			float step = (float)i / (float)Values.Length;
// 			anchors[i] = Rect.ToScreen(new Vector2(Mathf.Cos(step*2f*Mathf.PI), Mathf.Sin(step*2f*Mathf.PI)));
// 		}
// 		Vector2[] positions = new Vector2[Frames];
// 		for(int i=0; i<Values.Length; i++) {
// 			int _index = 0;
// 			foreach(float value in Values[i]) {
// 				positions[_index] += value * anchors[i];
// 				_index += 1;
// 			}
// 		}
// 		for(int i=1; i<positions.Length; i++) {
// 			UltiDraw.GUILine(pivot + positions[i-1], pivot + positions[i], 0.1f*radius, UltiDraw.Black.Opacity((float)(i+1)/(float)positions.Length));
// 		}
// 		for(int i=0; i<anchors.Length; i++) {
// 			UltiDraw.GUILine(pivot + positions.Last(), pivot + anchors[i], 0.1f*radius, colors[i].Opacity(Weights[i]));
// 			UltiDraw.GUICircle(pivot + anchors[i], Mathf.Max(0.5f*radius, Weights[i].Normalize(0f, 1f, 0.5f, 1f) * radius), Color.Lerp(UltiDraw.Black, colors[i], Weights[i]));
// 		}
// 		UltiDraw.GUICircle(pivot + positions.Last(), 0.5f*radius, UltiDraw.Purple);
// 		UltiDraw.End();
// 	}
// }