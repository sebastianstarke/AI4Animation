using System.Collections;
using System.Collections.Generic;
using DeepLearning;
using UnityEngine;

public class ExpertActivation : MonoBehaviour {

	public string ID = "";

	public enum MODE {Function, Bars, Graph}

	public MODE Mode = MODE.Function;

	public int Frames = 100;

	[Range(0f, 1f)] public float X = 0.5f;
    [Range(0f, 1f)] public float Y = 0.1f;
	[Range(0f, 1f)] public float W = 0.75f;
	[Range(0f, 1f)] public float H = 0.1f;

	private NeuralNetwork NN;
	private Queue<float>[] Values;

	void Awake() {
		NN = GetComponent<NeuralNetwork>();
	}

	void Start() {
		Setup();
	}

	void OnEnable() {
		Awake();
		Start();
	}

	private bool Setup() {
		Tensor tensor = NN.GetTensor(ID);
		if(tensor == null) {
			return false;
		}
		if(Values == null || Values.Length != tensor.GetRows()) {
			Values = new Queue<float>[tensor.GetRows()];
			for(int i=0; i<Values.Length; i++) {
				Values[i] = new Queue<float>();
				for(int j=0; j<Frames; j++) {
					Values[i].Enqueue(0f);
				}
			}
		}
		return true;
	}

	void OnRenderObject() {
		if(!Application.isPlaying) {
			return;
		}

		if(!Setup()) {
			return;
		}

		UltiDraw.Begin();

		float[] values = new float[Values.Length];
		for(int i=0; i<Values.Length; i++) {
			values[i] = NN.GetTensor(ID).GetValue(i, 0);
		}

		for(int i=0; i<values.Length; i++) {
			values[i] = Mathf.Pow(values[i], 2f);
		}
		float sum = values.Sum();
		for(int i=0; i<values.Length; i++) {
			values[i] = values[i] / sum;
		}

		for(int i=0; i<Values.Length; i++) {
			Values[i].Dequeue();
			Values[i].Enqueue(values[i]);
		}

		switch(Mode) {
			case MODE.Function:
			Vector2 center = new Vector2(X, Y);
			float border = 0.0025f;
			UltiDraw.DrawGUIRectangle(
				center,
				new Vector2(W+2f*border/Screen.width*Screen.height, H+2f*border),
				UltiDraw.Black);
			UltiDraw.DrawGUIRectangle(
				center,
				new Vector2(W, H),
				UltiDraw.White);

			Color[] colors = UltiDraw.GetRainbowColors(Values.Length);
			for(int i=0; i<colors.Length; i++) {
				DrawControlPoint(center.x - W/2f, center.y + H/2f, W, H, Values[i], colors[i]);
			}
			//for(int i=0; i<colors.Length; i++) {
			//	Vector2 start = center - new Vector2(width/2f, -height/2f);
			//	UltiDraw.DrawGUIRectangle(start + (float)i/(float)(colors.Length-1)*new Vector2(width, 0f), new Vector2(0.025f, 0.025f), colors[i]);
			//}
			break;
			case MODE.Bars:

			break;
			case MODE.Graph:
			Vector2 pivot = new Vector2(X, Y);
			float radius = 0.2f * W;
			UltiDraw.DrawGUICircle(pivot, W*1.05f, UltiDraw.Gold);
			UltiDraw.DrawGUICircle(pivot, W, UltiDraw.White);
			Vector2[] anchors = new Vector2[Values.Length];
			for(int i=0; i<Values.Length; i++) {
				float step = (float)i / (float)Values.Length;
				anchors[i] = new Vector2((W - radius/2f)*Screen.height/Screen.width*Mathf.Cos(step*2f*Mathf.PI), (W - radius/2f)*Mathf.Sin(step*2f*Mathf.PI));
				UltiDraw.DrawGUICircle(pivot + anchors[i], Mathf.Max(0.5f * radius, Utility.Normalise(values[i], 0f, 1f, 0.5f, 1f) * radius), UltiDraw.Black);
			}
			Vector2[] positions = new Vector2[Frames];
			for(int i=0; i<Values.Length; i++) {
				int _index = 0;
				foreach(float value in Values[i]) {
					positions[_index] += value * anchors[i];
					_index += 1;
				}
			}
			for(int i=1; i<positions.Length; i++) {
				UltiDraw.DrawGUILine(pivot + positions[i-1], pivot + positions[i], 0.1f * radius, UltiDraw.Black.Transparent((float)(i+1)/(float)positions.Length));
			}
			for(int i=0; i<Values.Length; i++) {
				UltiDraw.DrawGUILine(pivot + positions[positions.Length-1], pivot + anchors[i], 0.1f*radius, UltiDraw.Black.Transparent(values[i]));
			}
			UltiDraw.DrawGUICircle(pivot + positions[positions.Length-1], 0.5f * radius, UltiDraw.Purple);
			break;
		}

		UltiDraw.End();
	}

	private void DrawControlPoint(float x, float y, float width, float height, Queue<float> values, Color color) {
		int _index = 0;
		float _x = 0f;
		float _xPrev = 0f;
		float _y = 0f;
		float _yPrev = 0f;
		foreach(float value in values) {
			_x = x + (float)(_index)/(float)(Frames-1) * width;
			_y = y - height + value*height;
			if(_index > 0) {
				UltiDraw.DrawGUILine(
					new Vector2(_xPrev,	_yPrev),
					new Vector2(_x, _y),
					0.002f,
					color
				);
			}
			_xPrev = _x; 
			_yPrev = _y;
			_index += 1;
		}
	}
}
