using System.Collections;
using System.Collections.Generic;
using DeepLearning;
using UnityEngine;

public class BlendingActivation : MonoBehaviour {

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
		Values = new Queue<float>[NN.GetTensor("BY").GetRows()];
		for(int i=0; i<Values.Length; i++) {
			Values[i] = new Queue<float>();
			for(int j=0; j<Frames; j++) {
				Values[i].Enqueue(0f);
			}
		}
	}

	void OnEnable() {
		Awake();
		Start();
	}

	void OnRenderObject() {
		for(int i=0; i<Values.Length; i++) {
			Values[i].Dequeue();
			Values[i].Enqueue(NN.GetTensor("BY").GetValue(i, 0));
		}

		UltiDraw.Begin();
		Vector2 center = new Vector2(X, Y);
		float border = 0.0025f;
		UltiDraw.DrawGUIRectangle(
			center,
			new Vector2(W+2f*border/Screen.width*Screen.height, H+2f*border),
			UltiDraw.Black.Transparent(0.5f));
		UltiDraw.DrawGUIRectangle(
			center,
			new Vector2(W, H),
			UltiDraw.White.Transparent(0.5f));

		Color[] colors = UltiDraw.GetRainbowColors(Values.Length);
		for(int i=0; i<colors.Length; i++) {
			DrawControlPoint(center.x - W/2f, center.y + H/2f, W, H, Values[i], colors[i]);
		}
		//for(int i=0; i<colors.Length; i++) {
		//	Vector2 start = center - new Vector2(width/2f, -height/2f);
		//	UltiDraw.DrawGUIRectangle(start + (float)i/(float)(colors.Length-1)*new Vector2(width, 0f), new Vector2(0.025f, 0.025f), colors[i]);
		//}
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
