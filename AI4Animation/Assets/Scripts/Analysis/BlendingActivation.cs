using System.Collections;
using System.Collections.Generic;
using DeepLearning;
using UnityEngine;

public class BlendingActivation : MonoBehaviour {

	public int Frames = 150;

    [Range(0f, 1f)] public float Y = 0.1f;

	private Model Model;
	private Queue<float>[] Values;

	void Awake() {
		Model = GetComponent<BioAnimation>().NN.Model;
	}

	void Start() {
		Values = new Queue<float>[Model.GetTensor("BY").GetRows()];
		for(int i=0; i<Values.Length; i++) {
			Values[i] = new Queue<float>();
			for(int j=0; j<Frames; j++) {
				Values[i].Enqueue(0f);
			}
		}
	}

	void OnRenderObject() {
		for(int i=0; i<Values.Length; i++) {
			Values[i].Dequeue();
			Values[i].Enqueue(Model.GetTensor("BY").GetValue(i, 0));
		}

		UltiDraw.Begin();
		Vector2 center = new Vector2(0.5f, Y);
		float width = 0.95f;
		float height = 0.1f;
		float border = 0.0025f;
		UltiDraw.DrawGUIRectangle(
			center,
			new Vector2(width+2f*border/Screen.width*Screen.height, height+2f*border),
			UltiDraw.Black.Transparent(0.5f));
		UltiDraw.DrawGUIRectangle(
			center,
			new Vector2(width, height),
			UltiDraw.White.Transparent(0.5f));

		Color[] colors = UltiDraw.GetRainbowColors(Values.Length);
		for(int i=0; i<colors.Length; i++) {
			DrawControlPoint(center.x - width/2f, center.y + height/2f, width, height, Values[i], colors[i]);
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
