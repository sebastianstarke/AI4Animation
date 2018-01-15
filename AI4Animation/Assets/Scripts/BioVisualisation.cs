using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(BioAnimation_APFNN))]
public class BioVisualisation : MonoBehaviour {

	private BioAnimation_APFNN Animation;

	private int Frames = 60;
	private Queue<float>[] CP;

	void Awake() {
		Animation = GetComponent<BioAnimation_APFNN>();
		CP = new Queue<float>[4];
		for(int i=0; i<4; i++) {
			CP[i] = new Queue<float>();
			for(int j=0; j<Frames; j++) {
				CP[i].Enqueue(0f);
			}
		}
	}

	private void UpdateData() {
		for(int i=0; i<CP.Length; i++) {
			CP[i].Dequeue();
			CP[i].Enqueue(Animation.APFNN.GetControlPoint(i));
		}
	}

	private void DrawControlPoint(float x, float y, float width, float height, int index, Color color) {
		UnityGL.DrawGUIQuad(x, y-height, width, height, Utility.Black.Transparent(0.75f));
		int _index;
		_index = 0;
		float _x = 0f;
		float _xPrev = 0f;
		float _y = 0f;
		float _yPrev = 0f;
		foreach(float value in CP[index]) {
			_x = x + (float)(_index)/(float)(Frames-1) * width;
			_y = y - height + value*height;
			if(_index > 0) {
				UnityGL.DrawGUILine(
					_xPrev,
					_yPrev, 
					_x,
					_y,
					color
				);
			}
			_xPrev = _x; 
			_yPrev = _y;
			_index += 1;
		}
	}

	void OnRenderObject() {
		UpdateData();

		UnityGL.Start();
		DrawControlPoint(0f, 1f, 0.25f, 0.1f, 0, Utility.Red);
		DrawControlPoint(0f, 0.9f, 0.25f, 0.1f, 1, Utility.Green);
		DrawControlPoint(0f, 0.8f, 0.25f, 0.1f, 2, Utility.Cyan);
		DrawControlPoint(0f, 0.7f, 0.25f, 0.1f, 3, Utility.Orange);

		UnityGL.DrawGUILine(0f, 0.9f, 0.25f, 0.9f, Utility.White);
		UnityGL.DrawGUILine(0f, 0.8f, 0.25f, 0.8f, Utility.White);
		UnityGL.DrawGUILine(0f, 0.7f, 0.25f, 0.7f, Utility.White);
		UnityGL.Finish();
	}

}
