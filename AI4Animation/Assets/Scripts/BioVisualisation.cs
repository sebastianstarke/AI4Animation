using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(BioAnimation_APFNN))]
public class BioVisualisation : MonoBehaviour {

	private BioAnimation_APFNN Animation;

	private int Frames = 120;
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

	private void DrawControlPoint(float x, float y, float width, float height, Queue<float> cp, Color color) {
		int _index = 0;
		float _x = 0f;
		float _xPrev = 0f;
		float _y = 0f;
		float _yPrev = 0f;
		foreach(float value in cp) {
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
		float x = 0f;
		float y = 1f;
		float width = 0.25f;
		float height = 0.1f;
		UnityGL.DrawGUIQuad(x, y-height, width, height, Utility.Black.Transparent(0.75f));
		DrawControlPoint(x, y, width, height, CP[0], Utility.Red);
		DrawControlPoint(x, y, width, height, CP[1], Utility.Green);
		DrawControlPoint(x, y, width, height, CP[2], Utility.Cyan);
		DrawControlPoint(x, y, width, height, CP[3], Utility.Orange);
		UnityGL.Finish();
	}

}
