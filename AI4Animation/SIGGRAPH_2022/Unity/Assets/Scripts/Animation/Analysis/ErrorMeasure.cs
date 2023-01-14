using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ErrorMeasure : MonoBehaviour {

	public enum TYPE {Distance, Angle};

	public TYPE Type = TYPE.Distance;
	public int Frames = 100;
	public UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.15f, 0.15f, 0.25f, 0.25f);
	public AnimationController[] Animations = new AnimationController[0];
	public bool AutoYMax = true;
	public float YMax = 1f;

	private List<List<float>> Values;
	// private float Amplitude = 0f;

	void Start() {
		Values = new List<List<float>>();
		for(int i=0; i<Animations.Length; i++) {
			Values.Add(new List<float>());
		}
	}

	void LateUpdate () {
		for(int i=0; i<Animations.Length; i++) {
			Values[i].Add(GetError(Animations[i]));
			while(Values[i].Count > Frames) {
				Values[i].RemoveAt(0);
			}
		}
	}

	void OnRenderObject() {
		// List<float[]> values = new List<float[]>();
		// float max = 0f;
		// for(int i=0; i<Values.Count; i++) {
		// 	float[] v = Values[i].ToArray();
		// 	max = Mathf.Max(max, v.Max());
		// 	values.Add(v);;
		// }
		// float weight = Mathf.Min(Time.deltaTime, 1f);
		// Amplitude = (1f-weight)*Amplitude + weight*max;
		// UltiDraw.Begin();
		// UltiDraw.DrawGUIFunctions(Rect.GetCenter(), Rect.GetSize(), values, 0f, AutoYMax ? Amplitude : YMax, 0.0025f, UltiDraw.DarkGrey, UltiDraw.GetRainbowColors(values.Count));
		// UltiDraw.End();
	}

    private float GetError(AnimationController animation) {
        // TimeSeries timeseries = ((SIGGRAPH_Asia_2019)animation).GetTimeSeries();
        // Matrix4x4 root = ((TimeSeries.Root)timeseries.GetSeries("Root")).Transformations[timeseries.Pivot];
        // root[1,3] = 0f;
        // Matrix4x4 goal = ((TimeSeries.Goal)timeseries.GetSeries("Goal")).Transformations[timeseries.Pivot];
        // goal[1,3] = 0f;
        // switch(Type) {
        //     case TYPE.Distance:
        //     return Vector3.Distance(root.GetPosition(), goal.GetPosition());
        //     case TYPE.Angle:
        //     return Quaternion.Angle(root.GetRotation(), goal.GetRotation());
        // }
        return 0f;
    }

}
