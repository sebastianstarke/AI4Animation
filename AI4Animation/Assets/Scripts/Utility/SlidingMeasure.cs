using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SlidingMeasure : MonoBehaviour {

	[Range(0f,1f)] public float Position = 0.9f;

	public Transform[] Bones;

	private Vector3[] PreviousPositions;
	private List<float> Slidings;

	void Start() {
		PreviousPositions = new Vector3[Bones.Length];
		for(int i=0; i<PreviousPositions.Length; i++) {
			PreviousPositions[i] = Bones[i].position;
		}
		Slidings = new List<float>();
	}

	void LateUpdate () {
		float heightThreshold = 0.01f;
		for(int i=0; i<Bones.Length; i++) {
			float height = Mathf.Max(0f, Bones[i].position.y);
			if(height < heightThreshold) {
				Vector3 oldPosition = PreviousPositions[i];
				oldPosition.y = 0f;
				Vector3 newPosition = Bones[i].position;
				newPosition.y = 0f;
				float weight = 1f - Mathf.Abs(height) / heightThreshold;
				Slidings.Add(weight * Vector3.Distance(oldPosition, newPosition) * 60f);
			} else {
				Slidings.Add(0f);
			}
			PreviousPositions[i] = Bones[i].position;
			
			/*
			float heightThreshold = i==0 || i==1 ? 0.025f : 0.05f;
			float velocityThreshold = i==0 || i==1 ? 0.015f : 0.015f;
			Vector3 oldPosition = PreviousPositions[i];
			Vector3 newPosition = Bones[i].position;
			float velocityWeight = Utility.Exponential01((newPosition-oldPosition).magnitude / velocityThreshold);
			float heightWeight = Utility.Exponential01(newPosition.y / heightThreshold);
			float weight = 1f - Mathf.Min(velocityWeight, heightWeight);
			Vector3 slide = newPosition - oldPosition;
			slide.y = 0f;
			Slidings.Add(weight * slide.magnitude * 60f);
			PreviousPositions[i] = newPosition;
			*/
		}
	}

	void OnGUI() {
		GUI.color = UltiDraw.Mustard;
		GUI.backgroundColor = UltiDraw.Black;
		if(GUI.Button(Utility.GetGUIRect(0.005f, Position, 0.02f, 0.02f), "X")) {
			Slidings.Clear();
		};
		GUI.Box(Utility.GetGUIRect(0.025f, Position, 0.175f, 0.02f), "Average Sliding: " + Utility.ComputeMean(Slidings.ToArray()));
	}
}
