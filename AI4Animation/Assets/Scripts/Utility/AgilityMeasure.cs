using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgilityMeasure : MonoBehaviour {

	[Range(0f, 1f)] public float Position = 0.9f;

	public Transform[] Bones;

	private Quaternion[] PreviousRotations;
	private List<float> Angles;

	void Start() {
		PreviousRotations = new Quaternion[Bones.Length];
		for(int i=0; i<PreviousRotations.Length; i++) {
			PreviousRotations[i] = Bones[i].localRotation;
		}
		Angles = new List<float>();
	}

	void LateUpdate () {
		for(int i=0; i<Bones.Length; i++) {
			Quaternion rotation = Bones[i].localRotation;
			Angles.Add(Quaternion.Angle(PreviousRotations[i], rotation));
			PreviousRotations[i] = rotation;
		}
	}

	void OnGUI() {
		GUI.color = UltiDraw.Mustard;
		GUI.backgroundColor = UltiDraw.Black;
		if(GUI.Button(Utility.GetGUIRect(0.005f, Position, 0.02f, 0.02f), "X")) {
			Angles.Clear();
		};
		GUI.Box(Utility.GetGUIRect(0.025f, Position, 0.175f, 0.02f), "Average Agility: " + Utility.ComputeMean(Angles.ToArray()));
	}
}
