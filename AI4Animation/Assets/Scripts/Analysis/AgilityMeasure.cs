using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgilityMeasure : MonoBehaviour {

	[Range(0f, 1f)] public float Position = 0.9f;

	public Transform[] Bones;

	private Quaternion[] PreviousRotations;
	private List<float> Values;

	[ContextMenu("Setup")]
	public void Setup() {
		Bones = new Transform[0];
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("LeftUpLeg"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("LeftLeg"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("LeftFoot"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("RightUpLeg"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("RightLeg"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("RightFoot"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("LeftShoulder"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("LeftArm"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("LeftForeArm"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("LeftHand"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("RightShoulder"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("RightArm"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("RightForeArm"));
		ArrayExtensions.Add(ref Bones, GetComponent<Actor>().FindTransform("RightHand"));
	}

	void Start() {
		PreviousRotations = new Quaternion[Bones.Length];
		for(int i=0; i<PreviousRotations.Length; i++) {
			PreviousRotations[i] = Bones[i].localRotation;
		}
		Values = new List<float>();
	}

	void LateUpdate () {
		float value = 0f;
		for(int i=0; i<Bones.Length; i++) {
			Quaternion rotation = Bones[i].localRotation;
			value += Quaternion.Angle(PreviousRotations[i], rotation);
			PreviousRotations[i] = rotation;
		}
		value /= Bones.Length;
		//value /= Time.deltaTime;
		Values.Add(value);
	}

	void OnGUI() {
		GUI.color = UltiDraw.Mustard;
		GUI.backgroundColor = UltiDraw.Black;
		if(GUI.Button(Utility.GetGUIRect(0.005f, Position, 0.05f, 0.05f), "X")) {
			Values.Clear();
		};
		GUI.Box(Utility.GetGUIRect(0.055f, Position, 0.25f, 0.05f), "Average Agility: " + Utility.ComputeMean(Values.ToArray()));
	}
}
