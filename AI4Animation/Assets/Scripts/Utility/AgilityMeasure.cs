using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgilityMeasure : MonoBehaviour {

	public float Agility;

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
		Agility = Utility.ComputeMean(Angles.ToArray());
	}
}
