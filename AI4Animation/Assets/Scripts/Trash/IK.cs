using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class IK : MonoBehaviour {

	public Transform[] Targets;
	public Transform[] EndEffectors;

	[Range(0f,1f)] public float RootUpdate = 1f;

	public int Iterations = 10;

	void Start() {

	}
	
	/*
	void LateUpdate() {
		for(int k=0; k<Iterations; k++) {
			for(int i=0; i<Bones.Length-1; i++) {

				Bones[i].rotation = Quaternion.Slerp(
					Bones[i].rotation,
					Quaternion.Inverse(Bones[Bones.Length-1].rotation) * Target.rotation * Bones[i].rotation,
					(float)(i+1)/(float)Bones.Length
				);

				Bones[i].rotation = Quaternion.Slerp(
					Bones[i].rotation,
					Quaternion.FromToRotation(Bones[Bones.Length-1].position - Bones[i].position, Target.position - Bones[i].position) * Bones[i].rotation,
					(float)(i+1)/(float)Bones.Length
				);
				
			}
			Bones[Bones.Length-1].rotation = Target.rotation;
		}
	}
	*/

	void LateUpdate() {
		for(int k=0; k<Iterations; k++) {

			Optimise(transform);
			
		}
	}

	private void Optimise(Transform bone) {
		bool proceed = false;
		for(int i=0; i<EndEffectors.Length; i++) {
			proceed = proceed || IsChild(bone, EndEffectors[i]);
		}
		if(proceed) {
			List<Quaternion> rotations = new List<Quaternion>();

			
			for(int i=0; i<EndEffectors.Length; i++) {
				if(IsChild(bone, EndEffectors[i])) {
					rotations.Add(
						Quaternion.Slerp(
							bone.rotation,
							Quaternion.Inverse(EndEffectors[i].rotation) * Targets[i].rotation * bone.rotation,
							0.5f
						)
					);
				}
			}
			bone.rotation = Utility.AverageQuaternions(rotations.ToArray());
			

			for(int i=0; i<EndEffectors.Length; i++) {
				if(IsChild(bone, EndEffectors[i])) {
					rotations.Add(
						Quaternion.Slerp(
							bone.rotation,
							Quaternion.FromToRotation(EndEffectors[i].position - bone.position, Targets[i].position - bone.position) * bone.rotation,
							0.5f
						)
					);
				}
			}
			bone.rotation = Utility.AverageQuaternions(rotations.ToArray());

			for(int i=0; i<bone.childCount; i++) {
				Optimise(bone.GetChild(i));
			}
		}

		
		for(int i=0; i<EndEffectors.Length; i++) {
			EndEffectors[i].rotation = Targets[i].rotation;
		}
		

		Vector3 direction = Vector3.zero;
		for(int i=0; i<EndEffectors.Length; i++) {
			direction += Targets[i].position - EndEffectors[i].position;
		}
		direction /= EndEffectors.Length;
		transform.position = Vector3.Lerp(transform.position, transform.position + direction, RootUpdate);
	}

	private bool IsChild(Transform bone, Transform child) {
		if(bone == child) {
			return false;
		}
		while(child.parent != bone) {
			if(child.parent == null) {
				return false;
			}
			child = child.parent;
		}
		return true;
	}
}
