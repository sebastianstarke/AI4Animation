using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CCD : MonoBehaviour {

	public Transform Target;

	public Transform[] Bones;

	public int Iterations = 10;

	void Start() {
		
	}
	
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
}
