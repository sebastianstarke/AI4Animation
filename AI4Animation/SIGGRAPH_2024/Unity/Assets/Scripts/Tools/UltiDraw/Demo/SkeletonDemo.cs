using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SkeletonDemo : MonoBehaviour {

	public float BoneSize = 0.025f;
	public Color BoneColor = UltiDraw.Black;
	public Color JointColor = UltiDraw.Mustard;
	public bool DrawTransforms = false;

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			Draw();
		}
	}

	void OnRenderObject() {
		Draw();
	}

	void Draw() {
		UltiDraw.Begin();

		Action<Transform, Transform> recursion = null;
		recursion = new Action<Transform, Transform>((segment, parent) => {
			if(segment == null) {
				return;
			}
			if(parent != null) {
				UltiDraw.DrawSphere(
					parent.position,
					parent.rotation,
					5f/8f * BoneSize,
					JointColor
				);
				float distance = Vector3.Distance(parent.position, segment.position);
				if(distance > 0.05f) {
					UltiDraw.DrawBone(
						parent.position,
						Quaternion.FromToRotation(parent.forward, segment.position - parent.position) * parent.rotation,
						4f*BoneSize, distance,
						BoneColor
					);
				}
			}
			parent = segment;
			for(int i=0; i<segment.childCount; i++) {
				recursion(segment.GetChild(i), parent);
			}
		});
		recursion(transform, null);
		
		if(DrawTransforms) {
			Action<Transform> f = null;
			f = new Action<Transform>((segment) => {
				UltiDraw.DrawTranslateGizmo(segment.position, segment.rotation, 0.075f);
				for(int i=0; i<segment.childCount; i++) {
					f(segment.GetChild(i));
				}
			});
			f(transform);
		}

		UltiDraw.End();
	}

}
