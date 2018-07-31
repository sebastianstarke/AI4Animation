using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class KeypointField {

	public List<Vector3> Keypoints = new List<Vector3>();
	public Vector3[] Positions = new Vector3[0];
	public Vector3[] Gradients = new Vector3[0];

	private Actor Actor;
	private float Radius;
	private LayerMask Mask;

	public KeypointField(Actor actor, float radius, LayerMask mask) {
		Actor = actor;
		Radius = radius;
		Mask = mask;
	}

	public void Sense() {
		Keypoints.Clear();
		Positions = new Vector3[Actor.Bones.Length];
		Gradients = new Vector3[Actor.Bones.Length];
		for(int i=0; i<Actor.Bones.Length; i++) {
			Positions[i] = Actor.Bones[i].Transform.position;
			Collider[] colliders = Physics.OverlapSphere(Positions[i], Radius, Mask);
			List<Vector3> points = new List<Vector3>();
			List<Vector3> centers = new List<Vector3>();
			for(int j=0; j<colliders.Length; j++) {
				if(!(colliders[j] is MeshCollider && !((MeshCollider)colliders[j]).convex)) {
					points.Add(colliders[j].ClosestPoint(Positions[i]));
					centers.Add(colliders[j].bounds.center);
				}
			}
			Vector3 gradient = Vector3.zero;
			for(int j=0; j<points.Count; j++) {
				float w = 1f - Vector3.Distance(Positions[i], points[j]) / Radius;
				Vector3 v = Utility.Interpolate((points[j] - Positions[i]).normalized, (centers[j] - Positions[i]).normalized, w * w);
				gradient += w * Radius * v;
			}
			gradient = Vector3.ClampMagnitude(gradient, Radius);
			Gradients[i] = gradient;
			Keypoints.AddRange(points);
		}
	}

	public void Draw() {
		UltiDraw.Begin();
		for(int i=0; i<Keypoints.Count; i++) {
			UltiDraw.DrawSphere(Keypoints[i], Quaternion.identity, 0.025f, UltiDraw.Red);
		}
		for(int i=0; i<Actor.Bones.Length; i++) {
			UltiDraw.DrawWireSphere(Actor.Bones[i].Transform.position, Actor.Bones[i].Transform.rotation, 2f*Radius, UltiDraw.Black.Transparent(0.05f));
			UltiDraw.DrawArrow(Positions[i], Positions[i] + Gradients[i], 0.8f, 0.005f, 0.015f, UltiDraw.Cyan.Transparent(0.5f));
		}
		UltiDraw.End();
	}

}
