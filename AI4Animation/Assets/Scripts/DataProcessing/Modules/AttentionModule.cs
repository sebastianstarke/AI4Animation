#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class AttentionModule : Module {

	public float Size = 1f;
	public LayerMask Mask = -1;

	public override TYPE Type() {
		return TYPE.Attention;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		Inspect = true;
		return this;
	}

	public override void Draw(MotionEditor editor) {
		UltiDraw.Begin();
		Frame frame = editor.GetCurrentFrame();
		for(int i=0; i<Data.Source.Bones.Length; i++) {
			float radius = Size/2f;
			Vector3 bone = frame.GetBoneTransformation(i, editor.Mirror).GetPosition();
			UltiDraw.DrawSphere(bone, Quaternion.identity, 2f*radius, UltiDraw.Purple.Transparent(0.05f));
			Collider[] colliders = Physics.OverlapSphere(bone, radius, Mask);
			List<Vector3> points = new List<Vector3>();
			List<Vector3> normals = new List<Vector3>();
			for(int j=0; j<colliders.Length; j++) {
				if(!(colliders[j] is MeshCollider && !((MeshCollider)colliders[j]).convex)) {
					Vector3 point = colliders[j].ClosestPoint(bone);
					Vector3 normal = GetNormal(bone, point, colliders[j], radius);
					points.Add(point);
					normals.Add(normal);
					UltiDraw.DrawSphere(point, Quaternion.identity, 0.025f, UltiDraw.Mustard);
					UltiDraw.DrawArrow(point, point + 0.1f*normal, 0.8f, 0.005f, 0.0125f, UltiDraw.Mustard);
				}
			}
			if(points.Count > 0) {
				Vector3 gradient = Vector3.zero;
				for(int j=0; j<points.Count; j++) {
					float w = 1f - Vector3.Distance(bone, points[j]) / radius;
					Vector3 v = Utility.Interpolate((points[j] - bone).normalized, -normals[j], Mathf.Pow(w, 5f));
					gradient += w * radius * v;
				}
				gradient /= points.Count;
				UltiDraw.DrawArrow(bone, bone + gradient, 0.8f, 0.005f, 0.015f, UltiDraw.Cyan.Transparent(0.5f));
			}
		}
		UltiDraw.End();
	}

	public Vector3 GetNormal(Vector3 bone, Vector3 point, Collider collider, float radius) {
		if(bone == point) {
			List<RaycastHit> hits = new List<RaycastHit>();
			Quaternion rotation = collider.transform.rotation;

			Vector3 x = rotation * Vector3.right;
			Vector3 y = rotation * Vector3.up;
			Vector3 z = rotation * Vector3.forward;

			RaycastHit XP;
			if(Physics.Raycast(point + radius * x, -x, out XP, 2f*radius, Mask)) {
				hits.Add(XP);
			}
			RaycastHit XN;
			if(Physics.Raycast(point + radius * -x, x, out XN, 2f*radius, Mask)) {
				hits.Add(XN);
			}
			RaycastHit YP;
			if(Physics.Raycast(point + radius * y, -y, out YP, 2f*radius, Mask)) {
				hits.Add(YP);
			}
			RaycastHit YN;
			if(Physics.Raycast(point + radius * -y, y, out YN, 2f*radius, Mask)) {
				hits.Add(YN);
			}
			RaycastHit ZP;
			if(Physics.Raycast(point + radius * z, -z, out ZP, 2f*radius, Mask)) {
				hits.Add(ZP);
			}
			RaycastHit ZN;
			if(Physics.Raycast(point + radius * -z, z, out ZN, 2f*radius, Mask)) {
				hits.Add(ZN);
			}
			
			if(hits.Count > 0) {
				RaycastHit closest = hits[0];
				for(int k=1; k<hits.Count; k++) {
					if(Vector3.Distance(hits[k].point, point) < Vector3.Distance(closest.point, point)) {
						closest = hits[k];
					}
				}
				return closest.normal;
			} else {
				Debug.Log("Could not compute normal for collider " + collider.name + ".");
				return Vector3.zero;
			}
		} else {
			RaycastHit hit;
			if(Physics.Raycast(bone, (point - bone).normalized, out hit, 2f*radius, Mask)) {
				return hit.normal;
			} else {
				Debug.Log("Could not compute normal for collider " + collider.name + ".");
				return Vector3.zero;
			}
			
		}
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Size = EditorGUILayout.FloatField("Size", Size);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

}
#endif
