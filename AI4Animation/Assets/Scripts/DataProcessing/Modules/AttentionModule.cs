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
		//GetHeightMap(editor.GetCurrentFrame(), editor.Mirror).Draw();
		Frame frame = editor.GetCurrentFrame();
		for(int i=0; i<Data.Source.Bones.Length; i++) {
			Vector3 bonePosition = frame.GetBoneTransformation(i, editor.Mirror).GetPosition();
			UltiDraw.DrawSphere(bonePosition, Quaternion.identity, Size, UltiDraw.Purple.Transparent(0.05f));
			Collider[] colliders = Physics.OverlapSphere(bonePosition, Size/2f, Mask);
			List<float> weights = new List<float>();
			List<Vector3> points = new List<Vector3>();
			List<Vector3> normals = new List<Vector3>();
			for(int j=0; j<colliders.Length; j++) {
				if(!(colliders[j] is MeshCollider && !((MeshCollider)colliders[j]).convex)) {
					Vector3 point = colliders[j].ClosestPoint(bonePosition);
					Vector3 normal = Vector3.zero;

					List<RaycastHit> hits = new List<RaycastHit>();
					Quaternion rotation = colliders[j].transform.rotation;

					RaycastHit hitXP;
					if(Physics.Raycast(point + Size/2f * (rotation*Vector3.right), -(rotation*Vector3.right), out hitXP, Size, Mask)) {
						hits.Add(hitXP);
					}
					RaycastHit hitXN;
					if(Physics.Raycast(point + Size/2f * (rotation*Vector3.left), -(rotation*Vector3.left), out hitXN, Size, Mask)) {
						hits.Add(hitXN);
					}

					RaycastHit hitYP;
					if(Physics.Raycast(point + Size/2f * (rotation*Vector3.up), -(rotation*Vector3.up), out hitYP, Size, Mask)) {
						hits.Add(hitYP);
					}
					RaycastHit hitYN;
					if(Physics.Raycast(point + Size/2f * (rotation*Vector3.down), -(rotation*Vector3.down), out hitYN, Size, Mask)) {
						hits.Add(hitYN);
					}

					RaycastHit hitZP;
					if(Physics.Raycast(point + Size/2f * (rotation*Vector3.forward), -(rotation*Vector3.forward), out hitZP, Size, Mask)) {
						hits.Add(hitZP);
					}
					RaycastHit hitZN;
					if(Physics.Raycast(point + Size/2f * (rotation*Vector3.back), -(rotation*Vector3.back), out hitZN, Size, Mask)) {
						hits.Add(hitZN);
					}
					
					if(hits.Count > 0) {
						RaycastHit closest = hits[0];
						for(int k=1; k<hits.Count; k++) {
							if(Vector3.Distance(hits[k].point, point) < Vector3.Distance(closest.point, point)) {
								closest = hits[k];
							}
						}
						normal = closest.normal;
					}
					
					float weight = Vector3.Distance(bonePosition, point) / (Size/2f);
					UltiDraw.DrawSphere(point, Quaternion.identity, 0.05f, UltiDraw.Mustard);
					UltiDraw.DrawArrow(point, point + 0.1f*normal, 0.8f, 0.005f, 0.0125f, UltiDraw.Mustard);
					points.Add(point);
					normals.Add(normal);
					weights.Add(weight);
				}
			}
			if(points.Count > 0) {
				Vector3 gradient = Vector3.zero;
				for(int j=0; j<points.Count; j++) {
					Vector3 g = (points[j] - bonePosition).normalized;
					Vector3 n = -normals[j];
					float w = weights[j];
					Vector3 v = Utility.Interpolate(n, g, Mathf.Pow(w, 0.25f));
					gradient += (1f-w) * Size/2f * v;
				}
				gradient /= points.Count;
				UltiDraw.DrawArrow(bonePosition, bonePosition + gradient, 0.8f, 0.01f, 0.025f, UltiDraw.Cyan);
			}
		}
		UltiDraw.End();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		Size = EditorGUILayout.FloatField("Size", Size);
		Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
	}

}
#endif
