using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using AI4Animation;

public class MetricTrackingAccuracy : MonoBehaviour {

    public enum SPACE {World, Local}

    public string Tag = string.Empty;
    public Actor Actor;
    public Actor Reference;
    public SPACE Space = SPACE.World;
    public int Horizon = 0;
    [NonSerialized] public KeyCode ResetKey = KeyCode.K;
    public float CaptureRate = 10f;
    public string[] Bones = new string[0];

    private List<float> PositionValues = new List<float>();
    private List<float> RotationValues = new List<float>();

    void Start() {
        StartCoroutine(Capture());
    }

    private IEnumerator Capture() {
        while(true) {
            yield return new WaitForSeconds(1f/CaptureRate);
            Collect();
        }
    }

    private void Collect() {
        if(Input.GetKey(ResetKey)) {
            PositionValues.Clear();
            RotationValues.Clear();
        }

        if(Horizon > 0) {
            while(PositionValues.Count >= Horizon) {
                PositionValues.RemoveAt(0);
                RotationValues.RemoveAt(0);
            }
        }

        if(Reference == null) {
            Debug.Log("Reference not defined.");
            return;
        }
        float positionError = 0f;
        float rotationError = 0f;
        int count = 0;
        foreach(string bone in Bones) {
            Actor.Bone self = Actor.FindBone(bone);
            Actor.Bone target = Reference.FindBone(bone);
            if(self == null || target == null) {
                Debug.Log("Bone " + bone + " could not be mapped.");
                return;
            } else {
                if(Space == SPACE.World) {
                    positionError += Vector3.Distance(self.GetPosition(), target.GetPosition());
                    rotationError += Quaternion.Angle(self.GetRotation(), target.GetRotation());
                }
                if(Space == SPACE.Local) {
                    positionError += Vector3.Distance(self.GetPosition().PositionTo(Actor.GetRoot().GetWorldMatrix()), target.GetPosition().PositionTo(Reference.GetRoot().GetWorldMatrix()));
                    rotationError += Quaternion.Angle(self.GetRotation().RotationTo(Actor.GetRoot().GetWorldMatrix()), target.GetRotation().RotationTo(Reference.GetRoot().GetWorldMatrix()));
                }
                count += 1;
            }
        }
        PositionValues.Add(count == 0 ? 0f : (positionError / count));
        RotationValues.Add(count == 0 ? 0f : (rotationError / count));
    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(MetricTrackingAccuracy), true)]
	public class MetricTrackingAccuracyEditor : Editor {

        public override bool RequiresConstantRepaint() => true;

		public override void OnInspectorGUI() {
            MetricTrackingAccuracy instance = (MetricTrackingAccuracy)target;
			DrawDefaultInspector();
            EditorGUILayout.HelpBox("Position Mean:" + instance.PositionValues.ToArray().Mean(), MessageType.None);
            EditorGUILayout.HelpBox("Position Std:" + instance.PositionValues.ToArray().Sigma(), MessageType.None);
            EditorGUILayout.HelpBox("Rotation Mean:" + instance.RotationValues.ToArray().Mean(), MessageType.None);
            EditorGUILayout.HelpBox("Rotation Std:" + instance.RotationValues.ToArray().Sigma(), MessageType.None);
		}

	}
	#endif    
}
