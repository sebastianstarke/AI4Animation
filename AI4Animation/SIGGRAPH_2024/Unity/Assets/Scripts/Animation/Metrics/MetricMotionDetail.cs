using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using AI4Animation;

public class MetricMotionDetail : MonoBehaviour {

    public string Tag = string.Empty;
    public Actor Actor;
    public int Horizon = 0;
    [NonSerialized] public KeyCode ResetKey = KeyCode.K;
    public float CaptureRate = 10f;
    public string[] Bones = new string[0];

    private List<Sample> History = null;

    private List<float> Values = new List<float>();

    public class Sample {
        public float Timestamp;
        public Matrix4x4[] World;
        public Matrix4x4[] Local;

        public Sample(Actor actor) {
            Timestamp = Time.time;
            World = actor.GetBoneTransformations();
            Local = new Matrix4x4[actor.Bones.Length];
            for(int i=0; i<actor.Bones.Length; i++) {
                Local[i] = actor.Bones[i].GetTransform().GetLocalMatrix();
            }
        }
    }

    void Start() {
        StartCoroutine(Capture());
    }

    private IEnumerator Capture() {
        History = new List<Sample>();
        while(true) {
            yield return new WaitForSeconds(1f/CaptureRate);
            Collect();
        }
    }

    private void Collect() {
        if(Input.GetKey(ResetKey)) {
            History.Clear();
            Values.Clear();
        }

        if(Horizon > 0) {
            while(History.Count >= Horizon) {
                History.RemoveAt(0);
            }
            while(Values.Count >= Horizon) {
                Values.RemoveAt(0);
            }
        }
        History.Add(new Sample(Actor));
        if(History.Count > 2) {
            float value = 0f;
            Actor.Bone[] bones = Actor.Bones;
            foreach(Actor.Bone bone in Bones.Length == 0 ? Actor.Bones : Actor.FindBones(Bones)) {
                Sample previous = History[History.Count-2];
                Sample current = History[History.Count-1];
                value += Quaternion.Angle(previous.Local[bone.GetIndex()].GetRotation(), current.Local[bone.GetIndex()].GetRotation()) / (current.Timestamp - previous.Timestamp);
            }
            Values.Add(value / bones.Length);
        }
    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(MetricMotionDetail), true)]
	public class MetricMotionDetailEditor : Editor {

        public override bool RequiresConstantRepaint() => true;

		public override void OnInspectorGUI() {
            MetricMotionDetail instance = (MetricMotionDetail)target;
			DrawDefaultInspector();
            EditorGUILayout.HelpBox("Mean:" + instance.Values.ToArray().Mean(), MessageType.None);
            EditorGUILayout.HelpBox("Std:" + instance.Values.ToArray().Sigma(), MessageType.None);
		}

	}
	#endif    
}
