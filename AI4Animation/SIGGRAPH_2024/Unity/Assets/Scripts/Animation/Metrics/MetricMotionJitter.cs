using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using AI4Animation;

public class MetricMotionJitter : MonoBehaviour {

    public Actor Actor;
    public int Horizon = 0;
    [NonSerialized] public KeyCode ResetKey = KeyCode.K;
    public float CaptureRate = 10f;

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
            for(int i=0; i<Actor.Bones.Length; i++) {
                Sample a = History[History.Count-3];
                Sample b = History[History.Count-2];
                Sample c = History[History.Count-1];
                Vector3 currentPosition = c.World[i].GetPosition();
                Vector3 previousPosition = b.World[i].GetPosition();
                float dt = b.Timestamp - a.Timestamp;
                Vector3 previousVelocity = (b.World[i].GetPosition() - a.World[i].GetPosition()) / dt;
                value += Vector3.Distance(previousPosition + previousVelocity * dt, currentPosition) * dt;
            }
            Values.Add(value / Actor.Bones.Length);
        }
    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(MetricMotionJitter), true)]
	public class MetricMotionJitterEditor : Editor {

        public override bool RequiresConstantRepaint() => true;

		public override void OnInspectorGUI() {
            MetricMotionJitter instance = (MetricMotionJitter)target;
			DrawDefaultInspector();
            EditorGUILayout.HelpBox("Jitter Mean:" + instance.Values.ToArray().Mean(), MessageType.None);
            EditorGUILayout.HelpBox("Jitter Std:" + instance.Values.ToArray().Sigma(), MessageType.None);
		}

	}
	#endif    
}
