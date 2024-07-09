using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using AI4Animation;

public class MetricRootAccuracy : MonoBehaviour {
    public string Tag = string.Empty;
    public Transform Root;
    public Transform Target;
    public int Horizon = 0;
    public float CaptureRate = 10f;
    [NonSerialized] public KeyCode ResetKey = KeyCode.K;

    public bool DrawRoot = true;
    public bool DrawTarget = true;
    public bool DrawErrors = false;
    public float PointThickness = 0.05f;
    public float LineThickness = 0.025f;
    [Range(0f,1f)] public float DirectionLength = 0.5f;
    public bool DepthRendering = false;
    public bool DrawLines = true;
    public bool DrawPositions = true;
    public bool DrawDirections = true;
    public Color RootColor = UltiDraw.Green;
    public Color TargetColor = UltiDraw.Red;
    public Color ErrorColor = UltiDraw.Magenta;
    public float ErrorSize = 0.05f;

    private List<Sample> History = null;
    private List<float> PositionError = new List<float>();
    private List<float> RotationError = new List<float>();

    public class Sample {
        public float Timestamp;
        public Matrix4x4 Root;
        public Matrix4x4 Target;

        public Sample(Transform root, Transform target) {
            Timestamp = Time.time;
            Root = root.GetWorldMatrix();
            Target = target.GetWorldMatrix();
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
            PositionError.Clear();
            RotationError.Clear();
        }

        if(Horizon > 0) {
            while(History.Count >= Horizon) {
                History.RemoveAt(0);
            }
            while(PositionError.Count >= Horizon) {
                PositionError.RemoveAt(0);
            }
            while(RotationError.Count >= Horizon) {
                RotationError.RemoveAt(0);
            }
        }
        History.Add(new Sample(Root, Target));
        PositionError.Add(Vector3.Distance(Root.position, Target.position));
        RotationError.Add(Quaternion.Angle(Root.rotation, Target.rotation));
    }

    void OnRenderObject() {
        UltiDraw.Begin();
        UltiDraw.SetDepthRendering(DepthRendering);

        if(DrawErrors) {
            for(int i=0; i<History.Count; i++) {
                UltiDraw.DrawLine(History[i].Target.GetPosition(), History[i].Root.GetPosition(), ErrorSize, 0f, ErrorColor);
            }
        }

        if(DrawLines) {
            if(DrawTarget) {
                for(int i=1; i<History.Count; i++) {
                    UltiDraw.DrawLine(History[i-1].Target.GetPosition(), History[i].Target.GetPosition(), LineThickness, TargetColor);
                }
            }
            if(DrawRoot) {
                for(int i=1; i<History.Count; i++) {
                    UltiDraw.DrawLine(History[i-1].Root.GetPosition(), History[i].Root.GetPosition(), LineThickness, RootColor);
                }
            }
        }

        if(DrawPositions) {
            if(DrawTarget) {
                for(int i=0; i<History.Count; i++) {
                    UltiDraw.DrawSphere(History[i].Target, PointThickness, TargetColor);
                }
            }
            if(DrawRoot) {
                for(int i=0; i<History.Count; i++) {
                    UltiDraw.DrawSphere(History[i].Root, PointThickness, RootColor);
                }
            }
        }

        if(DrawDirections) {
            if(DrawTarget) {
                for(int i=0; i<History.Count; i++) {
                    UltiDraw.DrawLine(History[i].Target.GetPosition(), History[i].Target.GetPosition() + DirectionLength*History[i].Target.GetForward(), 0.05f, 0f, TargetColor);
                }
            }
            if(DrawRoot) {
                for(int i=0; i<History.Count; i++) {
                    UltiDraw.DrawLine(History[i].Root.GetPosition(), History[i].Root.GetPosition() + DirectionLength*History[i].Root.GetForward(), 0.05f, 0f, RootColor);
                }
            }
        }

        UltiDraw.SetDepthRendering(false);
        UltiDraw.End();
    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(MetricRootAccuracy), true)]
	public class MetricRootAccuracyEditor : Editor {

        public override bool RequiresConstantRepaint() => true;

		public override void OnInspectorGUI() {
            MetricRootAccuracy instance = (MetricRootAccuracy)target;
			DrawDefaultInspector();
            EditorGUILayout.HelpBox("Position Mean:" + instance.PositionError.ToArray().Mean(), MessageType.None);
            EditorGUILayout.HelpBox("Rotation Mean:" + instance.RotationError.ToArray().Mean(), MessageType.None);
		}

	}
	#endif    
}
