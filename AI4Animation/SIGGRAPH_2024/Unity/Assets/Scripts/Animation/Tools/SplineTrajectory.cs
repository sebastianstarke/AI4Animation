using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class SplineTrajectory : MonoBehaviour {
    
    public enum MODE_DIRECTION {Manual, Automatic, Identity}

    [Header("Trajectory")]
    public float Duration = 1f;
    public float Anticipation = 0.5f;
    public int Resolution = 30;
    public MODE_DIRECTION DirectionMode = MODE_DIRECTION.Automatic;
    
    [Header("Drawing")]
    public bool DrawControlPoints = true;
    public bool DrawTrajectory = true;
    public bool DrawCurrent = true;
    public bool DrawTarget = true;
    public bool DepthRendering = false;

    [Header("Visualization")]
    public Color LineColor = UltiDraw.Black;
    public float LineThickness = 0.025f;
    public Color PositionColor = UltiDraw.Black;
    public float PositionSize = 0.1f;
    public Color DirectionColor = UltiDraw.Mustard;
    [Range(0f,1f)] public float DirectionLength = 0.5f;
    public float RenderOffset = 0f;

    [Header("External")]
    public Transform CurrentPoint;
    public Transform TargetPoint;

    private float Step {get{return 1f/(Resolution-1);}}

    void Update() {
        if(CurrentPoint != null) {
            CurrentPoint.SetTransformation(GetCurrentPoint());
        }
        if(TargetPoint != null) {
            TargetPoint.SetTransformation(GetTargetPoint());
        }
    }

    public Matrix4x4 GetCurrentPoint() {
        return GetSplinePoint(Percentage(Time.time));
    }

    public Matrix4x4 GetTargetPoint() {
        return GetSplinePoint(Percentage(Time.time + Anticipation));
    }

    private float Percentage(float time) {
        if(!Application.isPlaying || Duration <= 0f) {
            return 0f;
        }
        return Mathf.Repeat(time, Duration) / Duration;
    }

    private Matrix4x4[] GetControlPoints() {
        Transform[] transforms = transform.GetChilds();
        if(transforms.Length == 0) {
            return new Matrix4x4[]{Matrix4x4.identity};
        }
        Matrix4x4[] points = new Matrix4x4[transforms.Length];
        for(int i=0; i<transforms.Length; i++) {
            points[i] = transforms[i].GetWorldMatrix();
        }
        return points;
    }

    private Matrix4x4 GetSplinePoint(float percentage) {
        Matrix4x4[] points = GetControlPoints();
        Vector3[] positions = points.GetPositions();
        Quaternion[] rotations = points.GetRotations();
        Vector3 pos = UltiMath.UltiMath.GetPointOnSpline(positions, percentage);
        Quaternion rot = Quaternion.identity;
        if(DirectionMode == MODE_DIRECTION.Manual) {
            rot = UltiMath.UltiMath.GetPointOnSpline(rotations, percentage);
        }
        if(DirectionMode == MODE_DIRECTION.Automatic) {
            rot = Quaternion.LookRotation((UltiMath.UltiMath.GetPointOnSpline(positions, percentage + Step) - UltiMath.UltiMath.GetPointOnSpline(positions, percentage)).normalized, Vector3.up);
        }
        return Matrix4x4.TRS(pos, rot, Vector3.one);
    }

    void OnDrawGizmos() {
        if(!Application.isPlaying) {
            OnRenderObject();
        }
    }

    void OnRenderObject() {
        UltiDraw.Begin();
        UltiDraw.SetDepthRendering(DepthRendering);

        Vector3 offset = new Vector3(0f, RenderOffset, 0f);

        if(DrawControlPoints) {
            foreach(Matrix4x4 point in GetControlPoints()) {
                UltiDraw.DrawSphere(point.GetPosition() + offset, Quaternion.identity, 0.1f, UltiDraw.Red);
            }
        }

        if(DrawTrajectory) {
            for(int i=0; i<Resolution; i++) {
                float ratio = i.Ratio(0, Resolution-1);
                Matrix4x4 current = GetSplinePoint(ratio);
                UltiDraw.DrawSphere(current.GetPosition() + offset, Quaternion.identity, PositionSize, PositionColor);
                if(DirectionMode != MODE_DIRECTION.Identity) {
                    UltiDraw.DrawLine(current.GetPosition() + offset, current.GetPosition() + offset + DirectionLength*current.GetForward(), Vector3.up, 0.05f, 0f, DirectionColor);
                }
                if(i<Resolution-1) {
                    Matrix4x4 next = GetSplinePoint(ratio+Step);
                    UltiDraw.DrawLine(current.GetPosition() + offset, next.GetPosition() + offset, Vector3.up, LineThickness, LineColor);
                }
            }
        }

        if(DrawCurrent) {
            Matrix4x4 point = GetCurrentPoint();
            UltiDraw.DrawSphere(point.GetPosition() + offset, point.GetRotation(), 0.25f, UltiDraw.Red);
            UltiDraw.DrawLine(point.GetPosition() + offset, point.GetPosition() + offset + point.GetForward(), Vector3.up, 0.1f, 0f, UltiDraw.Red.Opacity(0.5f));
        }

        if(DrawTarget) {
            Matrix4x4 point = GetTargetPoint();
            UltiDraw.DrawSphere(point.GetPosition() + offset, point.GetRotation(), 0.25f, UltiDraw.Cyan);
            UltiDraw.DrawLine(point.GetPosition() + offset, point.GetPosition() + offset + point.GetForward(), Vector3.up, 0.1f, 0f, UltiDraw.Cyan.Opacity(0.5f));
        }

        UltiDraw.SetDepthRendering(false);
        UltiDraw.End();
    }

    #if UNITY_EDITOR
    [CustomEditor(typeof(SplineTrajectory))]
    public class Inspector : Editor {
        public override void OnInspectorGUI() {
            SplineTrajectory instance = (SplineTrajectory)target;

            DrawDefaultInspector();

            if(GUI.changed) {
                SceneView.RepaintAll();
            }
        }
    }
    #endif
}