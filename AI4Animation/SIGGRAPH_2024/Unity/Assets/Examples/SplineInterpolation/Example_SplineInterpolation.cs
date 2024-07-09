using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Example_SplineInterpolation : MonoBehaviour {
    
    [Range(0f,1f)] public float Percentage = 0.5f;
    public int Resolution = 10;

    void OnDrawGizmos() {
        UltiDraw.Begin();

        Transform[] transforms = transform.GetChilds();
        Vector3[] positions = new Vector3[transforms.Length];
        Quaternion[] rotations = new Quaternion[transforms.Length];

        for(int i=0; i<positions.Length; i++) {
            positions[i] = transforms[i].position;
        }

        for(int i=0; i<rotations.Length; i++) {
            rotations[i] = transforms[i].rotation;
        }

        for(int i=0; i<positions.Length; i++) {
            Gizmos.color = Color.red;
            if(i>0) {
                Gizmos.DrawLine(positions[i-1], positions[i]);
            }
            Gizmos.DrawSphere(positions[i], 0.1f);
        }

        for(int i=0; i<rotations.Length; i++) {
            UltiDraw.DrawTranslateGizmo(positions[i], rotations[i], 0.5f);
        }

        Vector3[] splinePositions = UltiMath.UltiMath.GetPointsOnSpline(positions, Resolution);
        Quaternion[] splineRotations = UltiMath.UltiMath.GetPointsOnSpline(rotations, Resolution);
        Matrix4x4[] spline = new Matrix4x4[Resolution];
        for (int i = 0; i < spline.Length; i++)
        {
            spline[i] = Matrix4x4.TRS(splinePositions[i], splineRotations[i], Vector3.one);
        }

        for(int i=0; i<spline.Length; i++) {
            if(i>0) {
                UltiDraw.DrawLine(spline[i-1].GetPosition(), spline[i].GetPosition(), Color.green);
            }
            UltiDraw.DrawSphere(spline[i], 0.1f, Color.green);
            UltiDraw.DrawTranslateGizmo(spline[i].GetPosition(), spline[i].GetRotation(), 0.1f);
        }


        Vector3 pos = UltiMath.UltiMath.GetPointOnSpline(positions, Percentage);
        Quaternion rot = UltiMath.UltiMath.GetPointOnSpline(rotations, Percentage);
        UltiDraw.DrawPyramid(pos, rot, 0.2f, 1f, Color.blue);

        UltiDraw.End();
    }
}
