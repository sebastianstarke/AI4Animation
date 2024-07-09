using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UltiMath {
    public class UltiMath : MonoBehaviour {

        public static Vector3 GetPointOnSpline(Vector3[] values, float percentage) {
            if(values.Length == 0) {
                Debug.Log("No points provided.");
                return Vector3.zero;
            }

            if(values.Length == 1) {
                return values[0];
            }

            List<Vector3> tmp = new List<Vector3>(values);
            tmp.Insert(0, values[0]);
            tmp.Add(values.Last());
            values = tmp.ToArray();
            
            //Convert the input range (0 to 1) to range (0 to numSections)
            int numSections = values.Length - 3;
            int curPoint = Mathf.Min(Mathf.FloorToInt(percentage * (float)numSections), numSections - 1);
            float t = percentage * (float)numSections - (float)curPoint;

            //Get the 4 control points around the location to be sampled.
            Vector3 p0 = values[curPoint];
            Vector3 p1 = values[curPoint + 1];
            Vector3 p2 = values[curPoint + 2];
            Vector3 p3 = values[curPoint + 3];

            //The Catmull-Rom spline can be written as:
            // 0.5 * (2*P1 + (-P0 + P2) * t + (2*P0 - 5*P1 + 4*P2 - P3) * t^2 + (-P0 + 3*P1 - 3*P2 + P3) * t^3)
            //Variables P0 to P3 are the control points.
            //Variable t is the position on the spline, with a range of 0 to numSections.
            //C# way of writing the function. Note that f means float (to force precision).
            Vector3 result = .5f * (2f * p1 + (-p0 + p2) * t + (2f * p0 - 5f * p1 + 4f * p2 - p3) * (t * t) + (-p0 + 3f * p1 - 3f * p2 + p3) * (t * t * t));

            return result;
        }

        public static Quaternion GetPointOnSpline(Quaternion[] values, float percentage) {
            Vector3[] forwards = new Vector3[values.Length];
            Vector3[] ups = new Vector3[values.Length];
            for(int i=0; i<values.Length; i++) {
                forwards[i] = values[i].GetForward();
                ups[i] = values[i].GetUp();
            }
            return Quaternion.LookRotation(
                GetPointOnSpline(forwards, percentage).normalized,
                GetPointOnSpline(ups, percentage).normalized
            );
        }

        public static Matrix4x4[] GetPointsOnSpline(Matrix4x4[] values, int resolution){
            return GetPointsOnSpline(values.GetPositions(), values.GetRotations(), resolution);
        }

        public static Matrix4x4[] GetPointsOnSpline(Vector3[] positions, Quaternion[] rotations, int resolution){
            Vector3[] splinePositions = GetPointsOnSpline(positions, resolution);
            Quaternion[] splineRotations = GetPointsOnSpline(rotations, resolution);
            Matrix4x4[] spline = new Matrix4x4[resolution];
            for (int i = 0; i < spline.Length; i++)
            {
                spline[i] = Matrix4x4.TRS(splinePositions[i], splineRotations[i], Vector3.one);
            }
            return spline;
        }

        public static Vector3[] GetPointsOnSpline(Vector3[] positions, int resolution) {
            Vector3[] result = new Vector3[resolution];
            for(int i=0; i<resolution; i++) {
                result[i] = GetPointOnSpline(positions, i.Ratio(0, resolution-1));
            }
            return result;
        }

        public static Quaternion[] GetPointsOnSpline(Quaternion[] rotations, int resolution) {
            Quaternion[] result = new Quaternion[resolution];
            for(int i=0; i<resolution; i++) {
                result[i] = GetPointOnSpline(rotations, i.Ratio(0, resolution-1));
            }
            return result;
        }

    }
}