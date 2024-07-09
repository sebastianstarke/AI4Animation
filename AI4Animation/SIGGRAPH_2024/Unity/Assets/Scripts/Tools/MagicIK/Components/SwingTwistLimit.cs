using System;
using UnityEngine;
using UnityEditor;

namespace MagicIK {
    // [ExecuteInEditMode]
    public class SwingTwistLimit : Limit {
        public enum AXIS {X, Y, Z}
        public AXIS SwingAxis = AXIS.Z;
        public AXIS TwistAxis = AXIS.X;
        public float LowerLimit = 0f;
        public float UpperLimit = 0f;
        public float TwistLimit = 0f;
        
        private const float Size = 0.25f;

        private Vector3 PrimaryAxis {get{
            switch(SwingAxis) {
                case AXIS.X: return Vector3.right;
                case AXIS.Y: return Vector3.up;
                case AXIS.Z: return Vector3.forward;
                default: return Vector3.zero;
            };
        }}
        
        private Vector3 SecondaryAxis {get{
            switch(TwistAxis) {
                case AXIS.X: return Vector3.right;
                case AXIS.Y: return Vector3.up;
                case AXIS.Z: return Vector3.forward;
                default: return Vector3.zero;
            };
        }}

        private Vector3 CrossAxis {get{
            return Vector3.Cross(PrimaryAxis, SecondaryAxis);
        }}

        void OnValidate() {
            LowerLimit = Mathf.Clamp(LowerLimit, -180f, 0f);
            UpperLimit = Mathf.Clamp(UpperLimit, 0f, 180f);
            TwistLimit = Mathf.Clamp(TwistLimit, 0f, 180f);
        }

        // void Update() {
        //     // Debug.Log("SOLVING:" + transform.name);
        //     // Solve();
        // }

        public override void Solve() {
            //Swing
            float swing = Mathf.Clamp(GetSwingAngle(), LowerLimit, UpperLimit);
            transform.localRotation = Quaternion.FromToRotation(
                transform.localRotation * SecondaryAxis, 
                ZeroRotation * Quaternion.AngleAxis(swing, PrimaryAxis) * SecondaryAxis
            ) * transform.localRotation;

            //Twist
            float twist = Mathf.Clamp(GetTwistAngle(swing), -TwistLimit, TwistLimit);
            transform.localRotation = Quaternion.FromToRotation(
                transform.localRotation * CrossAxis, 
                ZeroRotation * Quaternion.AngleAxis(swing, PrimaryAxis) * Quaternion.AngleAxis(twist, SecondaryAxis) * CrossAxis
            ) * transform.localRotation;
        }

        private float GetSwingAngle() {
            return Vector3.SignedAngle(
                ZeroRotation * SecondaryAxis,
                transform.localRotation * SecondaryAxis,
                ZeroRotation * PrimaryAxis
            );
        }

        private float GetTwistAngle(float swing) {
            return Vector3.SignedAngle(
                ZeroRotation * Quaternion.AngleAxis(swing, PrimaryAxis) * CrossAxis,
                transform.localRotation * CrossAxis,
                ZeroRotation * Quaternion.AngleAxis(swing, PrimaryAxis) * SecondaryAxis
            );
        }
        #if UNITY_EDITOR
        [CustomEditor(typeof(SwingTwistLimit))]
        public class SwingTwistLimitEditor : Editor {
            
            private SwingTwistLimit Instance;

            public override void OnInspectorGUI() {
                Instance = (SwingTwistLimit)target;
                DrawDefaultInspector();
                EditorGUILayout.HelpBox("Swing: " + Instance.GetSwingAngle() + " Twist: " + Instance.GetTwistAngle(Instance.GetSwingAngle()), MessageType.None);
                if(Utility.GUIButton("Reset", UltiDraw.DarkGrey, UltiDraw.White)) {
                    Instance.Zero();
                }
            }

            void OnSceneGUI() {
                Instance = (SwingTwistLimit)target;

                Handles.color = Color.cyan.Opacity(0.1f);
                Handles.ArrowHandleCap(0, Instance.transform.position, Quaternion.LookRotation(GetPrimary()), Size, EventType.Repaint);
                Handles.DrawSolidArc(Instance.transform.position, GetPrimary(), GetSecondary(Instance.LowerLimit), Instance.UpperLimit-Instance.LowerLimit, Size);
                Handles.color = Color.cyan;
                Handles.DrawLine(Instance.transform.position, Instance.transform.position + Size * GetSecondary(Instance.LowerLimit));
                Handles.DrawLine(Instance.transform.position, Instance.transform.position + Size * GetSecondary(Instance.UpperLimit));
                Handles.color = Color.red;
                Handles.DrawLine(Instance.transform.position, Instance.transform.position + Size * GetSecondary(Instance.GetSwingAngle()));

                Handles.color = Color.magenta.Opacity(0.1f);
                Handles.ArrowHandleCap(0, Instance.transform.position, Quaternion.LookRotation(GetCross(0f)), Size, EventType.Repaint);
                Handles.DrawSolidArc(Instance.transform.position, GetSecondary(0f), GetCross(-Instance.TwistLimit), 2f*Instance.TwistLimit, Size);
                Handles.color = Color.magenta;
                Handles.DrawLine(Instance.transform.position, Instance.transform.position + Size * GetCross(-Instance.TwistLimit));
                Handles.DrawLine(Instance.transform.position, Instance.transform.position + Size * GetCross(Instance.TwistLimit));
                Handles.color = Color.red;
                Handles.DrawLine(Instance.transform.position, Instance.transform.position + Size * GetCross(Instance.GetTwistAngle(Instance.GetSwingAngle())));
            }

            private Vector3 GetPrimary() {
                return Instance.transform.parent.rotation * Instance.ZeroRotation * Instance.PrimaryAxis;
            }

            private Vector3 GetSecondary(float value) {
                return Instance.transform.parent.rotation * Instance.ZeroRotation * Quaternion.AngleAxis(value, Instance.PrimaryAxis) * Instance.SecondaryAxis;
            }

            private Vector3 GetCross(float value) {
                return Instance.transform.parent.rotation * Instance.ZeroRotation * Quaternion.AngleAxis(value, Instance.SecondaryAxis) * Instance.CrossAxis;
            }
        }
        #endif
    }
}