#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

public class RealignBone : MonoBehaviour {

    public Transform From;
    public Transform To;
    public Transform Target;
    public bool Rescale = false;

    // [ContextMenu("Process")]
    // public void Process() {
    //     Undo.RecordObject(this, "RealignBone"+name);
    //     From.rotation = Quaternion.FromToRotation(To.position - From.position, Target.position - From.position) * From.rotation;
    // }

    [CustomEditor(typeof(RealignBone))]
    public class RealignBoneEditor : Editor {
        public override void OnInspectorGUI() {
            DrawDefaultInspector();
            RealignBone t = (target as RealignBone);
            if(Utility.GUIButton("Process", UltiDraw.DarkGrey, UltiDraw.White)) {
                Undo.RecordObject(t.From.transform, "RealignBoneRotation"+t.From.name);
                t.From.rotation = Quaternion.FromToRotation(t.To.position - t.From.position, t.Target.position - t.From.position) * t.From.rotation;
                if(t.Rescale) {
                    Undo.RecordObject(t.To.transform, "RealignBonePosition"+t.To.name);
                    t.To.position = t.Target.position;
                }
            }
        }
    }
}
#endif