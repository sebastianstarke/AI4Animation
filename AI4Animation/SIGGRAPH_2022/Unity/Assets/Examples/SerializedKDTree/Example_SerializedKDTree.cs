using UnityEngine;
using UnityEditor;
using System;
using System.Collections.Generic;
using SerializedKDTree;

public class Example_SerializedKDTree : MonoBehaviour {

    public int Count = 10;
    public int K = 5;
    public SerializedKDTree.KDTree KDTree = null;

    public float Size = 1f;
    public float Min = -1f;
    public float Max = 1f;

    public List<Vector3> Vectors = null;

    public class Example_Vector3 : KDNode.Value {
        public Vector3 Vector;
        public Example_Vector3(Vector3 vector) {
            Vector = vector;
        }
    }

    public void BuildDatabase() {
        Vectors.Clear();
        KDTree = ScriptableObjectExtensions.Create<SerializedKDTree.KDTree>(string.Empty, "Database");
        KDTree.Initialize(3);
        for(int i=0; i<Count; i++) {
            Vector3 point = new Vector3(UnityEngine.Random.Range(Min, Max), UnityEngine.Random.Range(Min, Max), UnityEngine.Random.Range(Min, Max));
            KDTree.AddPoint(new float[3]{point.x, point.y, point.z}, new Example_Vector3(point));
            Vectors.Add(point);
        }
        KDTree.Finish();
    }

    [CustomEditor(typeof(Example_SerializedKDTree))]
    public class Example_SerializedKDTreeEditor : Editor {

        public Example_SerializedKDTree Target;

        void Awake() {
            Target = (Example_SerializedKDTree)target;
        }

        public override void OnInspectorGUI() {
            DrawDefaultInspector();
            if(Utility.GUIButton("Build Database", UltiDraw.DarkGrey, UltiDraw.White)) {
                Target.BuildDatabase();
            }
            if(GUI.changed) {
                EditorUtility.SetDirty(Target);
            }
        }
    }

    void OnDrawGizmos() {
        if(!Application.isPlaying) {
            return;
        }
        if(KDTree == null) {
            return;
        }

        Gizmos.color = Color.black;
        foreach(Vector3 v in Vectors) {
            Gizmos.DrawSphere(v, Size);
        }

        Gizmos.color = Color.red;
        Gizmos.DrawSphere(transform.position, Size);

        Gizmos.color = Color.green;
        NearestNeighbour neighbours = KDTree.NearestNeighbors(new float[3]{transform.position.x, transform.position.y, transform.position.z}, K);
        foreach(Example_Vector3 neighbour in neighbours) {
            Gizmos.DrawSphere(neighbour.Vector, 2f*Size);
        }
    }

}
