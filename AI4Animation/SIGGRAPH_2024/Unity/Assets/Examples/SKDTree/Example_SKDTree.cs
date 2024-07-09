using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System;

public class Example_SKDTree : MonoBehaviour {

    public int Count = 10;
    public int K = 5;
    public Example_Database Database = null;

    public float Size = 1f;
    public float Min = -1f;
    public float Max = 1f;

    public List<Vector3> Points = null;
    [Serializable]
    public class Value {
        public Vector3 Point;
        public Color Color;
        public Value(Vector3 point, Color color) {
            Point = point;
            Color = color;
        }
    }

    void Awake() {
        
        Database.Build();
    }

    void OnRenderObject() {
        if(Database == null) {
            return;
        }

        UltiDraw.Begin();
        foreach(Vector3 p in Points) {
            UltiDraw.DrawSphere(p, Quaternion.identity, Size, Color.black);
        }

        UltiDraw.DrawSphere(transform.position, Quaternion.identity, Size, Color.red);

        Value[] results = Database.Query(new float[3]{transform.position.x, transform.position.y, transform.position.z}, K);
        foreach(Value neighbour in results) {
            UltiDraw.DrawSphere(neighbour.Point, Quaternion.identity, 2f*Size, neighbour.Color);
        }
        UltiDraw.End();
    }

    #if UNITY_EDITOR
    [CustomEditor(typeof(Example_SKDTree))]
    public class Example_SKDTreeEditor : Editor {

        public Example_SKDTree Target;

        void Awake() {
            Target = (Example_SKDTree)target;
        }

        public void CreateDatabase() {
            Target.Points.Clear();
            for(int i=0; i<Target.Count; i++) {
                Vector3 point = new Vector3(UnityEngine.Random.Range(Target.Min, Target.Max), UnityEngine.Random.Range(Target.Min, Target.Max), UnityEngine.Random.Range(Target.Min, Target.Max));
                Target.Points.Add(point);
            }

            Target.Database = ScriptableObjectExtensions.Create<Example_Database>(Target.Database, "Database");
            for(int i=0; i<Target.Points.Count; i++) {
                Vector3 point = Target.Points[i];
                Value value = new Value(point, UltiDraw.GetRandomColor());
                Target.Database.AddSample(
                    new float[3]{point.x, point.y, point.z},
                    value
                );
            }
            Target.Database.Save();
        }

        public override void OnInspectorGUI() {
            DrawDefaultInspector();
            if(Utility.GUIButton("Create Database", UltiDraw.DarkGrey, UltiDraw.White)) {
                CreateDatabase();
            }
            if(GUI.changed) {
                EditorUtility.SetDirty(Target);
            }
        }
    }
    #endif

}
