using System;
using UnityEngine;
using UnityEditor;

[Serializable]
public class Example_Abstract : MonoBehaviour {
    [SerializeReference] public BaseClass A;

    public void Generate() {
        A = new DerivedClass();
    }

    [CustomEditor(typeof(Example_Abstract))]
    public class Example_Abstract_Editor : Editor {

        public Example_Abstract Target;

        void Awake() {
            Target = (Example_Abstract)target;
        }

        public override void OnInspectorGUI() {
            DrawDefaultInspector();
            if(Utility.GUIButton("Generate", UltiDraw.DarkGrey, UltiDraw.White)) {
                Target.Generate();
                EditorUtility.SetDirty(Target);
            }
        }
    }
}

[Serializable]
public abstract class BaseClass {

}

[Serializable]
public class DerivedClass : BaseClass {
    public float Value;
}