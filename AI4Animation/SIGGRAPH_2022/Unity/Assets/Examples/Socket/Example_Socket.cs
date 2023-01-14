using UnityEngine;
using UnityEditor;
using AI4Animation;

public class Example_Socket : MonoBehaviour {

    public SocketNetwork Model;

    void Awake() {
        Model.CreateSession();
    }

    void OnDestroy() {
        Model.CloseSession();
    }

    void Update() {
        
    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(Example_Socket), true)]
	public class Example_Socket_Editor : Editor {

		public Example_Socket Target;

		void Awake() {
			Target = (Example_Socket)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

            Target.Model.Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}
	#endif
}
