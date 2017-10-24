using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(BioAnimation))]
public class BioAnimation_Editor : Editor {

		public BioAnimation Target;

		void Awake() {
			Target = (BioAnimation)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			//DrawDefaultInspector();

			Target.Controller.Inspector();
			Target.Character.Inspector();
			Target.Trajectory.Inspector();
			Target.PFNN.Inspector();

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

		void OnSceneGUI() {
			
		}

		private void DrawSphere(Vector3 position, float radius, Color color) {
			Handles.color = color;
			Handles.SphereHandleCap(0, position, Quaternion.identity, radius, EventType.Repaint);
		}

		private void DrawCube(Vector3 position, Quaternion rotation, float size, Color color) {
			Handles.color = color;
			Handles.CubeHandleCap(0, position, rotation, size, EventType.Repaint);
		}

		private void DrawLine(Vector3 a, Vector3 b, float width, Color color) {
			Handles.color = color;
			Handles.DrawAAPolyLine(width, new Vector3[2] {a,b});
		}

		private void DrawDottedLine(Vector3 a, Vector3 b, float width, Color color) {
			Handles.color = color;
			Handles.DrawDottedLine(a, b, width);
		}

		private void DrawArrow(Vector3 position, Quaternion rotation, float length, Color color) {
			Handles.color = color;
			Handles.ArrowHandleCap(0, position, rotation, length, EventType.repaint);
		}

		private void DrawSolidArc(Vector3 position, Vector3 normal, Vector3 from, float angle, float radius, Color color) {
			Handles.color = color;
			Handles.DrawSolidArc(position, normal, from, angle, radius);
		}

		private void SetGUIColor(Color color) {
			GUI.backgroundColor = color;
		}
}