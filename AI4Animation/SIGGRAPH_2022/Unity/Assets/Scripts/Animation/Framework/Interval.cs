using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	[System.Serializable]
	public class Interval {
		public int Start;
		public int End;

		public Interval(Interval interval) {
			Start = interval.Start;
			End = interval.End;
		}

		public Interval(int start, int end) {
			SetStart(start);
			SetEnd(end);
		}

		public int GetLength() {
			return End - Start + 1;
		}

		public float GetDuration(float framerate) {
			return GetLength() / framerate;
		}

		public void SetStart(int index) {
			Start = index;
		}

		public void SetEnd(int index) {
			End = index;
		}

		public bool Contains(int index) {
			return index >= Start && index <= End;
		}

		public int[] GetIndices() {
			int[] indices = new int[End-Start+1];
			for(int i=0; i<indices.Length; i++) {
				indices[i] = Start+i-1;
			}
			return indices;
		}

		#if UNITY_EDITOR
		public bool Inspector() {
			EditorGUI.BeginChangeCheck();
			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				Utility.SetGUIColor(UltiDraw.White);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.BeginHorizontal();
					Start = EditorGUILayout.IntField("Start", Start);
					End = EditorGUILayout.IntField("End", End);
					EditorGUILayout.EndHorizontal();
				}
			}
			return EditorGUI.EndChangeCheck();
		}

		public bool Inspector(int min, int max) {
			EditorGUI.BeginChangeCheck();
			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				Utility.SetGUIColor(UltiDraw.White);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.BeginHorizontal();
					Start = Mathf.Clamp(EditorGUILayout.IntField("Start", Start), min, max);
					End = Mathf.Clamp(EditorGUILayout.IntField("End", End), min, max);
					EditorGUILayout.EndHorizontal();
				}
			}
			return EditorGUI.EndChangeCheck();
		}
		#endif
	}
}