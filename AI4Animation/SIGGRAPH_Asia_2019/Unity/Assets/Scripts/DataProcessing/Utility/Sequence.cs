#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

[System.Serializable]
public class Sequence {
	public int Start;
	public int End;

	public Sequence(int start, int end) {
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

	// public int[] GetPivots() {
	// 	int[] indices = new int[End-Start+1];
	// 	for(int i=0; i<indices.Length; i++) {
	// 		indices[i] = Start+i;
	// 	}
	// 	return indices;
	// }

	public int[] GetIndices() {
		int[] indices = new int[End-Start+1];
		for(int i=0; i<indices.Length; i++) {
			indices[i] = Start+i-1;
		}
		return indices;
	}

	public void Inspector() {
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
	}
}
#endif