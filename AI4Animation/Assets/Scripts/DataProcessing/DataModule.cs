#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public abstract class DataModule : ScriptableObject {

	public enum TYPE {Style, Phase};

	public MotionData Data;
	public TYPE Type;

	public DataModule Initialise(MotionData data) {
		Data = data;
		return this;
	}

	public abstract void Inspector();

}
#endif