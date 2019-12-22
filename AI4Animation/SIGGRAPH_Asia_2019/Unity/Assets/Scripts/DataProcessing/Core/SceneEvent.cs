#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class SceneEvent : MonoBehaviour {

	[System.NonSerialized] public bool Blocked = false;

	public abstract void Callback(MotionEditor editor);
	public abstract void Identity(MotionEditor editor);

}
#endif