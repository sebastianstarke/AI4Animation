#if UNITY_EDITOR
using UnityEngine;
public abstract class SceneEvent : MonoBehaviour {
	public abstract void Callback(MotionEditor editor);
}
#endif