#if UNITY_EDITOR
using UnityEngine;

namespace AI4Animation {
	public abstract class SceneEvent : MonoBehaviour {
		public abstract void Callback(MotionEditor editor);
	}
}

#endif