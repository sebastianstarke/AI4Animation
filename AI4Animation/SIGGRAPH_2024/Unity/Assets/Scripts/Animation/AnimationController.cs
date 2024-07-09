using UnityEngine;
using AI4Animation;

public abstract class AnimationController : MonoBehaviour {

	public Actor Actor;
	public float Framerate = 30f;

	protected abstract void Setup();
	protected abstract void Destroy();
	protected abstract void Control();
	protected abstract void OnGUIDerived();
	protected abstract void OnRenderObjectDerived();

	void Reset() {
		Actor = GetComponent<Actor>();
	}

    void Start() {
		Time.fixedDeltaTime = 1f/Framerate;
		Utility.SetFPS(Mathf.RoundToInt(Framerate));
		Setup();
    }

	void OnDestroy() {
		Destroy();
	}

	void FixedUpdate() {
		Control();
	}

    void OnGUI() {
		OnGUIDerived();
    }

	void OnRenderObject() {
		OnRenderObjectDerived();
	}
	
}