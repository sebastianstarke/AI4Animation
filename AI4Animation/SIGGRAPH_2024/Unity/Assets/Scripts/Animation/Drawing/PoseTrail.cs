using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoseTrail : MonoBehaviour {

	public int TrailLength = 10;
	public float TimeDifference = 0.1f;
	public float Transparency = 0.25f;
	public Color MeshColor = Color.black;
	public float FadePower = 1f;
	public bool ApplyOpacityFade = false;
	public bool ApplyRainbowColors = false;

    [NonSerialized] public KeyCode ResetKey = KeyCode.K;

	private GameObject Prototype;
	private Queue<GameObject> Instances;

	private float Timestamp;

	void Start() {
		Instances = new Queue<GameObject>();
		Timestamp = Time.time;
		Prototype = CreatePrototype();
	}

	void OnDestroy() {
		if(Prototype != null) {
			Utility.Destroy(Prototype);
		}
	}

	void Update() {
		if(Input.GetKey(ResetKey)) {
			foreach(GameObject instance in Instances) {
				Utility.Destroy(instance);
			}
			Instances.Clear();
		}
		
		if(Time.time - Timestamp >= TimeDifference) {
			Timestamp = Time.time;
			Instances.Enqueue(CreateInstance());
			while(Instances.Count > TrailLength) {
				Utility.Destroy(Instances.Dequeue());
			}
		}
	}

	void OnRenderObject() {
		UltiDraw.Begin();
		int index = 0;
		//GameObject previous = null;
		foreach(GameObject instance in Instances) {
			index += 1;

			//Colors
			foreach(Renderer r in instance.GetComponentsInChildren<Renderer>()) {
				if(!r.material.HasProperty("_Color")) {
					Debug.Log("Renderer has no color property.");
				} else {
					r.material.color = ApplyRainbowColors ? UltiDraw.GetRainbowColor(index, Instances.Count) : MeshColor;
				}
			}

			//Opacities
			float transparency = Transparency;
			if(ApplyOpacityFade) {
				transparency = Mathf.Pow(index.Ratio(0, Instances.Count-1), FadePower).Normalize(0f, 1f, 0f, Transparency);
			}
			instance.GetComponent<Transparency>().SetTransparency(transparency);

			// if(index > 1) {
			// 	UltiDraw.DrawSphere(instance.transform.position, Quaternion.identity, 0.025f, UltiDraw.Magenta.Opacity(0.8f));
			// }
			//previous = instance;
		}
		UltiDraw.End();
	}

	private GameObject CreatePrototype() {
		GameObject instance = Instantiate(gameObject);
		instance.name = "Prototype";
		instance.SetActive(false);
		// instance.hideFlags = HideFlags.HideInHierarchy;
		Cleanup(instance.transform);
		instance.AddComponent<Transparency>();
		return instance;
	}

	private GameObject CreateInstance() {
		GameObject instance = Instantiate(Prototype);
		instance.name = name + " (Motion Trail)";
		instance.SetActive(true);
		Copy(gameObject.transform, instance.transform);
		return instance;
	}

	private void Cleanup(Transform t) {
		foreach(Component c in t.GetComponents<Component>()) {
			if(!(c is Transform)) {
				if(c is Renderer) {
					Renderer r = (Renderer)c;
					if(!r.material.HasProperty("_Color")) {
					//	Utility.Destroy(c);
					}
				} else {
					Utility.Destroy(c);
				}
			}
		}
		for(int i=0; i<t.childCount; i++) {
			Cleanup(t.GetChild(i));
		}
	}

	private void Copy(Transform original, Transform instance) {
		instance.localPosition = original.localPosition;
		instance.localRotation = original.localRotation;
		instance.localScale = original.localScale;
		for(int i=0; i<original.childCount; i++) {
			Copy(original.GetChild(i), instance.GetChild(i));
		}
	}

}
