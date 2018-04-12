using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WorldScaler : MonoBehaviour {

	public float Rescale = 1f;

    [ContextMenu("Apply")]
    public void Apply() {
		transform.localScale *= Rescale;
		Apply(transform);
	}

	private void Apply(Transform t) {
		foreach(Terrain terrain in t.GetComponents<Terrain>()) {
			terrain.terrainData.size *= Rescale;
			//Debug.Log(terrain.terrainData.size);
		}
		for(int i=0; i<t.childCount; i++) {
			Apply(t.GetChild(i));
		}
	}

}
