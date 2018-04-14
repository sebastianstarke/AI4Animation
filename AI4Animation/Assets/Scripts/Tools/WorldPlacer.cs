using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WorldPlacer : MonoBehaviour {

	public Vector3 MinimumRotation = Vector3.zero;
	public Vector3 MaximumRotation = Vector3.zero;
	public Vector3 MinimumScale = Vector3.one;
	public Vector3 MaximumScale = Vector3.one;

    [ContextMenu("Apply")]
    public void Apply() {
		transform.localRotation = Quaternion.Euler(
			Random.Range(MinimumRotation.x, MaximumRotation.x),
			Random.Range(MinimumRotation.y, MaximumRotation.y),
			Random.Range(MinimumRotation.z, MaximumRotation.z)
		);
		transform.localScale = new Vector3(
			Random.Range(MinimumScale.x, MaximumScale.x),
			Random.Range(MinimumScale.y, MaximumScale.y),
			Random.Range(MinimumScale.z, MaximumScale.z)
		);
	}

}
