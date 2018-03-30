using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class TransformExtensions {

	public static Matrix4x4 GetLocalMatrix(this Transform transform) {
		return Matrix4x4.TRS(transform.localPosition, transform.localRotation, transform.localScale);
	}

	public static Matrix4x4 GetWorldMatrix(this Transform transform) {
		return Matrix4x4.TRS(transform.position, transform.rotation, transform.lossyScale);
	}

}
