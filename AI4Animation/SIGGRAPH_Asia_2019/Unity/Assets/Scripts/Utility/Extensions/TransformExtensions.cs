using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class TransformExtensions {

	public static Vector3 GetAxis(this Transform transform, Axis axis) {
		switch(axis) {
			case Axis.XPositive:
			return transform.right;
			case Axis.XNegative:
			return -transform.right;
			case Axis.YPositive:
			return transform.up;
			case Axis.YNegative:
			return -transform.up;
			case Axis.ZPositive:
			return transform.forward;
			case Axis.ZNegative:
			return -transform.forward;
		}
		return Vector3.zero;
	}

	public static Matrix4x4 GetLocalMatrix(this Transform transform, bool unitScale=false) {
		return Matrix4x4.TRS(transform.localPosition, transform.localRotation, unitScale ? Vector3.one : transform.localScale);
	}

	public static Matrix4x4 GetWorldMatrix(this Transform transform, bool unitScale=false) {
		return Matrix4x4.TRS(transform.position, transform.rotation, unitScale ? Vector3.one : transform.lossyScale);
	}

	public static void OverrideMatrix(this Transform transform, Matrix4x4 matrix) {
		Transform[] childs = new Transform[transform.childCount];
		for(int i=0; i<childs.Length; i++) {
			childs[i] = transform.GetChild(i);
		}
		transform.DetachChildren();
		transform.position = matrix.GetPosition();
		transform.rotation = matrix.GetRotation();
		for(int i=0; i<childs.Length; i++) {
			childs[i].SetParent(transform);
		}
	}

}
