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

	public static Matrix4x4 GetLocalMatrix(this Transform transform) {
		return Matrix4x4.TRS(transform.localPosition, transform.localRotation, transform.localScale);
	}

	public static Matrix4x4 GetGlobalMatrix(this Transform transform) {
		return Matrix4x4.TRS(transform.position, transform.rotation, transform.lossyScale);
	}

	public static Matrix4x4 GetWorldMatrix(this Transform transform) {
		return Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one);
	}

	public static Matrix4x4 GetTransformationMatrix(this Transform transform) {
		return transform.localToWorldMatrix;
	}

	public static Transform[] GetChilds(this Transform t) {
		Transform[] childs = new Transform[t.childCount];
		for(int i=0; i<childs.Length; i++) {
			childs[i] = t.GetChild(i);
		}
		return childs;
	}

	public static Transform FindRecursive(this Transform root, string name) {
		foreach(Transform child in root.GetComponentsInChildren<Transform>()) {
			if(child.name == name) {
				return child;
			}
		}
		return null;
	}

	public static Transform[] FindRecursive(this Transform root, params string[] names) {
		List<Transform> childs = new List<Transform>();
		foreach(Transform child in root.GetComponentsInChildren<Transform>()) {
			if(names.Contains(child.name)) {
				childs.Add(child);
			}
		}
		return childs.ToArray();
	}

	public static void OverridePosition(this Transform transform, Vector3 position) {
		Vector3[] positions = new Vector3[transform.childCount];
		Quaternion[] rotations = new Quaternion[transform.childCount];
		for(int i=0; i<transform.childCount; i++) {
			positions[i] = transform.GetChild(i).position;
			rotations[i] = transform.GetChild(i).rotation;
		}
		transform.position = position;
		for(int i=0; i<transform.childCount; i++) {
			transform.GetChild(i).position = positions[i];
			transform.GetChild(i).rotation = rotations[i];
		}
	}

	public static void OverrideRotation(this Transform transform, Quaternion rotation) {
		Vector3[] positions = new Vector3[transform.childCount];
		Quaternion[] rotations = new Quaternion[transform.childCount];
		for(int i=0; i<transform.childCount; i++) {
			positions[i] = transform.GetChild(i).position;
			rotations[i] = transform.GetChild(i).rotation;
		}
		transform.rotation = rotation;
		for(int i=0; i<transform.childCount; i++) {
			transform.GetChild(i).position = positions[i];
			transform.GetChild(i).rotation = rotations[i];
		}
	}

	public static void OverridePositionAndRotation(this Transform transform, Vector3 position, Quaternion rotation) {
		Vector3[] positions = new Vector3[transform.childCount];
		Quaternion[] rotations = new Quaternion[transform.childCount];
		for(int i=0; i<transform.childCount; i++) {
			positions[i] = transform.GetChild(i).position;
			rotations[i] = transform.GetChild(i).rotation;
		}
		transform.position = position;
		transform.rotation = rotation;
		for(int i=0; i<transform.childCount; i++) {
			transform.GetChild(i).position = positions[i];
			transform.GetChild(i).rotation = rotations[i];
		}
	}

	public static void OverrideLocalPosition(this Transform transform, Vector3 position) {
		Vector3[] positions = new Vector3[transform.childCount];
		Quaternion[] rotations = new Quaternion[transform.childCount];
		for(int i=0; i<transform.childCount; i++) {
			positions[i] = transform.GetChild(i).localPosition;
			rotations[i] = transform.GetChild(i).localRotation;
		}
		transform.localPosition = position;
		for(int i=0; i<transform.childCount; i++) {
			transform.GetChild(i).localPosition = positions[i];
			transform.GetChild(i).localRotation = rotations[i];
		}
	}

	public static void OverrideLocalRotation(this Transform transform, Quaternion rotation) {
		Vector3[] positions = new Vector3[transform.childCount];
		Quaternion[] rotations = new Quaternion[transform.childCount];
		for(int i=0; i<transform.childCount; i++) {
			positions[i] = transform.GetChild(i).localPosition;
			rotations[i] = transform.GetChild(i).localRotation;
		}
		transform.localRotation = rotation;
		for(int i=0; i<transform.childCount; i++) {
			transform.GetChild(i).localPosition = positions[i];
			transform.GetChild(i).localRotation = rotations[i];
		}
	}

	public static void OverrideLocalPositionAndLocalRotation(this Transform transform, Vector3 position, Quaternion rotation) {
		Vector3[] positions = new Vector3[transform.childCount];
		Quaternion[] rotations = new Quaternion[transform.childCount];
		for(int i=0; i<transform.childCount; i++) {
			positions[i] = transform.GetChild(i).localPosition;
			rotations[i] = transform.GetChild(i).localRotation;
		}
		transform.localPosition = position;
		transform.localRotation = rotation;
		for(int i=0; i<transform.childCount; i++) {
			transform.GetChild(i).localPosition = positions[i];
			transform.GetChild(i).localRotation = rotations[i];
		}
	}

}
