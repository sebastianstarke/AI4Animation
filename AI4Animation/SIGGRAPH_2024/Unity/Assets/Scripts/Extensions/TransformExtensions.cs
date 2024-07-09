using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class TransformExtensions {
	public static void SetTransformation(this Transform transform, Matrix4x4 matrix) {
		transform.SetPositionAndRotation(matrix.GetPosition(), matrix.GetRotation());
	}

	public static void BlendTransformation(this Transform transform, Matrix4x4 matrix, float blend) {
		transform.SetPositionAndRotation(Vector3.Lerp(transform.position, matrix.GetPosition(), blend), Quaternion.Slerp(transform.rotation, matrix.GetRotation(), blend));
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

	public static Transform[] GetAllChilds(this Transform t) {
		Transform[] childs = t.GetComponentsInChildren<Transform>(true);
		ArrayExtensions.RemoveAt(ref childs, 0);
		return childs;
	}

	public static bool HasParentInHierarchy(this Transform root, Transform parent) {
		while(root != root.root) {
			root = root.parent;
			if(root == parent) {
				return true;
			}
		}
		return false;
	}
	
	public static T FindComponentInParents<T>(this Transform t) {
		while(t != t.root) {
			t = t.parent;
			if(t.GetComponent<T>() != null) {
				return t.GetComponent<T>();
			}
		}
		return default(T);
	}


	public static Transform FindRecursive(this Transform root, string name) {
		foreach(Transform child in root.GetComponentsInChildren<Transform>(true)) {
			if(child.name == name) {
				return child;
			}
		}
		return null;
	}

	public static Transform FindRecursive(this Transform root, System.Func<Transform, bool> func) {
		foreach(Transform child in root.GetComponentsInChildren<Transform>(true)) {
			if(func(child)) {
				return child;
			}
		}
		return null;
	}

	public static Transform[] FindRecursive(this Transform root, params string[] names) {
		List<Transform> childs = new List<Transform>();
		foreach(Transform child in root.GetComponentsInChildren<Transform>(true)) {
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
		Transform[] childs = transform.GetChilds();
		transform.DetachChildren();
		// Vector3[] positions = new Vector3[transform.childCount];
		// Quaternion[] rotations = new Quaternion[transform.childCount];
		// for(int i=0; i<transform.childCount; i++) {
		// 	positions[i] = transform.GetChild(i).position;
		// 	rotations[i] = transform.GetChild(i).rotation;
		// }
		transform.rotation = rotation;
		foreach(Transform child in childs) {
			child.SetParent(transform, true);
		}
		// for(int i=0; i<transform.childCount; i++) {
		// 	transform.GetChild(i).position = positions[i];
		// 	transform.GetChild(i).rotation = rotations[i];
		// }
	}

	public static void OverrideTransformation(this Transform transform, Matrix4x4 transformation) {
		transform.OverridePositionAndRotation(transformation.GetPosition(), transformation.GetRotation());
	}

	public static void OverridePositionAndRotation(this Transform transform, Vector3 position, Quaternion rotation) {
		Matrix4x4[] matrices = new Matrix4x4[transform.childCount];
		for(int i=0; i<transform.childCount; i++) {
			matrices[i] = transform.GetChild(i).GetWorldMatrix();
		}
		transform.position = position;
		transform.rotation = rotation;
		for(int i=0; i<transform.childCount; i++) {
			transform.GetChild(i).position = matrices[i].GetPosition();
			transform.GetChild(i).rotation = matrices[i].GetRotation();
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
