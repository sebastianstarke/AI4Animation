using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[ExecuteInEditMode]
[RequireComponent(typeof(VoxelCollider))]
public class Interaction : MonoBehaviour {

    public VoxelCollider Geometry = null;
	public Transform[] Contacts = new Transform[0];

	public bool ShowContacts = false;

	[ContextMenu("Reorganize")]
	public void Reorganize() {
		Transform container = transform.Find("Contacts");
		if(container == null) {
			container = new GameObject("Contacts").transform;
			container.SetParent(transform);
			container.localPosition = Vector3.zero;
			container.localRotation = Quaternion.identity;
			container.localScale = Vector3.one;
		}
		foreach(Transform c in Contacts) {
			c.SetParent(container);
		}
	}

	void Awake() {
		if(Application.isPlaying) {
			BoxCollider trigger = gameObject.AddComponent<BoxCollider>();
			trigger.isTrigger = true;
			trigger.size = 2f*Geometry.GetExtents();
			trigger.center = Geometry.GetCenter();
		}
	}

	public VoxelCollider GetGeometry() {
		if(Geometry == null) {
			Geometry = GetComponent<VoxelCollider>();
		}
		return Geometry;
	}
	
	public Vector3 GetExtents() {
		return Vector3.Scale(transform.lossyScale.Positive(), GetGeometry().GetExtents());
	}

	public void AddContact() {
		Transform container = transform.Find("Contacts");
		if(container == null) {
			container = new GameObject("Contacts").transform;
			container.SetParent(transform);
			container.localPosition = Vector3.zero;
			container.localRotation = Quaternion.identity;
			container.localScale = Vector3.one;
		}
		Transform contact = new GameObject("Contact").transform;
		contact.SetParent(container);
		contact.transform.localPosition = Vector3.zero;
		contact.transform.localRotation = Quaternion.identity;
		contact.transform.localScale = Vector3.one;
		contact.gameObject.layer = gameObject.layer;
		ArrayExtensions.Add(ref Contacts, contact);
	}

	public void RemoveContact() {
		if(Contacts.Last() != null) {
			Utility.Destroy(Contacts.Last().gameObject);
		}
		ArrayExtensions.Shrink(ref Contacts);
	}

	public bool ContainsContact(string bone) {
		return GetContactTransform(bone) != null;
	}

	public Transform GetContactTransform(string bone) {
		return System.Array.Find(Contacts, x => x != null && x.name == bone);
	}

	public Matrix4x4 GetContact(string bone) {
		return GetContact(bone, transform.GetWorldMatrix());
	}

	public Matrix4x4 GetContact(string bone, Matrix4x4 root) {
		Transform t = GetContactTransform(bone);
		if(t == null) {
			return root;
		} else {
			Vector3 position = root.GetPosition() + root.GetRotation() * Vector3.Scale(root.GetScale().Positive(), t.localPosition);
			Quaternion rotation = root.GetRotation() * t.localRotation;
			return Matrix4x4.TRS(position, rotation, Vector3.one);
		}
	}

	public Matrix4x4 GetOrigin() {
		return transform.GetWorldMatrix();
	}

	public Matrix4x4 GetOrigin(Matrix4x4 root) {
		return root;
	}

	public Matrix4x4 GetCenter() {
		return GetCenter(transform.GetWorldMatrix());
	}

	public Matrix4x4 GetCenter(Matrix4x4 root) {
		Vector3 position = root.GetPosition() + root.GetRotation() * Vector3.Scale(root.GetScale(), GetGeometry().GetCenter());
		Quaternion rotation = root.GetRotation();
		return Matrix4x4.TRS(position, rotation, Vector3.one);
	}

	void OnRenderObject() {
		if(ShowContacts) {
			UltiDraw.Begin();
			Color[] colors = UltiDraw.GetRainbowColors(Contacts.Length);
			for(int i=0; i<Contacts.Length; i++) {
				if(Contacts[i] != null) {
					UltiDraw.DrawSphere(Contacts[i].position, Contacts[i].rotation, 0.05f, colors[i]);
				}
			}
			UltiDraw.End();
		}
	}

	void OnDrawGizmos() {
		if(!Application.isPlaying) {
			OnRenderObject();
		}
	}
	
	#if UNITY_EDITOR

	public Matrix4x4 GetContact(string bone, Frame frame, bool mirrored) {
		return GetComponent<Transformation>() != null ? GetContact(bone, GetComponent<Transformation>().GetTransformation(frame, mirrored)) : GetContact(bone);
	}

	public Matrix4x4 GetOrigin(Frame frame, bool mirrored) {
		return GetComponent<Transformation>() != null ? GetOrigin(GetComponent<Transformation>().GetTransformation(frame, mirrored)) : GetOrigin();
	}

	public Matrix4x4 GetCenter(Frame frame, bool mirrored) {
		return GetComponent<Transformation>() != null ? GetCenter(GetComponent<Transformation>().GetTransformation(frame, mirrored)) : GetCenter();
	}

	[CustomEditor(typeof(Interaction))]
	public class Interaction_Editor : Editor {

		public Interaction Target;

		void Awake() {
			Target = (Interaction)target;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();
				EditorGUILayout.HelpBox("Contacts", MessageType.None);
				Target.ShowContacts = EditorGUILayout.Toggle("Show Contacts", Target.ShowContacts);			
				for(int i=0; i<Target.Contacts.Length; i++) {
					Target.Contacts[i] = (Transform)EditorGUILayout.ObjectField(Target.Contacts[i], typeof(Transform), true);
				}
				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Add", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.AddContact();
				}
				if(Utility.GUIButton("Remove", UltiDraw.DarkGrey, UltiDraw.White)) {
					Target.RemoveContact();
				}
				EditorGUILayout.EndHorizontal();
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}
        
	}
	#endif

}
