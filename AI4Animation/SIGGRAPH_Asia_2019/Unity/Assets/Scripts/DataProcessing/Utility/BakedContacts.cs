#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class BakedContacts : MonoBehaviour {

    public Data[] Contacts = new Data[0];

    public void Setup(string[] bones, int frames) {
        Contacts = new Data[bones.Length];
        for(int i=0; i<Contacts.Length; i++) {
            Contacts[i] = new Data();
            Contacts[i].Bone = bones[i];
            Contacts[i].RegularPoints = new Vector3[frames];
            Contacts[i].InversePoints = new Vector3[frames];
        }
    }

    public void BakeContact(string bone, Vector3 point, Frame frame, bool mirrored) {
        int idx = System.Array.FindIndex(Contacts, x => x.Bone == bone);
        if(idx != -1) {
            BakeContact(idx, point, frame, mirrored);
        }
    }

    public void BakeContact(int bone, Vector3 point, Frame frame, bool mirrored) {
        Matrix4x4 m = Matrix4x4.identity;
        if(GetComponent<Transformation>() == null) {
            m = mirrored ? transform.GetWorldMatrix().GetMirror(frame.Data.MirrorAxis) : transform.GetWorldMatrix();
        } else {
            m = GetComponent<Transformation>().GetTransformation(frame, mirrored);
        }
        if(mirrored) {
            Contacts[bone].InversePoints[frame.Index-1] = point.GetRelativePositionTo(m);
        } else {
            Contacts[bone].RegularPoints[frame.Index-1] = point.GetRelativePositionTo(m);
        }
        EditorUtility.SetDirty(this);
    }

    public Vector3 GetContactPoint(string bone, Frame frame, bool mirrored) {
        int idx = System.Array.FindIndex(Contacts, x => x.Bone == bone);
        if(idx == -1) {
            return Vector3.zero;
        }
        return GetContactPoint(idx, frame, mirrored);
    }

    private Vector3 GetContactPoint(int bone, Frame frame, bool mirrored) {
        return (mirrored ? Contacts[bone].InversePoints[frame.Index-1].GetMirror(frame.Data.MirrorAxis) : Contacts[bone].RegularPoints[frame.Index-1]).GetRelativePositionFrom(transform.GetWorldMatrix());
    }

    [System.Serializable]
	public class Data {
		public string Bone = string.Empty;
		public Vector3[] RegularPoints = new Vector3[0];
		public Vector3[] InversePoints = new Vector3[0];
	}

    /*
	[CustomEditor(typeof(BakedContacts))]
	public class BakedContacts_Editor : Editor {

		public BakedContacts Target;

		void Awake() {
			Target = (BakedContacts)target;
		}

		public override void OnInspectorGUI() {
            Undo.RecordObject(Target, Target.name);
            Target.Active = EditorGUILayout.Toggle("Active", Target.Active);
            Target.ShowContacts = EditorGUILayout.Toggle("Show Contacts", Target.ShowContacts);
            Target.Size = EditorGUILayout.FloatField("Size", Target.Size);
            EditorGUILayout.HelpBox("Count: " + Target.Count, MessageType.None);
            if(GUI.changed) {
                EditorUtility.SetDirty(Target);
            }
		}
        
	}
    */

}
#endif