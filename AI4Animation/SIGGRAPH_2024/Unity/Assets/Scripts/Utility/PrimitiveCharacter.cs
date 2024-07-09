using System.Collections.Generic;
using AI4Animation;
using UnityEngine;
using UnityEditor;

public class PrimitiveCharacter : MonoBehaviour {

    public enum MODE{Collider, Cube, Sphere, Capsule}

    public Material Material;

    public Actor Actor = null;

    public float Scale = 0.5f;
    public float Power = 0.5f;

    public MODE Mode = MODE.Capsule;

    public List<Transform> Primitives = new List<Transform>();

    [ContextMenu("Delete Primitives")]
    public void Delete() {
        foreach(Transform t in Primitives) {
            if(t != null){
                Utility.Destroy(t.gameObject);
            }
        }
        Primitives = new List<Transform>();
    }

    [ContextMenu("Process")]
    public void Process() {
        Delete();
        switch(Mode) {
            case MODE.Collider:
            RecursionCollider(transform);
            break;
            case MODE.Cube:
            RecursionCube(transform);
            break;
            case MODE.Sphere:
            RecursionSphere(transform);
            break;
            case MODE.Capsule:
            if(Actor != null) {
                RecursionCapsule(Actor.Bones.First());
            } else {
                RecursionCapsule(transform);
            }
            break;
        }
        ApplyMaterial();
    }

    [ContextMenu("Apply Material")]
    public void ApplyMaterial() {
        foreach(Transform t in Primitives) {
            if(Material != null) {
                t.GetComponent<MeshRenderer>().sharedMaterial = Material;
            }
        }
    }

    [ContextMenu("Show")]
    public void Show() {
        foreach(Transform t in Primitives) {
            if(t != null){
                t.GetComponent<MeshRenderer>().enabled = true;
            }
        }
    }

    [ContextMenu("Hide")]
    public void Hide() {
        foreach(Transform t in Primitives) {
            if(t != null){
                t.GetComponent<MeshRenderer>().enabled = false;
            }
        }
    }

    private void RecursionCollider(Transform t) {
        Collider c = t.GetComponent<Collider>();
        Transform primitive = null;
        if(c != null) {
            if(c is SphereCollider) {
                primitive = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
                SphereCollider collider = (SphereCollider)c;
                primitive.SetParent(t);
                primitive.localPosition = collider.center;
                primitive.localRotation = Quaternion.identity;
                primitive.localScale = 2f*collider.radius*Vector3.one;
                GameObject.DestroyImmediate(primitive.GetComponent<Collider>());
                Primitives.Add(primitive);
            }
            if(c is CapsuleCollider) {
                primitive = GameObject.CreatePrimitive(PrimitiveType.Capsule).transform;
                CapsuleCollider collider = (CapsuleCollider)c;
                primitive.SetParent(t);
                primitive.gameObject.layer = t.gameObject.layer;
                primitive.localPosition = collider.center;
                primitive.localRotation = Quaternion.FromToRotation(Vector3.up, collider.GetAxis());
                primitive.localScale = new Vector3(2f*collider.radius, 0.5f*collider.height, 2f*collider.radius);
                GameObject.DestroyImmediate(primitive.GetComponent<Collider>());
                Primitives.Add(primitive);
            }
            if(c is BoxCollider) {
                primitive = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
                BoxCollider collider = (BoxCollider)c;
                primitive.SetParent(t);
                primitive.localPosition = collider.center;
                primitive.localRotation = Quaternion.identity;
                primitive.localScale = collider.size;
                GameObject.DestroyImmediate(primitive.GetComponent<Collider>());
                Primitives.Add(primitive);
            }
        }
        for(int i=0; i<t.childCount; i++) {
            if(t.GetChild(i) != primitive) {
                RecursionCollider(t.GetChild(i));
            }
        }
    }

    private void RecursionCube(Transform t) {
        Transform primitive =  GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
        primitive.SetParent(t);
        primitive.gameObject.layer = t.gameObject.layer;
        primitive.localPosition = Vector3.zero;
        primitive.localRotation = Quaternion.identity;
        float length = t == transform ? GetAverageLengthToChildren(t) : GetLengthBetween(t.parent, t);
        primitive.localScale = 0.5f*length*Vector3.one;
        Primitives.Add(primitive);
        for(int i=0; i<t.childCount; i++) {
            if(t.GetChild(i) != primitive) {
                RecursionCube(t.GetChild(i));
            }
        }
    }

    private void RecursionSphere(Transform t) {
        Transform primitive =  GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
        primitive.SetParent(t);
        primitive.gameObject.layer = t.gameObject.layer;
        primitive.localPosition = Vector3.zero;
        primitive.localRotation = Quaternion.identity;
        float length = t == transform ? GetAverageLengthToChildren(t) : GetLengthBetween(t.parent, t);
        primitive.localScale = 0.5f*length*Vector3.one;
        Primitives.Add(primitive);
        for(int i=0; i<t.childCount; i++) {
            if(t.GetChild(i) != primitive) {
                RecursionSphere(t.GetChild(i));
            }
        }
    }

    private void RecursionCapsule(Transform t) {
        Transform[] childs = t.GetChilds();
        foreach(Transform c in childs) {
            Transform primitive =  GameObject.CreatePrimitive(PrimitiveType.Capsule).transform;
            primitive.SetParent(t);
            primitive.gameObject.layer = t.gameObject.layer;
            primitive.localPosition = 0.5f * (c.position - t.position).DirectionTo(t.rotation);
            primitive.localRotation = Quaternion.LookRotation((c.position - t.position).normalized).RotationTo(t.rotation) * Quaternion.Euler(90f, 0f, 0f);
            // primitive.localRotation = Quaternion.Euler(new Vector3(90f, 90f, 0f));
            float length = c.localPosition.magnitude;
            length = Mathf.Pow(length * Scale, Power);
            primitive.localScale = new Vector3(0.5f, 0.5f, 0.5f) * length;
            Primitives.Add(primitive);
        }
        foreach(Transform c in childs) {
            RecursionCapsule(c);
        }
    }

    private void RecursionCapsule(Actor.Bone bone) {
        foreach(Actor.Bone child in bone.GetChilds()) {
            Transform primitive =  GameObject.CreatePrimitive(PrimitiveType.Capsule).transform;
            primitive.SetParent(bone.GetTransform());
            primitive.gameObject.layer = bone.GetTransform().gameObject.layer;
            primitive.localPosition = 0.5f * (child.GetPosition() - bone.GetPosition()).DirectionTo(bone.GetRotation());
            primitive.localRotation = Quaternion.LookRotation((child.GetPosition() - bone.GetPosition()).normalized).RotationTo(bone.GetRotation()) * Quaternion.Euler(90f, 0f, 0f);
            // primitive.localRotation = Quaternion.Euler(new Vector3(90f, 90f, 0f));
            float length = Vector3.Distance(bone.GetPosition(), child.GetPosition());
            length = Mathf.Pow(length * Scale, Power);
            primitive.localScale = length * Vector3.one;
            Primitives.Add(primitive);
        }
        foreach(Actor.Bone c in bone.GetChilds()) {
            RecursionCapsule(c);
        }
    }

    private Vector3 GetAveragePositionToChildren(Transform t) {
        Vector3 position = t.position;
        for(int i=0; i<t.childCount; i++) {
            position += t.GetChild(i).position;
        }
        return position /= 1+t.childCount;
    }

    private float GetLengthBetween(Transform a, Transform b) {
        return Vector3.Distance(a.position, b.position);
    }

    private float GetAverageLengthToChildren(Transform t) {
        float length = 0f;
        for(int i=0; i<t.childCount; i++) {
            length += GetLengthBetween(t, t.GetChild(i));
        }
        return length /= 1+t.childCount;
    }


	#if UNITY_EDITOR
	[CustomEditor(typeof(PrimitiveCharacter), true)]
	public class PrimitiveCharacterEditor : Editor {

        public override bool RequiresConstantRepaint() => true;

        public bool AutoUpdate = false;

		public override void OnInspectorGUI() {
            PrimitiveCharacter instance = (PrimitiveCharacter)target;
            AutoUpdate = EditorGUILayout.Toggle("Auto Update", AutoUpdate);
			DrawDefaultInspector();
            
            if(AutoUpdate) {
                instance.Process();
            }
		}

	}
	#endif    
}
