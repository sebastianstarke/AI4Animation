using System.Collections.Generic;
using UnityEngine;

public class PrimitiveCharacter : MonoBehaviour {

    public enum MODE{Collider, Cube, Sphere}

    public Material Material;

    public MODE Mode = MODE.Collider;

    public List<Transform> Primitives = new List<Transform>();

    [ContextMenu("Process")]
    public void Process() {
        foreach(Transform t in Primitives) {
            GameObject.DestroyImmediate(t.gameObject);
        }
        Primitives = new List<Transform>();
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
        }
        foreach(Transform t in Primitives) {
            if(Material != null) {
                t.GetComponent<MeshRenderer>().sharedMaterial = Material;
            }
        }
    }

    [ContextMenu("Show")]
    public void Show() {
        foreach(Transform t in Primitives) {
            t.GetComponent<MeshRenderer>().enabled = true;
        }
    }

    [ContextMenu("Hide")]
    public void Hide() {
        foreach(Transform t in Primitives) {
            t.GetComponent<MeshRenderer>().enabled = false;
        }
    }

    private void RecursionCollider(Transform t) {
        Collider c = t.GetComponent<Collider>();
        if(c != null) {
            if(c is SphereCollider) {
                Transform primitive = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
                SphereCollider collider = (SphereCollider)c;
                primitive.SetParent(t);
                primitive.localPosition = collider.center;
                primitive.localRotation = Quaternion.identity;
                primitive.localScale = 2f*collider.radius*Vector3.one;
                GameObject.DestroyImmediate(primitive.GetComponent<Collider>());
                Primitives.Add(primitive);
            }
            if(c is CapsuleCollider) {
                Transform primitive = GameObject.CreatePrimitive(PrimitiveType.Capsule).transform;
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
                Transform primitive = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
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
            if(t.GetChild(i) != Primitives.Last()) {
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
            if(t.GetChild(i) != Primitives.Last()) {
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
            if(t.GetChild(i) != Primitives.Last()) {
                RecursionSphere(t.GetChild(i));
            }
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

}
