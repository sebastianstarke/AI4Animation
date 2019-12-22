using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PrimitiveCharacter : MonoBehaviour {

    public enum MODE{Collider, Cube, Sphere}

    public Material Material;

    public MODE Mode = MODE.Collider;

    [ContextMenu("Process")]
    public void Process() {
        List<Primitive> primitives = new List<Primitive>();
        switch(Mode) {
            case MODE.Collider:
            RecursionCollider(transform, primitives);
            break;
            case MODE.Cube:
            RecursionCube(transform, primitives);
            break;
            case MODE.Sphere:
            RecursionSphere(transform, primitives);
            break;
        }
        foreach(Primitive p in primitives) {
            p.Instance.SetParent(p.Reference);
            if(Material != null) {
                p.Instance.GetComponent<MeshRenderer>().sharedMaterial = Material;
            }
        }
    }

    private void RecursionCollider(Transform t, List<Primitive> primitives) {
        Collider c = t.GetComponent<Collider>();
        if(c != null) {
            if(c is SphereCollider) {
                Primitive primitive = new Primitive();
                primitive.Instance =  GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
                primitive.Reference = t;
                SphereCollider collider = (SphereCollider)c;
                primitive.Instance.SetParent(primitive.Reference);
                primitive.Instance.localPosition = collider.center;
                primitive.Instance.localRotation = Quaternion.identity;
                primitive.Instance.localScale = 2f*collider.radius*Vector3.one;
                primitive.Instance.SetParent(null);
                GameObject.DestroyImmediate(collider);
                primitives.Add(primitive);
            }
            if(c is CapsuleCollider) {
                Primitive primitive = new Primitive();
                primitive.Instance =  GameObject.CreatePrimitive(PrimitiveType.Capsule).transform;
                primitive.Reference = t;
                CapsuleCollider collider = (CapsuleCollider)c;
                primitive.Instance.SetParent(primitive.Reference);
                primitive.Instance.gameObject.layer = primitive.Reference.gameObject.layer;
                primitive.Instance.localPosition = collider.center;
                primitive.Instance.localRotation = Quaternion.FromToRotation(Vector3.up, DirectionToAxis(collider.direction));
                primitive.Instance.localScale = new Vector3(2f*collider.radius, 0.5f*collider.height, 2f*collider.radius);
                primitive.Instance.SetParent(null);
                GameObject.DestroyImmediate(collider);
                primitives.Add(primitive);
            }
        }
        for(int i=0; i<t.childCount; i++) {
            RecursionCollider(t.GetChild(i), primitives);
        }
    }

    private void RecursionCube(Transform t, List<Primitive> primitives) {
        Primitive primitive = new Primitive();
        primitive.Instance =  GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
        primitive.Reference = t;
        primitive.Instance.SetParent(primitive.Reference);
        primitive.Instance.gameObject.layer = primitive.Reference.gameObject.layer;
        primitive.Instance.localPosition = Vector3.zero;
        primitive.Instance.localRotation = Quaternion.identity;
        float length = t == transform ? GetAverageLengthToChildren(t) : GetLengthBetween(t.parent, t);
        primitive.Instance.localScale = 0.5f*length*Vector3.one;
        primitive.Instance.SetParent(null);
        primitives.Add(primitive);
        for(int i=0; i<t.childCount; i++) {
            RecursionCube(t.GetChild(i), primitives);
        }
    }

    private void RecursionSphere(Transform t, List<Primitive> primitives) {
        Primitive primitive = new Primitive();
        primitive.Instance =  GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
        primitive.Reference = t;
        primitive.Instance.SetParent(primitive.Reference);
        primitive.Instance.gameObject.layer = primitive.Reference.gameObject.layer;
        primitive.Instance.localPosition = Vector3.zero;
        primitive.Instance.localRotation = Quaternion.identity;
        float length = t == transform ? GetAverageLengthToChildren(t) : GetLengthBetween(t.parent, t);
        primitive.Instance.localScale = 0.5f*length*Vector3.one;
        primitive.Instance.SetParent(null);
        primitives.Add(primitive);
        for(int i=0; i<t.childCount; i++) {
            RecursionSphere(t.GetChild(i), primitives);
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

    private Vector3 DirectionToAxis(int direction) {
        if(direction == 0) {
            return Vector3.right;
        }
        if(direction == 1) {
            return Vector3.up;
        }
        if(direction == 2) {
            return Vector3.forward;
        }
        return Vector3.forward;
    }

    private class Primitive {
        public Transform Instance;
        public Transform Reference;
    }

}
