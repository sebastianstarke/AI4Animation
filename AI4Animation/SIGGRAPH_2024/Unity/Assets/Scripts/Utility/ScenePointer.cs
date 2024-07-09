using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScenePointer : MonoBehaviour {
    public LayerMask SelectionMask = ~0;

    public bool Draw = true;

    void Update() {
        RaycastHit hit = Utility.GetMouseSensor(SelectionMask);
        if(Input.GetMouseButtonDown(0) && hit.collider != null) {
            Debug.Log("Hit Transform: " + hit.transform.name + " Collider: " + hit.collider.name);
        }
    }

    void OnRenderObject() {
        if(!Draw) {return;}
        UltiDraw.Begin();
        RaycastHit hit = Utility.GetMouseSensor(SelectionMask);
        UltiDraw.DrawSphere(hit.point, Quaternion.identity, 0.1f, UltiDraw.Black);
        UltiDraw.DrawCollider(hit.collider, Input.GetMouseButton(0) ? UltiDraw.Red : UltiDraw.White);
        UltiDraw.End();
    }
}