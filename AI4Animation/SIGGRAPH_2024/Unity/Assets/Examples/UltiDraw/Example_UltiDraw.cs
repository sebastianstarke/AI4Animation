using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Example_UltiDraw : MonoBehaviour {
    void OnDrawGizmos() {
        if(!Application.isPlaying) {
            OnRenderObject();
        }
    }
    void OnRenderObject() {
        UltiDraw.DrawCube(Vector3.zero, Quaternion.identity, 1f, UltiDraw.Gold);
        UltiDraw.DrawWireSphere(Vector3.zero, Quaternion.identity, 1f, UltiDraw.Gold);
    }
}
