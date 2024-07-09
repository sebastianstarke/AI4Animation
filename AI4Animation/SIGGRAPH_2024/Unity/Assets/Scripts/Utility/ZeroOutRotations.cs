using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ZeroOutRotations : MonoBehaviour {
    [ContextMenu("Process")]
    public void Process() {
        Iterate(transform);
    }
    private void Iterate(Transform t) {
        t.localRotation = Quaternion.identity;
        for(int i=0; i<t.childCount; i++) {
            Iterate(t.GetChild(i));
        }
    }
}
