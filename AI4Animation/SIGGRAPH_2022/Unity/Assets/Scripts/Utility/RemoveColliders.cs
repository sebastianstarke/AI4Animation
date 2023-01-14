using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RemoveColliders : MonoBehaviour {

    // public string ObjectsWithName = string.Empty;

    [ContextMenu("Process")]
    public void Process() {
        Recursion(transform);
    }

    private void Recursion(Transform t) {
        // if(t.name == ObjectsWithName) {
            foreach(Collider c in t.GetComponents<Collider>()) {
                Utility.Destroy(c);
            }
        // }
        for(int i=0; i<t.childCount; i++) {
            Recursion(t.GetChild(i));
        }
    }

}
