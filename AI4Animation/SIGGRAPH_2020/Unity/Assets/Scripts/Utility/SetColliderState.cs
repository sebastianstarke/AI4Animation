using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SetColliderState : MonoBehaviour {
    public enum STATE{Collider, Trigger};
    public STATE State = STATE.Collider;
    [ContextMenu("Process")]
    public void Process() {
        foreach(Collider c in GetComponentsInChildren<Collider>()) {
            c.isTrigger = State == STATE.Trigger;
        }
    } 
}
