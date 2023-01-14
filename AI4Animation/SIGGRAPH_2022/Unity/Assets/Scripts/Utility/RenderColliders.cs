using UnityEngine;

[ExecuteInEditMode]
public class RenderColliders : MonoBehaviour {

    void OnRenderObject() {
        UltiDraw.Begin();
        foreach(Collider collider in gameObject.GetComponentsInChildren<Collider>()) {
            UltiDraw.DrawCollider(collider, Color.black);
        }
        UltiDraw.End();
	}

}
