using UnityEngine;

[ExecuteInEditMode]
public class RenderColliders : MonoBehaviour {

    public bool Draw = true;
    public Color Color = Color.black;
    public bool Depth = false;
    public float Curvature = 0.25f;
    public float Filling = 0f;

    void OnRenderObject() {
        if(!Draw) {
            return;
        }
        UltiDraw.Begin();
        UltiDraw.SetDepthRendering(Depth);
        UltiDraw.SetCurvature(Curvature);
        UltiDraw.SetFilling(Filling);
        foreach(Collider collider in gameObject.GetComponentsInChildren<Collider>()) {
            UltiDraw.DrawCollider(collider, Color);
        }
        UltiDraw.SetDepthRendering(false);
        UltiDraw.SetCurvature(0.25f);
        UltiDraw.SetFilling(0f);
        UltiDraw.End();
	}

}
