using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class GridDrawer : MonoBehaviour {
    public float Offset = 0f;
    public int Cells = 10;
    public Vector2 Scale = Vector2.one;
    public Color Color = Color.white;
    void OnRenderObject() {
        UltiDraw.Begin();
        UltiDraw.SetDepthRendering(true);
        UltiDraw.DrawGrid(transform.position + new Vector3(0f, Offset, 0f), transform.rotation, Cells, Cells, Scale.x/Cells, Scale.y/Cells, Color);
        UltiDraw.SetDepthRendering(false);
        UltiDraw.End();
    }
}
