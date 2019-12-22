using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace VoxelSystem.Demo
{

    [RequireComponent (typeof(MeshRenderer))]
    public class PatternFloor : MonoBehaviour {

        new Renderer renderer;
        MaterialPropertyBlock block;

        void Start () {
            block = new MaterialPropertyBlock();
            renderer = GetComponent<Renderer>();
            renderer.GetPropertyBlock(block);
            block.SetTexture("_MainTex", CreatePattern(1 << 7));
            renderer.SetPropertyBlock(block);
        }

        Texture2D CreatePattern(int size, int division = 4, int marker = 16)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGB24, true);

            var unit = size / division;
            float
                markerMin = size / marker * 0.5f,
                markerMax = unit - markerMin;

            Color 
                white = Color.white, 
                black = Color.black;

            for(int y = 0; y < size; y++)
            {
                var ry = y % unit;
                var dy = Mathf.Abs(unit - ry);
                var flagY = (dy <= markerMin || dy >= markerMax);

                for(int x = 0; x < size; x++)
                {
                    var rx = x % unit;
                    var dx = Mathf.Abs(unit - rx);
                    var flagX = (dx <= markerMin || dx >= markerMax);

                    if(
                        (ry == 0 || rx == 0) && (flagY && flagX)
                    ) {
                        tex.SetPixel(x, y, black);
                    } else
                    {
                        tex.SetPixel(x, y, white);
                    }
                }
            }
            tex.Apply();
            return tex;
        }

    }

}

