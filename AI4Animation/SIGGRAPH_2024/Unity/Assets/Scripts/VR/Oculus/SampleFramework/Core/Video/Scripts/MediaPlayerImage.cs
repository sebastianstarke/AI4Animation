// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class MediaPlayerImage : Image
{
    public enum ButtonType
    {
        Play,
        Pause,
        FastForward,
        Rewind,
        SkipForward,
        SkipBack,
        Stop
    }

    [SerializeField]
    private ButtonType m_ButtonType;
    public ButtonType buttonType
    {
        get
        {
            return m_ButtonType;
        }
        set
        {
            if (m_ButtonType != value)
            {
                m_ButtonType = value;
                SetAllDirty();
            }
        }
    }

    protected override void OnPopulateMesh(VertexHelper toFill)
    {
        var r = GetPixelAdjustedRect();
        var v = new Vector4(r.x, r.y, r.x + r.width, r.y + r.height);

        Color32 color32 = color;
        toFill.Clear();

        switch(m_ButtonType)
        {
            case ButtonType.Play:
                {
                    toFill.AddVert(new Vector3(v.x, v.y), color32, new Vector2(0f, 0f));
                    toFill.AddVert(new Vector3(v.x, v.w), color32, new Vector2(0f, 1f));
                    toFill.AddVert(new Vector3(v.z, Mathf.Lerp(v.y, v.w, 0.5f)), color32, new Vector2(1f, 0.5f));
                    toFill.AddTriangle(0, 1, 2);
                }
                break;

            case ButtonType.Pause:
                {
                    const float PAUSE_BAR_WIDTH = 0.35f;
                    toFill.AddVert(new Vector3(v.x, v.y), color32, new Vector2(0f, 0f));
                    toFill.AddVert(new Vector3(v.x, v.w), color32, new Vector2(0f, 1f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, PAUSE_BAR_WIDTH), v.w), color32, new Vector2(PAUSE_BAR_WIDTH, 1f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, PAUSE_BAR_WIDTH), v.y), color32, new Vector2(PAUSE_BAR_WIDTH, 0f));

                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 1 - PAUSE_BAR_WIDTH), v.y), color32, new Vector2(1 - PAUSE_BAR_WIDTH, 0f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 1 - PAUSE_BAR_WIDTH), v.w), color32, new Vector2(1 - PAUSE_BAR_WIDTH, 1f));
                    toFill.AddVert(new Vector3(v.z, v.w), color32, new Vector2(1f, 1f));
                    toFill.AddVert(new Vector3(v.z, v.y), color32, new Vector2(1f, 0f));

                    toFill.AddTriangle(0, 1, 2);
                    toFill.AddTriangle(2, 3, 0);
                    toFill.AddTriangle(4, 5, 6);
                    toFill.AddTriangle(6, 7, 4);
                }
                break;
            case ButtonType.FastForward:
                {
                    toFill.AddVert(new Vector3(v.x, v.y), color32, new Vector2(0f, 0f));
                    toFill.AddVert(new Vector3(v.x, v.w), color32, new Vector2(0f, 1f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f), Mathf.Lerp(v.y, v.w, 0.5f)), color32, new Vector2(0.5f, 0.5f));

                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f), v.y), color32, new Vector2(0.5f, 0f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f), v.w), color32, new Vector2(0.5f, 1f));
                    toFill.AddVert(new Vector3(v.z, Mathf.Lerp(v.y, v.w, 0.5f)), color32, new Vector2(1f, 0.5f));
                    toFill.AddTriangle(0, 1, 2);
                    toFill.AddTriangle(3, 4, 5);
                }
                break;
            case ButtonType.Rewind:
                {
                    toFill.AddVert(new Vector3(v.x, Mathf.Lerp(v.y, v.w, 0.5f)), color32, new Vector2(0f, 0.5f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f), v.w), color32, new Vector2(0.5f, 1f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f), v.y), color32, new Vector2(0.5f, 0f));

                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f), Mathf.Lerp(v.y, v.w, 0.5f)), color32, new Vector2(0.5f, 0.5f));
                    toFill.AddVert(new Vector3(v.z, v.w), color32, new Vector2(1f, 1f));
                    toFill.AddVert(new Vector3(v.z, v.y), color32, new Vector2(1f, 0f));
                    toFill.AddTriangle(0, 1, 2);
                    toFill.AddTriangle(3, 4, 5);
                }
                break;
            case ButtonType.SkipForward:
                {
                    const float SKIP_FORWARD_BAR_WIDTH = 0.125f;

                    toFill.AddVert(new Vector3(v.x, v.y), color32, new Vector2(0f, 0f));
                    toFill.AddVert(new Vector3(v.x, v.w), color32, new Vector2(0f, 1f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f - SKIP_FORWARD_BAR_WIDTH / 2), Mathf.Lerp(v.y, v.w, 0.5f)), color32, new Vector2(0.5f - SKIP_FORWARD_BAR_WIDTH / 2, 0.5f));

                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f - SKIP_FORWARD_BAR_WIDTH / 2), v.y), color32, new Vector2(0.5f - SKIP_FORWARD_BAR_WIDTH / 2, 0f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f - SKIP_FORWARD_BAR_WIDTH / 2), v.w), color32, new Vector2(0.5f - SKIP_FORWARD_BAR_WIDTH / 2, 1f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 1 - SKIP_FORWARD_BAR_WIDTH), Mathf.Lerp(v.y, v.w, 0.5f)), color32, new Vector2(1f - SKIP_FORWARD_BAR_WIDTH, 0.5f));

                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 1 - SKIP_FORWARD_BAR_WIDTH), v.y), color32, new Vector2(1 - SKIP_FORWARD_BAR_WIDTH, 0f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 1 - SKIP_FORWARD_BAR_WIDTH), v.w), color32, new Vector2(1 - SKIP_FORWARD_BAR_WIDTH, 1f));
                    toFill.AddVert(new Vector3(v.z, v.w), color32, new Vector2(1f, 1f));
                    toFill.AddVert(new Vector3(v.z, v.y), color32, new Vector2(1f, 0f));

                    toFill.AddTriangle(0, 1, 2);
                    toFill.AddTriangle(3, 4, 5);
                    toFill.AddTriangle(6, 7, 8);
                    toFill.AddTriangle(8, 9, 6);
                }
                break;
            case ButtonType.SkipBack:
                {
                    const float SKIP_BACK_BAR_WIDTH = 0.125f;
                    toFill.AddVert(new Vector3(v.x, v.y), color32, new Vector2(0f, 0f));
                    toFill.AddVert(new Vector3(v.x, v.w), color32, new Vector2(0f, 1f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, SKIP_BACK_BAR_WIDTH), v.w), color32, new Vector2(SKIP_BACK_BAR_WIDTH, 1f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, SKIP_BACK_BAR_WIDTH), v.y), color32, new Vector2(SKIP_BACK_BAR_WIDTH, 0f));

                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, SKIP_BACK_BAR_WIDTH), Mathf.Lerp(v.y, v.w, 0.5f)), color32, new Vector2(SKIP_BACK_BAR_WIDTH, 0.5f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f + SKIP_BACK_BAR_WIDTH / 2), v.w), color32, new Vector2(0.5f + SKIP_BACK_BAR_WIDTH / 2, 1f));
                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f + SKIP_BACK_BAR_WIDTH / 2), v.y), color32, new Vector2(0.5f + SKIP_BACK_BAR_WIDTH / 2, 0f));

                    toFill.AddVert(new Vector3(Mathf.Lerp(v.x, v.z, 0.5f + SKIP_BACK_BAR_WIDTH / 2), Mathf.Lerp(v.y, v.w, 0.5f)), color32, new Vector2(0.5f + SKIP_BACK_BAR_WIDTH / 2, 0.5f));
                    toFill.AddVert(new Vector3(v.z, v.w), color32, new Vector2(1f, 1f));
                    toFill.AddVert(new Vector3(v.z, v.y), color32, new Vector2(1f, 0f));

                    toFill.AddTriangle(0, 1, 2);
                    toFill.AddTriangle(2, 3, 0);
                    toFill.AddTriangle(4, 5, 6);
                    toFill.AddTriangle(7, 8, 9);
                }
                break;
            case ButtonType.Stop:
            default: // by default draw a stop symbol (which just so happens to be square)
                {
                    toFill.AddVert(new Vector3(v.x, v.y), color32, new Vector2(0f, 0f));
                    toFill.AddVert(new Vector3(v.x, v.w), color32, new Vector2(0f, 1f));
                    toFill.AddVert(new Vector3(v.z, v.w), color32, new Vector2(1f, 1f));
                    toFill.AddVert(new Vector3(v.z, v.y), color32, new Vector2(1f, 0f));

                    toFill.AddTriangle(0, 1, 2);
                    toFill.AddTriangle(2, 3, 0);
                }
                break;
        }
    }
}
