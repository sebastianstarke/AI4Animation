using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class UI_Action : MonoBehaviour {
    
    private Image Image;

    private Color Active;
    private Color Inactive;

    void Awake() {
        Image = transform.Find("Gold").GetComponent<Image>();
        Active = Image.color;
        Inactive = UltiDraw.DarkGrey.Transparent(0.25f);
    }

    public void Process(UI_Demo demo) {
        Image.color = demo.Animation.GetController().GetSignal(name).Query() ? Active : Inactive;
    }
}
