using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class UI_Demo : MonoBehaviour {

    public bool Visualise = true;

    public SIGGRAPH_Asia_2019 Animation;

    private ColorBlock Active;
    private ColorBlock Inactive;

    void Awake() {
        Transform buttons = transform.Find("Buttons");
        for(int i=0; i<buttons.childCount; i++) {
            Button button = buttons.GetChild(i).GetComponent<Button>();
            button.transform.Find("Text").GetComponent<Text>().text = button.name;
            button.transform.localPosition += new Vector3(0f, i*50f, 0f);
            
            Active = button.colors;
            Active.normalColor = UltiDraw.Gold;
            Active.pressedColor = UltiDraw.Gold;
            Active.disabledColor = UltiDraw.Gold;
            Active.highlightedColor = UltiDraw.Gold;
            Inactive = button.colors;
            Inactive.normalColor = UltiDraw.DarkGrey;
            Inactive.pressedColor = UltiDraw.DarkGrey;
            Inactive.disabledColor = UltiDraw.DarkGrey;
            Inactive.highlightedColor = UltiDraw.DarkGrey;

            button.colors = Inactive;
            switch(button.name) {
                case "Experts":
                button.onClick.AddListener(() => {
                    Animation.GetComponent<ExpertActivation>().Visualise =! Animation.GetComponent<ExpertActivation>().Visualise;
                    button.colors = Animation.GetComponent<ExpertActivation>().Visualise ? Active : Inactive;
                });
                button.colors = Animation.GetComponent<ExpertActivation>().Visualise ? Active : Inactive;
                break;
                case "BiDirectional":
                button.onClick.AddListener(() => {
                    Animation.ShowBiDirectional = !Animation.ShowBiDirectional;
                    button.colors =  Animation.ShowBiDirectional ? Active : Inactive;
                });
                button.colors =  Animation.ShowBiDirectional ? Active : Inactive;
                break;
                case "Root":
                button.onClick.AddListener(() => {
                    Animation.ShowRoot = !Animation.ShowRoot;
                    button.colors = Animation.ShowRoot ? Active : Inactive;
                });
                button.colors = Animation.ShowRoot ? Active : Inactive;
                break;
                case "Goal":
                button.onClick.AddListener(() => {
                    Animation.ShowGoal = !Animation.ShowGoal;
                    button.colors = Animation.ShowGoal ? Active : Inactive;
                });
                button.colors = Animation.ShowGoal ? Active : Inactive;
                break;
                case "Current":
                button.onClick.AddListener(() => {
                    Animation.ShowCurrent = !Animation.ShowCurrent;
                    button.colors = Animation.ShowCurrent ? Active : Inactive;
                });
                button.colors = Animation.ShowCurrent ? Active : Inactive;
                break;
                case "Phase":
                button.onClick.AddListener(() => {
                    Animation.ShowPhase = !Animation.ShowPhase;
                    button.colors = Animation.ShowPhase ? Active : Inactive;
                });
                button.colors = Animation.ShowPhase ? Active : Inactive;
                break;
                case "Contacts":
                button.onClick.AddListener(() => {
                    Animation.ShowContacts = !Animation.ShowContacts;
                    button.colors = Animation.ShowContacts ? Active : Inactive;
                });
                button.colors = Animation.ShowContacts ? Active : Inactive;
                break;
                case "Environment":
                button.onClick.AddListener(() => {
                    Animation.ShowEnvironment = !Animation.ShowEnvironment;
                    button.colors = Animation.ShowEnvironment ? Active : Inactive;
                });
                button.colors = Animation.ShowEnvironment ? Active : Inactive;
                break;
                case "Interaction":
                button.onClick.AddListener(() => {
                    Animation.ShowInteraction = !Animation.ShowInteraction;
                    button.colors = Animation.ShowInteraction ? Active : Inactive;
                });
                button.colors = Animation.ShowInteraction ? Active : Inactive;
                break;
                case "Skeleton":
                button.onClick.AddListener(() => {
                    Animation.Actor.DrawSkeleton = !Animation.Actor.DrawSkeleton;
                    button.colors = Animation.Actor.DrawSkeleton ? Active : Inactive;
                });
                button.colors = Animation.Actor.DrawSkeleton ? Active : Inactive;
                break;
            }
        }
    }

    void Update() {
        if(Input.GetKeyDown(KeyCode.F1)) {
            Visualise = !Visualise;
            for(int i=0; i<transform.childCount; i++) {
                if(transform.GetChild(i).name != "Icons") {
                    transform.GetChild(i).gameObject.SetActive(Visualise);
                }
            }
        }

        foreach(UI_Action item in GetComponentsInChildren<UI_Action>()) {
            item.Process(this);
        }
    }
}
