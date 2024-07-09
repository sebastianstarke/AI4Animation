using System;
using AI4Animation;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace SIGGRAPH_2024 {
    public class NMM_Menu_UI : MonoBehaviour {
        public Button Demo1, Demo2;

        void Start() {
            Demo1.onClick.AddListener(LoadDemo1);
            Demo2.onClick.AddListener(LoadDemo2);
        }

        void Update(){
            if(Input.GetKey(KeyCode.Escape)) {
                Application.Quit();
            }
        }
        public void LoadDemo1() {  
            Debug.Log("Clicked");
            SceneManager.LoadScene(1);  
        }
        public void LoadDemo2() {  
            SceneManager.LoadScene(2);  
        } 
    }
}