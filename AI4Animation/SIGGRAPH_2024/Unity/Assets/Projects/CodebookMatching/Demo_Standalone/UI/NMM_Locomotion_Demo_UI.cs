using System;
using AI4Animation;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace SIGGRAPH_2024 {
    public class NMM_Locomotion_Demo_UI : MonoBehaviour {
        public MotionController Controller;
        public BinaryButton Control, CurrentSequence, History, Codebook;
        public Slider Rollout, KNN;
        public Color InactiveColor = UltiDraw.White;
        public Color ActiveColor = UltiDraw.Magenta;
        public UltiDraw.GUIRect VelocityRect = new UltiDraw.GUIRect(0.5f, 0.5f, 0.5f, 0.5f);
        public float LabelOffset = 0f;

        [Serializable]
        public class BinaryButton {
            public bool State;
            public Button Button;
        }

        public void LoadMenu() {  
            SceneManager.LoadScene(0);  
        }

        void Start() {
            Codebook.Button.onClick.AddListener(() => ToggleState(Codebook, SetCodebookState));
            History.Button.onClick.AddListener(() => ToggleState(History, SetHistoryState));
            CurrentSequence.Button.onClick.AddListener(() => ToggleState(CurrentSequence, SetCurrentSequenceState));
            Control.Button.onClick.AddListener(() => ToggleState(Control, SetControlState));

            Rollout.GetComponent<Slider>().onValueChanged.AddListener(delegate {SetRolloutLength(Mathf.RoundToInt(Rollout.value));});
            KNN.GetComponent<Slider>().onValueChanged.AddListener(delegate {SetKNN(Mathf.RoundToInt(KNN.value));});

            InitStates();
        }

        void Update() {
            if(Input.GetKey(KeyCode.Escape)) {
                LoadMenu();
                return;
            }
        }
        
        void InitStates(){
            SetCodebookState();
            SetState(Codebook, Codebook.State);
            SetHistoryState();
            SetState(History, History.State);
            SetCurrentSequenceState();
            SetState(CurrentSequence, CurrentSequence.State);
            SetControlState();
            SetState(Control, Control.State);
            SetRolloutLength(Mathf.RoundToInt(Controller.RolloutLength));
            SetKNN(Mathf.RoundToInt(Controller.KNN));
        }

        private void ToggleState(BinaryButton button, Action function) {
            SetState(button, !button.State);
            function();
        }

        private void SetCodebookState() {
            Controller.DrawLabels = Codebook.State;
            Controller.DrawCodebook = Codebook.State;
            Controller.DrawSimilarityMap = Codebook.State;
        }

        private void SetHistoryState() {
            Controller.GetComponent<Actor>().DrawHistory = History.State;
        }

        private void SetCurrentSequenceState() {
            Controller.DrawCurrentSequence = CurrentSequence.State;
        }

        private void SetControlState() {
            Controller.DrawWireCircles = Control.State;
            Controller.DrawRootControl = Control.State;
            Controller.DrawTarget = Control.State;
        }

        private void SetRolloutLength(int value) {
            Controller.RolloutLength = value;
            Rollout.value = value;
            Rollout.transform.Find("Value").GetComponent<TMP_Text>().text = Controller.RolloutLength.ToString();
        }

        private void SetKNN(int value) {
            Controller.KNN = value;
            Controller.Noise = Controller.KNN.Ratio(1, 10);
            KNN.value = value;
            KNN.transform.Find("Value").GetComponent<TMP_Text>().text = Controller.KNN.ToString();
        }

        private void SetState(BinaryButton button, bool state) {
            Image image = button.Button.GetComponent<Image>();
            image.color = state ? ActiveColor : InactiveColor;
            button.State = state;
        }

        private float GetControlVelocity() {
            return Controller.GetRootControl().GetVelocity(Controller.GetRootControl().Pivot).magnitude;
        }

        private float GetMaximumVelocity() {
            return Controller.ControlStrength * Controller.ControlRadius / Controller.GetRootControl().FutureWindow;
        }

        private void OnGUI() {
            UltiDraw.Begin();
            UltiDraw.OnGUILabel(VelocityRect.GetCenter() + new Vector2(0f, LabelOffset), VelocityRect.GetSize(), 0.02f, "Target Velocity: " + GetControlVelocity().ToString("F1") + " m/s", UltiDraw.White, TextAnchor.MiddleCenter);
            UltiDraw.End();
        }
        private void OnRenderObject() {
            UltiDraw.Begin();
            UltiDraw.PlotHorizontalBar(VelocityRect.GetCenter(), VelocityRect.GetSize(), GetControlVelocity()/GetMaximumVelocity(), fillColor: UltiDraw.Cyan, borderColor: UltiDraw.White, backgroundColor:UltiDraw.Black);
            UltiDraw.End();
        }
    }
}