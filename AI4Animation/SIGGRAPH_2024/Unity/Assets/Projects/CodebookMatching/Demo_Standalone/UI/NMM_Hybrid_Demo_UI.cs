using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using AI4Animation;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace SIGGRAPH_2024 {
    public class NMM_Hybrid_Demo_UI : MonoBehaviour {
        public TrackingSystem TS;
        public MotionController Controller;
        public List<MotionAsset> Assets = new List<MotionAsset>();

        public Actor ReferenceActor;
        public BinaryButton Control, BodyPrediction, FuturePrediction;
        public BinaryButton Clip1, Clip2, Clip3, Clip4, Clip5, Clip6;
        public Image ProgressionBar;
        public Color InactiveColor = UltiDraw.White;
        public Color ActiveColor = UltiDraw.Magenta;
        public Color ProgressColor = UltiDraw.Cyan;

        private MotionAsset Asset = null;

        [Serializable]
        public class BinaryButton {
            public bool State;
            public Button Button;
        }

        void Start() {
            Control.Button.onClick.AddListener(() => ToggleState(Control, SetControlState));
            BodyPrediction.Button.onClick.AddListener(() => ToggleState(BodyPrediction, SetBodyPredictionState));
            FuturePrediction.Button.onClick.AddListener(() => ToggleState(FuturePrediction, SetFuturePredictionState));

            Clip1.Button.onClick.AddListener(() => ToggleState(Clip1, () => SetAsset(Clip1, 0)));
            Clip2.Button.onClick.AddListener(() => ToggleState(Clip2, () => SetAsset(Clip2, 1)));
            Clip3.Button.onClick.AddListener(() => ToggleState(Clip3, () => SetAsset(Clip3, 2)));
            Clip4.Button.onClick.AddListener(() => ToggleState(Clip4, () => SetAsset(Clip4, 3)));
            Clip5.Button.onClick.AddListener(() => ToggleState(Clip5, () => SetAsset(Clip5, 4)));
            Clip6.Button.onClick.AddListener(() => ToggleState(Clip6, () => SetAsset(Clip6, 5)));
            InitStates();
        }

        void InitStates(){
            SetControlState();
            SetState(Control, Control.State);
            SetBodyPredictionState();
            SetState(BodyPrediction, BodyPrediction.State);
            SetFuturePredictionState();
            SetState(FuturePrediction, FuturePrediction.State);

            SetAsset(Clip1, 0);
        }

        public void LoadMenu() {  
            SceneManager.LoadScene(0);  
        }

        void Update(){
            if(Input.GetKey(KeyCode.Escape)) {
                LoadMenu();
                return;
            }

            if(TS.AssetTimestamp > TS.Asset.GetTotalTime()) {
                Initialize();
            }

            // Animate reference
            ReferenceActor.gameObject.SetActive(true);
            ReferenceActor.SetBoneTransformations(TS.Asset.GetFrame(TS.AssetTimestamp).GetBoneTransformations(ReferenceActor.GetBoneNames(), TS.AssetMirrored), ReferenceActor.GetBoneNames());   
            
            ProgressionBar.color = ProgressColor;
            ProgressionBar.fillAmount = TS.AssetTimestamp / TS.Asset.GetTotalTime();
            ProgressionBar.gameObject.GetComponentInChildren<TMP_Text>().text = "Progression " + (TS.AssetTimestamp / TS.Asset.GetTotalTime() * 100f).ToString("F1") + "%";
        }

        private void Initialize() {
            TS.SetMotionAsset(Asset);
            Controller.Initialize(
                TS.Asset.GetModule<RootModule>("BodyWorld").GetRootTransformation(TS.AssetTimestamp, false),
                TS.Asset.GetFrame(TS.AssetTimestamp).GetBoneTransformations(Controller.GetComponent<Actor>().GetBoneNames(), false),
                TS.Asset.GetFrame(TS.AssetTimestamp).GetBoneVelocities(Controller.GetComponent<Actor>().GetBoneNames(), false)
            );
        }

        private void SetAsset(BinaryButton button, int index){
            Asset = Assets[index];
            SetState(Clip1, false);
            SetState(Clip2, false);
            SetState(Clip3, false);
            SetState(Clip4, false);
            SetState(Clip5, false);
            SetState(Clip6, false);
            
            SetState(button, true);

            Initialize();
        }

        private void ToggleState(BinaryButton button, Action function) {
            SetState(button, !button.State);
            function();
        }

        private void SetControlState() {
            Controller.DrawWireCircles = Control.State;
            Controller.DrawRootControl = Control.State;
            Controller.DrawTarget = Control.State;
        }

        private void SetBodyPredictionState() {
            TS.ShowActor = BodyPrediction.State;
            TS.ShowTrackers = BodyPrediction.State;
        }

        private void SetFuturePredictionState() {
            TS.DrawMotionFuture = FuturePrediction.State;
        }

        private void SetState(BinaryButton button, bool state) {
            Image image = button.Button.GetComponent<Image>();
            image.color = state ? ActiveColor : InactiveColor;
            button.State = state;
        }

        public void FadeIn(Material material, float duration, float delay) {
            StartCoroutine(ProcessFade(material, duration, delay, 1f, 0f));
        }

        public void FadeOut(Material material, float duration, float delay) {
            StartCoroutine(ProcessFade(material, duration, delay, 0f, 1f));
        }

        private IEnumerator ProcessFade(Material material, float duration, float delay, float start, float end) {
            if(delay > 0f) {
                yield return new WaitForSeconds(delay);
            }
            float remaining = duration;
            while(remaining > 0f) {
                remaining -= Time.deltaTime;
                float value = remaining / duration;
                value = value.Normalize(0f, 1f, start, end);

                Color color = material.color;
                color.a = value;
                material.color = color;

                yield return 0f;
            }
        }
    }
}