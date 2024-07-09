using UnityEngine;
using UnityEngine.UI;

namespace SIGGRAPH_2024 {
    public class ControllerUI : MonoBehaviour {
        private TrackingSystem TrackingSystem;
        private Image LeftControllerArrow;
        private Image RightControllerArrow;
        private Image LeftControllerTrigger;
        private Image RightControllerTrigger;

        public Vector3 Velocity;

        void Awake() {
            TrackingSystem = FindObjectOfType<TrackingSystem>();
            LeftControllerArrow = transform.Find("LeftController/Arrow").GetComponent<Image>();
            RightControllerArrow = transform.Find("RightController/Arrow").GetComponent<Image>();
            LeftControllerTrigger = transform.Find("LeftController/Trigger").GetComponent<Image>();
            RightControllerTrigger = transform.Find("RightController/Trigger").GetComponent<Image>();
        }

        void Update() {
            UpdateArrow(LeftControllerArrow, TrackingSystem.LeftController.GetJoystickAxis());
            UpdateArrow(RightControllerArrow, TrackingSystem.RightController.GetJoystickAxis());
            LeftControllerTrigger.gameObject.SetActive(TrackingSystem.LeftController.GetButton(TrackingSystem.BUTTON.Trigger));
            RightControllerTrigger.gameObject.SetActive(TrackingSystem.RightController.GetButton(TrackingSystem.BUTTON.Trigger));
        }

        private void UpdateArrow(Image image, Vector2 axis) {
            image.transform.localRotation = Quaternion.Euler(0f, 0f, Vector2.SignedAngle(Vector2.up, axis));
            image.transform.localScale = axis.magnitude * new Vector3(1f, -1f, 1f);
        }
    }
}