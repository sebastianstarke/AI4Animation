/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using OVRTouchSample;
#if UNITY_EDITOR
using UnityEngine.SceneManagement;
#endif

namespace OVRTouchSample
{
    // Animated hand visuals for a user of a Touch controller.
    [RequireComponent(typeof(OVRGrabber))]
    public class Hand : MonoBehaviour
    {
        public const string ANIM_LAYER_NAME_POINT = "Point Layer";
        public const string ANIM_LAYER_NAME_THUMB = "Thumb Layer";
        public const string ANIM_PARAM_NAME_FLEX = "Flex";
        public const string ANIM_PARAM_NAME_POSE = "Pose";
        public const float THRESH_COLLISION_FLEX = 0.9f;

        public const float INPUT_RATE_CHANGE = 20.0f;

        public const float COLLIDER_SCALE_MIN = 0.01f;
        public const float COLLIDER_SCALE_MAX = 1.0f;
        public const float COLLIDER_SCALE_PER_SECOND = 1.0f;

        public const float TRIGGER_DEBOUNCE_TIME = 0.05f;
        public const float THUMB_DEBOUNCE_TIME = 0.15f;

        [SerializeField]
        private OVRInput.Controller m_controller = OVRInput.Controller.None;
        [SerializeField]
        private Animator m_animator = null;
        [SerializeField]
        private HandPose m_defaultGrabPose = null;

        private Collider[] m_colliders = null;
        private bool m_collisionEnabled = true;
        private OVRGrabber m_grabber;

        List<Renderer> m_showAfterInputFocusAcquired;

        private int m_animLayerIndexThumb = -1;
        private int m_animLayerIndexPoint = -1;
        private int m_animParamIndexFlex = -1;
        private int m_animParamIndexPose = -1;

        private bool m_isPointing = false;
        private bool m_isGivingThumbsUp = false;
        private float m_pointBlend = 0.0f;
        private float m_thumbsUpBlend = 0.0f;

        private bool m_restoreOnInputAcquired = false;

        private void Awake()
        {
            m_grabber = GetComponent<OVRGrabber>();
        }

        private void Start()
        {
            m_showAfterInputFocusAcquired = new List<Renderer>();

            // Collision starts disabled. We'll enable it for certain cases such as making a fist.
            m_colliders = this.GetComponentsInChildren<Collider>().Where(childCollider => !childCollider.isTrigger).ToArray();
            CollisionEnable(false);

            // Get animator layer indices by name, for later use switching between hand visuals
            m_animLayerIndexPoint = m_animator.GetLayerIndex(ANIM_LAYER_NAME_POINT);
            m_animLayerIndexThumb = m_animator.GetLayerIndex(ANIM_LAYER_NAME_THUMB);
            m_animParamIndexFlex = Animator.StringToHash(ANIM_PARAM_NAME_FLEX);
            m_animParamIndexPose = Animator.StringToHash(ANIM_PARAM_NAME_POSE);

            OVRManager.InputFocusAcquired += OnInputFocusAcquired;
            OVRManager.InputFocusLost += OnInputFocusLost;
#if UNITY_EDITOR
            OVRPlugin.SendEvent("custom_hand", (SceneManager.GetActiveScene().name == "CustomHands").ToString(), "sample_framework");
#endif
        }

        private void OnDestroy()
        {
            OVRManager.InputFocusAcquired -= OnInputFocusAcquired;
            OVRManager.InputFocusLost -= OnInputFocusLost;
        }

        private void Update()
        {
            UpdateCapTouchStates();

            m_pointBlend = InputValueRateChange(m_isPointing, m_pointBlend);
            m_thumbsUpBlend = InputValueRateChange(m_isGivingThumbsUp, m_thumbsUpBlend);

            float flex = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, m_controller);

            bool collisionEnabled = m_grabber.grabbedObject == null && flex >= THRESH_COLLISION_FLEX;
            CollisionEnable(collisionEnabled);

            UpdateAnimStates();
        }

        // Just checking the state of the index and thumb cap touch sensors, but with a little bit of
        // debouncing.
        private void UpdateCapTouchStates()
        {
            m_isPointing = !OVRInput.Get(OVRInput.NearTouch.PrimaryIndexTrigger, m_controller);
            m_isGivingThumbsUp = !OVRInput.Get(OVRInput.NearTouch.PrimaryThumbButtons, m_controller);
        }

        private void LateUpdate()
        {
            // Hand's collision grows over a short amount of time on enable, rather than snapping to on, to help somewhat with interpenetration issues.
            if (m_collisionEnabled && m_collisionScaleCurrent + Mathf.Epsilon < COLLIDER_SCALE_MAX)
            {
                m_collisionScaleCurrent = Mathf.Min(COLLIDER_SCALE_MAX, m_collisionScaleCurrent + Time.deltaTime * COLLIDER_SCALE_PER_SECOND);
                for (int i = 0; i < m_colliders.Length; ++i)
                {
                    Collider collider = m_colliders[i];
                    collider.transform.localScale = new Vector3(m_collisionScaleCurrent, m_collisionScaleCurrent, m_collisionScaleCurrent);
                }
            }
        }

        // Simple Dash support. Just hide the hands.
        private void OnInputFocusLost()
        {
            if (gameObject.activeInHierarchy)
            {
                m_showAfterInputFocusAcquired.Clear();
                Renderer[] renderers = GetComponentsInChildren<Renderer>();
                for (int i = 0; i < renderers.Length; ++i)
                {
                    if (renderers[i].enabled)
                    {
                        renderers[i].enabled = false;
                        m_showAfterInputFocusAcquired.Add(renderers[i]);
                    }
                }

                CollisionEnable(false);

                m_restoreOnInputAcquired = true;
            }
        }

        private void OnInputFocusAcquired()
        {
            if (m_restoreOnInputAcquired)
            {
                for (int i = 0; i < m_showAfterInputFocusAcquired.Count; ++i)
                {
                    if (m_showAfterInputFocusAcquired[i])
                    {
                        m_showAfterInputFocusAcquired[i].enabled = true;
                    }
                }
                m_showAfterInputFocusAcquired.Clear();

                // Update function will update this flag appropriately. Do not set it to a potentially incorrect value here.
                //CollisionEnable(true);

                m_restoreOnInputAcquired = false;
            }
        }

        private float InputValueRateChange(bool isDown, float value)
        {
            float rateDelta = Time.deltaTime * INPUT_RATE_CHANGE;
            float sign = isDown ? 1.0f : -1.0f;
            return Mathf.Clamp01(value + rateDelta * sign);
        }

        private void UpdateAnimStates()
        {
            bool grabbing = m_grabber.grabbedObject != null;
            HandPose grabPose = m_defaultGrabPose;
            if (grabbing)
            {
                HandPose customPose = m_grabber.grabbedObject.GetComponent<HandPose>();
                if (customPose != null) grabPose = customPose;
            }
            // Pose
            HandPoseId handPoseId = grabPose.PoseId;
            m_animator.SetInteger(m_animParamIndexPose, (int)handPoseId);

            // Flex
            // blend between open hand and fully closed fist
            float flex = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, m_controller);
            m_animator.SetFloat(m_animParamIndexFlex, flex);

            // Point
            bool canPoint = !grabbing || grabPose.AllowPointing;
            float point = canPoint ? m_pointBlend : 0.0f;
            m_animator.SetLayerWeight(m_animLayerIndexPoint, point);

            // Thumbs up
            bool canThumbsUp = !grabbing || grabPose.AllowThumbsUp;
            float thumbsUp = canThumbsUp ? m_thumbsUpBlend : 0.0f;
            m_animator.SetLayerWeight(m_animLayerIndexThumb, thumbsUp);

            float pinch = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, m_controller);
            m_animator.SetFloat("Pinch", pinch);
        }

        private float m_collisionScaleCurrent = 0.0f;

        private void CollisionEnable(bool enabled)
        {
            if (m_collisionEnabled == enabled)
            {
                return;
            }
            m_collisionEnabled = enabled;

            if (enabled)
            {
                m_collisionScaleCurrent = COLLIDER_SCALE_MIN;
                for (int i = 0; i < m_colliders.Length; ++i)
                {
                    Collider collider = m_colliders[i];
                    collider.transform.localScale = new Vector3(COLLIDER_SCALE_MIN, COLLIDER_SCALE_MIN, COLLIDER_SCALE_MIN);
                    collider.enabled = true;
                }
            }
            else
            {
                m_collisionScaleCurrent = COLLIDER_SCALE_MAX;
                for (int i = 0; i < m_colliders.Length; ++i)
                {
                    Collider collider = m_colliders[i];
                    collider.enabled = false;
                    collider.transform.localScale = new Vector3(COLLIDER_SCALE_MIN, COLLIDER_SCALE_MIN, COLLIDER_SCALE_MIN);
                }
            }
        }
    }
}
