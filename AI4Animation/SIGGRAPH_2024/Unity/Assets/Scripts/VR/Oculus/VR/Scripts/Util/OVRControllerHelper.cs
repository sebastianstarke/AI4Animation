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

using UnityEngine;

/// <summary>
/// Simple helper script that conditionally enables rendering of a controller if it is connected.
/// </summary>
public class OVRControllerHelper : MonoBehaviour
{
	/// <summary>
	/// The root GameObject that represents the Oculus Touch for Quest And RiftS Controller model (Left).
	/// </summary>
	public GameObject m_modelOculusTouchQuestAndRiftSLeftController;

	/// <summary>
	/// The root GameObject that represents the Oculus Touch for Quest And RiftS Controller model (Right).
	/// </summary>
	public GameObject m_modelOculusTouchQuestAndRiftSRightController;

	/// <summary>
	/// The root GameObject that represents the Oculus Touch for Rift Controller model (Left).
	/// </summary>
	public GameObject m_modelOculusTouchRiftLeftController;

	/// <summary>
	/// The root GameObject that represents the Oculus Touch for Rift Controller model (Right).
	/// </summary>
	public GameObject m_modelOculusTouchRiftRightController;

	/// <summary>
	/// The root GameObject that represents the Oculus Touch for Quest 2 Controller model (Left).
	/// </summary>
	public GameObject m_modelOculusTouchQuest2LeftController;

	/// <summary>
	/// The root GameObject that represents the Oculus Touch for Quest 2 Controller model (Right).
	/// </summary>
	public GameObject m_modelOculusTouchQuest2RightController;

	/// <summary>
	/// The root GameObject that represents the Meta Touch Pro Controller model (Left).
	/// </summary>
	public GameObject m_modelMetaTouchProLeftController;

	/// <summary>
	/// The root GameObject that represents the Meta Touch Pro Controller model (Right).
	/// </summary>
	public GameObject m_modelMetaTouchProRightController;

	/// <summary>
	/// The controller that determines whether or not to enable rendering of the controller model.
	/// </summary>
	public OVRInput.Controller m_controller;

	/// <summary>
	/// The animator component that contains the controller animation controller for animating buttons and triggers.
	/// </summary>
	private Animator m_animator;

	private GameObject m_activeController;

	private bool m_controllerModelsInitialized = false;

	private bool m_hasInputFocus = true;
	private bool m_hasInputFocusPrev = false;

	private enum ControllerType
	{
		QuestAndRiftS = 1,
		Rift = 2,
		Quest2 = 3,
		TouchPro = 4,
	}

	private ControllerType activeControllerType = ControllerType.Rift;

	private bool m_prevControllerConnected = false;
	private bool m_prevControllerConnectedCached = false;

	void Start()
	{
		if (OVRManager.OVRManagerinitialized)
		{
			InitializeControllerModels();
		}
	}

	void InitializeControllerModels()
	{
		if (m_controllerModelsInitialized)
			return;
	
		OVRPlugin.SystemHeadset headset = OVRPlugin.GetSystemHeadsetType();
		OVRPlugin.Hand controllerHand = m_controller == OVRInput.Controller.LTouch ? OVRPlugin.Hand.HandLeft : OVRPlugin.Hand.HandRight;
		switch (headset)
		{
			case OVRPlugin.SystemHeadset.Rift_CV1:
				activeControllerType = ControllerType.Rift;
				break;
			case OVRPlugin.SystemHeadset.Oculus_Quest_2:
				if (OVRPlugin.GetCurrentInteractionProfile(controllerHand) == OVRPlugin.InteractionProfile.TouchPro) 
				{
					activeControllerType = ControllerType.TouchPro;
				} else {
					activeControllerType = ControllerType.Quest2;
				}
				break;
			case OVRPlugin.SystemHeadset.Oculus_Link_Quest_2:
				if (OVRPlugin.GetCurrentInteractionProfile(controllerHand) == OVRPlugin.InteractionProfile.TouchPro) {
					activeControllerType = ControllerType.TouchPro;
				} else {
					activeControllerType = ControllerType.Quest2;
				}
				break;
			case OVRPlugin.SystemHeadset.Meta_Quest_Pro:
				activeControllerType = ControllerType.TouchPro;
				break;
			case OVRPlugin.SystemHeadset.Meta_Link_Quest_Pro:
				activeControllerType = ControllerType.TouchPro;
				break;
			default:
				activeControllerType = ControllerType.QuestAndRiftS;
				break;
		}

		Debug.LogFormat("OVRControllerHelp: Active controller type: {0} for product {1} (headset {2}, hand {3})", 
			activeControllerType, OVRPlugin.productName, headset, controllerHand);

		// Hide all controller models until controller get connected
		m_modelOculusTouchQuestAndRiftSLeftController.SetActive(false);
		m_modelOculusTouchQuestAndRiftSRightController.SetActive(false);
		m_modelOculusTouchRiftLeftController.SetActive(false);
		m_modelOculusTouchRiftRightController.SetActive(false);
		m_modelOculusTouchQuest2LeftController.SetActive(false);
		m_modelOculusTouchQuest2RightController.SetActive(false);
		m_modelMetaTouchProLeftController.SetActive(false);
		m_modelMetaTouchProRightController.SetActive(false);

		OVRManager.InputFocusAcquired += InputFocusAquired;
		OVRManager.InputFocusLost += InputFocusLost;

		m_controllerModelsInitialized = true;
	}

	void Update()
	{
		if (!m_controllerModelsInitialized)
		{
			if (OVRManager.OVRManagerinitialized)
			{
				InitializeControllerModels();
			}
			else
			{
				return;
			}
		}

		bool controllerConnected = OVRInput.IsControllerConnected(m_controller);

		if ((controllerConnected != m_prevControllerConnected) || !m_prevControllerConnectedCached || (m_hasInputFocus != m_hasInputFocusPrev))
		{
			if (activeControllerType == ControllerType.Rift)
			{
				m_modelOculusTouchQuestAndRiftSLeftController.SetActive(false);
				m_modelOculusTouchQuestAndRiftSRightController.SetActive(false);
				m_modelOculusTouchRiftLeftController.SetActive(controllerConnected && (m_controller == OVRInput.Controller.LTouch));
				m_modelOculusTouchRiftRightController.SetActive(controllerConnected && (m_controller == OVRInput.Controller.RTouch));
				m_modelOculusTouchQuest2LeftController.SetActive(false);
				m_modelOculusTouchQuest2RightController.SetActive(false);
				m_modelMetaTouchProLeftController.SetActive(false);
				m_modelMetaTouchProRightController.SetActive(false);

				m_animator = m_controller == OVRInput.Controller.LTouch ? m_modelOculusTouchRiftLeftController.GetComponent<Animator>() :
					m_modelOculusTouchRiftRightController.GetComponent<Animator>();
				m_activeController = m_controller == OVRInput.Controller.LTouch ? m_modelOculusTouchRiftLeftController : m_modelOculusTouchRiftRightController;
			}
			else if (activeControllerType == ControllerType.Quest2)
			{
				m_modelOculusTouchQuestAndRiftSLeftController.SetActive(false);
				m_modelOculusTouchQuestAndRiftSRightController.SetActive(false);
				m_modelOculusTouchRiftLeftController.SetActive(false);
				m_modelOculusTouchRiftRightController.SetActive(false);
				m_modelOculusTouchQuest2LeftController.SetActive(controllerConnected && (m_controller == OVRInput.Controller.LTouch));
				m_modelOculusTouchQuest2RightController.SetActive(controllerConnected && (m_controller == OVRInput.Controller.RTouch));
				m_modelMetaTouchProLeftController.SetActive(false);
				m_modelMetaTouchProRightController.SetActive(false);

				m_animator = m_controller == OVRInput.Controller.LTouch ? m_modelOculusTouchQuest2LeftController.GetComponent<Animator>() :
					m_modelOculusTouchQuest2RightController.GetComponent<Animator>();
				m_activeController = m_controller == OVRInput.Controller.LTouch ? m_modelOculusTouchQuest2LeftController : m_modelOculusTouchQuest2RightController;
			}
			else if (activeControllerType == ControllerType.QuestAndRiftS)
			{
				m_modelOculusTouchQuestAndRiftSLeftController.SetActive(controllerConnected && (m_controller == OVRInput.Controller.LTouch));
				m_modelOculusTouchQuestAndRiftSRightController.SetActive(controllerConnected && (m_controller == OVRInput.Controller.RTouch));
				m_modelOculusTouchRiftLeftController.SetActive(false);
				m_modelOculusTouchRiftRightController.SetActive(false);
				m_modelOculusTouchQuest2LeftController.SetActive(false);
				m_modelOculusTouchQuest2RightController.SetActive(false);
				m_modelMetaTouchProLeftController.SetActive(false);
				m_modelMetaTouchProRightController.SetActive(false);

				m_animator = m_controller == OVRInput.Controller.LTouch ? m_modelOculusTouchQuestAndRiftSLeftController.GetComponent<Animator>() :
					m_modelOculusTouchQuestAndRiftSRightController.GetComponent<Animator>();
				m_activeController = m_controller == OVRInput.Controller.LTouch ? m_modelOculusTouchQuestAndRiftSLeftController : m_modelOculusTouchQuestAndRiftSRightController;
			}
			else /*if (activeControllerType == ControllerType.TouchPro)*/ 
			{
				m_modelOculusTouchQuestAndRiftSLeftController.SetActive(false);
				m_modelOculusTouchQuestAndRiftSRightController.SetActive(false);
				m_modelOculusTouchRiftLeftController.SetActive(false);
				m_modelOculusTouchRiftRightController.SetActive(false);
				m_modelOculusTouchQuest2LeftController.SetActive(false);
				m_modelOculusTouchQuest2RightController.SetActive(false);
				m_modelMetaTouchProLeftController.SetActive(controllerConnected && (m_controller == OVRInput.Controller.LTouch));
				m_modelMetaTouchProRightController.SetActive(controllerConnected && (m_controller == OVRInput.Controller.RTouch));

				m_animator = m_controller == OVRInput.Controller.LTouch ? m_modelMetaTouchProLeftController.GetComponent<Animator>() :
					m_modelMetaTouchProRightController.GetComponent<Animator>();
				m_activeController = m_controller == OVRInput.Controller.LTouch ? m_modelMetaTouchProLeftController : m_modelMetaTouchProRightController;
			}

			m_activeController.SetActive(m_hasInputFocus && controllerConnected);

			m_prevControllerConnected = controllerConnected;
			m_prevControllerConnectedCached = true;
			m_hasInputFocusPrev = m_hasInputFocus;
		}

		if (m_animator != null)
		{
			m_animator.SetFloat("Button 1", OVRInput.Get(OVRInput.Button.One, m_controller) ? 1.0f : 0.0f);
			m_animator.SetFloat("Button 2", OVRInput.Get(OVRInput.Button.Two, m_controller) ? 1.0f : 0.0f);
			m_animator.SetFloat("Button 3", OVRInput.Get(OVRInput.Button.Start, m_controller) ? 1.0f : 0.0f);

			m_animator.SetFloat("Joy X", OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, m_controller).x);
			m_animator.SetFloat("Joy Y", OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, m_controller).y);

			m_animator.SetFloat("Trigger", OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, m_controller));
			m_animator.SetFloat("Grip", OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, m_controller));
		}
	}

	public void InputFocusAquired()
	{
		m_hasInputFocus = true;
	}

	public void InputFocusLost()
	{
		m_hasInputFocus = false;
	}
}
