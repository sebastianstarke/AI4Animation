// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

public class ButtonDownListener : MonoBehaviour, UnityEngine.EventSystems.IPointerDownHandler
{
    public event System.Action onButtonDown;

    public void OnPointerDown(PointerEventData eventData)
    {
        if (onButtonDown != null)
        {
            onButtonDown.Invoke();
        }
    }
}
