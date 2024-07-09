// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class UiAxis1dInspector : MonoBehaviour
{
    [Header("Settings")]
    [SerializeField] private float m_minExtent = 0;
    [SerializeField] private float m_maxExtent = 1;

    [Header("Components")]
    [SerializeField] private TextMeshProUGUI m_nameLabel = null;
    [SerializeField] private TextMeshProUGUI m_valueLabel = null;
    [SerializeField] private Slider m_slider = null;

    public void SetExtents(float minExtent, float maxExtent)
    {
        m_minExtent = minExtent;
        m_maxExtent = maxExtent;
    }

    public void SetName(string name)
    {
        m_nameLabel.text = name;
    }

    public void SetValue(float value)
    {
        m_valueLabel.text = string.Format("[{0}]", value.ToString("f2"));

        m_slider.minValue = Mathf.Min(value, m_minExtent);
        m_slider.maxValue = Mathf.Max(value, m_maxExtent);

        m_slider.value = value;
    }
}
