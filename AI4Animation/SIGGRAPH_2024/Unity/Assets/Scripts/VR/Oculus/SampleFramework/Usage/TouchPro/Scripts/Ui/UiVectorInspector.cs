// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using TMPro;
using UnityEngine;

public class UiVectorInspector : MonoBehaviour
{
    [Header("Components")]
    [SerializeField] private TextMeshProUGUI m_nameLabel = null;
    [SerializeField] private TextMeshProUGUI m_valueLabel = null;

    public void SetName(string name)
    {
        m_nameLabel.text = name;
    }

    public void SetValue(bool value)
    {
        m_valueLabel.text = string.Format("[{0}]", value);
    }
}
