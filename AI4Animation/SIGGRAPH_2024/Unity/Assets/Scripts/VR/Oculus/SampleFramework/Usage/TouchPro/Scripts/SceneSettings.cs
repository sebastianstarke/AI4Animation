// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using UnityEngine;

public class SceneSettings : MonoBehaviour
{
    [Header("Time")]
    [SerializeField] private float m_fixedTimeStep = 0.01f;

    [Header("Physics")]
    [SerializeField] private float m_gravityScalar = 0.75f;
    [SerializeField] private float m_defaultContactOffset = 0.001f;

    private void Awake()
    {
        // Time
        Time.fixedDeltaTime = m_fixedTimeStep;

        // Physics
        Physics.gravity = Vector3.down * 9.81f * m_gravityScalar;
        Physics.defaultContactOffset = m_defaultContactOffset;
    }

    private void Start()
    {
        // Update the contact offset for all existing colliders since setting
        // Physics.defaultContactOffset only applies to newly created colliders.
        CollidersSetContactOffset(m_defaultContactOffset);
    }

    private static void CollidersSetContactOffset(float contactOffset)
    {
        // @Note: This does not find inactive objects.
        Collider[] allColliders = GameObject.FindObjectsOfType<Collider>();
        foreach (Collider collider in allColliders)
        {
            collider.contactOffset = contactOffset;
        }
    }
}
