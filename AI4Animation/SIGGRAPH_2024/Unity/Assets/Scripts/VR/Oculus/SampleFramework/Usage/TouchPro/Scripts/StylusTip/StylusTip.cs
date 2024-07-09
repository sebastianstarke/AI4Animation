// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using UnityEngine;

public class StylusTip : MonoBehaviour
{
    private const int MaxBreadCrumbs = 60;
    private const float BreadCrumbMinSize = 0.005f;
    private const float BreadCrumbMaxSize = 0.02f;

    [Header("External")]
    [SerializeField] private Transform m_trackingSpace;

    [Header("Settings")]
    [SerializeField] private OVRInput.Handedness m_handedness = OVRInput.Handedness.LeftHanded;
    [SerializeField] private GameObject m_breadCrumbPf;

    private GameObject m_breadCrumbContainer;
    private GameObject[] m_breadCrumbs;

    private int m_breadCrumbIndexPrev = -1;
    private int m_breadCrumbIndexCurr = 0;

    private OVRInput.Controller m_controller;

    private void Awake()
    {
        m_controller = m_handedness == OVRInput.Handedness.LeftHanded ? OVRInput.Controller.LTouch : OVRInput.Controller.RTouch;

        // Create the bread crumbs
        m_breadCrumbContainer = new GameObject($"BreadCrumbContainer ({m_handedness})");
        m_breadCrumbs = new GameObject[MaxBreadCrumbs];
        for (int i = 0; i < m_breadCrumbs.Length; ++i)
        {
            // Create bread crumb
            GameObject breadCrumb = GameObject.Instantiate(m_breadCrumbPf, m_breadCrumbContainer.transform);
            breadCrumb.name = $"BreadCrumb ({i})";
            breadCrumb.SetActive(false);

            // Store bread crumb
            m_breadCrumbs[i] = breadCrumb;
        }
    }

    private void Update()
    {
        // Update stylus tip position
        Pose T_device = new Pose(OVRInput.GetLocalControllerPosition(m_controller), OVRInput.GetLocalControllerRotation(m_controller));
        Pose T_world_device = T_device.GetTransformedBy(m_trackingSpace);
        Pose T_world_stylusTip = GetT_Device_StylusTip(m_controller).GetTransformedBy(T_world_device);
        this.transform.SetPositionAndRotation(T_world_stylusTip.position, T_world_stylusTip.rotation);

        // Get stylus tip data
        float stylusTipForce = OVRInput.Get(OVRInput.Axis1D.PrimaryStylusForce, m_controller);
        bool isStylusTipTouching = stylusTipForce > 0;

        // Set the next crumb position
        GameObject nextCrumb = m_breadCrumbs[m_breadCrumbIndexCurr];
        nextCrumb.transform.position = this.transform.position;

        // Set next crumb visuals based on stylus tip force
        float nextCrumbSize = Mathf.Lerp(BreadCrumbMinSize, BreadCrumbMaxSize, stylusTipForce);
        nextCrumb.transform.localScale = new Vector3(nextCrumbSize, nextCrumbSize, nextCrumbSize);
        nextCrumb.GetComponent<MeshRenderer>().material.color = Color.Lerp(Color.white, Color.red, stylusTipForce);
        nextCrumb.SetActive(isStylusTipTouching);

        float crumbSeparation = 0;
        float distanceToPrevCrumb = Mathf.Infinity;
        if (m_breadCrumbIndexPrev >= 0) {
            // Compute next crumb distance to stylus tip
            distanceToPrevCrumb = (this.transform.position - m_breadCrumbs[m_breadCrumbIndexPrev].transform.position).magnitude;

            // Compute next crumb separation by averaging the previous and next crumb sizes
            crumbSeparation = (nextCrumbSize + m_breadCrumbs[m_breadCrumbIndexPrev].transform.localScale.x) * 0.5f;
        }

        // Determine if a new crumb should drop
        if (isStylusTipTouching && (distanceToPrevCrumb >= crumbSeparation)) {
            // Drop the crumb
            m_breadCrumbIndexPrev = m_breadCrumbIndexCurr;
            m_breadCrumbIndexCurr = (m_breadCrumbIndexCurr + 1) % m_breadCrumbs.Length;
        }
    }

    private static Pose GetT_Device_StylusTip(OVRInput.Controller controller) {
        // @Note: Only the next controller supports the stylus tip, but we compute the
        // transforms for all controllers so we can draw the tip at the correct location.
        Pose T_device_stylusTip = Pose.identity;

        if (controller == OVRInput.Controller.LTouch || controller == OVRInput.Controller.RTouch) {
            T_device_stylusTip = new Pose(
                new Vector3(0.0094f, -0.07145f, -0.07565f),
                Quaternion.Euler(35.305f, 50.988f, 37.901f)
            );
        }

        if (controller == OVRInput.Controller.LTouch) {
            T_device_stylusTip.position.x *= -1;
            T_device_stylusTip.rotation.y *= -1;
            T_device_stylusTip.rotation.z *= -1;
        }

        return T_device_stylusTip;
    }
}
