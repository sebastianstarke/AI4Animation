using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(OVRSceneAnchor))]
public class VolumeAndPlaneSwitcher : MonoBehaviour
{
  public OVRSceneAnchor planePrefab;
  public OVRSceneAnchor volumePrefab;
  public enum GeometryType
  {
    Plane,
    Volume,
  }
  [System.Serializable]
  public struct LabelGeometryPair
  {
    public string label;
    public GeometryType desiredGeometryType;
  }
  public List<LabelGeometryPair> desiredSwitches;

  private void ReplaceAnchor(OVRSceneAnchor prefab, Vector3 position, Quaternion rotation, Vector3 localScale)
  {
    var anchor = Instantiate(prefab, transform.parent);
    anchor.enabled = false; // disable so it won't update transform
    anchor.InitializeFrom(GetComponent<OVRSceneAnchor>());

    anchor.transform.SetPositionAndRotation(position, rotation);
    foreach (Transform child in anchor.transform)
    {
      child.localScale = localScale;
    }

    Destroy(gameObject);
  }

  void Start()
  {
    var classification = GetComponent<OVRSemanticClassification>();
    if (!classification) return;

    foreach (LabelGeometryPair pair in desiredSwitches)
    {
      if (classification.Contains(pair.label))
      {
        Vector3 position = Vector3.zero;
        Quaternion rotation = Quaternion.identity;
        Vector3 localScale = Vector3.zero;
        switch (pair.desiredGeometryType)
        {
          case GeometryType.Plane:
          {
            var volume = GetComponent<OVRSceneVolume>();
            if (!volume)
            {
              Debug.LogWarning($"Ignoring desired volume to plane switch for {pair.label} because it is not a volume.");
              continue;
            }

            Debug.Log($"IN Volume Position {transform.position}, Dimensions: {volume.Dimensions}");
            // This object is a volume, but we want a plane instead.
            GetTopPlaneFromVolume(
              transform,
              volume.Dimensions,
              out position,
              out rotation,
              out localScale);
            Debug.Log($"OUT Plane Position {position}, Dimensions: {localScale}");
            ReplaceAnchor(planePrefab, position, rotation, localScale);
            break;
          }
          case GeometryType.Volume:
          {
            var plane = GetComponent<OVRScenePlane>();
            if (!plane)
            {
              Debug.LogWarning($"Ignoring desired plane to volume switch for {pair.label} because it is not a plane.");
              continue;
            }

            Debug.Log($"IN Plane Position {transform.position}, Dimensions: {plane.Dimensions}");
            // This object is a plane, but we want a volume instead.
            GetVolumeFromTopPlane(
              transform,
              plane.Dimensions,
              transform.position.y,
              out position,
              out rotation,
              out localScale);
            Debug.Log($"OUT Volume Position {position}, Dimensions: {localScale}");
            ReplaceAnchor(volumePrefab, position, rotation, localScale);
            break;
          }
        }
      }
    }
    // IF we arrived here, no conversion was needed. Let's remove this component
    Destroy(this);
  }

  private void GetVolumeFromTopPlane(
    Transform plane,
    Vector2 dimensions,
    float height,
    out Vector3 position,
    out Quaternion rotation,
    out Vector3 localScale)
  {
    position = plane.position;
    rotation = plane.rotation;
    localScale = new Vector3(dimensions.x, dimensions.y, height);
  }

  private void GetTopPlaneFromVolume(
    Transform volume,
    Vector3 dimensions,
    out Vector3 position,
    out Quaternion rotation,
    out Vector3 localScale)
  {
    float halfHeight = dimensions.y / 2.0f;
    position = volume.position + Vector3.up * halfHeight;
    rotation = Quaternion.LookRotation(Vector3.up, -volume.forward);
    localScale = new Vector3(dimensions.x, dimensions.z, dimensions.y);
  }
}
