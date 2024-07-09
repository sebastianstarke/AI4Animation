using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(OVRSceneAnchor))]
[DefaultExecutionOrder(30)]
public class FurnitureSpawner : MonoBehaviour
{
    [Tooltip("Add a point at ceiling.")]
    public GameObject RoomLightPrefab;
    public List<Spawnable> SpawnablePrefabs;

    private OVRSceneAnchor _sceneAnchor;
    private OVRSemanticClassification _classification;
    private static GameObject _roomLightRef;

    private int _frameCounter;

    private void Start()
    {
        _sceneAnchor = GetComponent<OVRSceneAnchor>();
        _classification = GetComponent<OVRSemanticClassification>();
        AddRoomLight();
        SpawnSpawnable();
    }

    private void SpawnSpawnable()
    {
        Spawnable currentSpawnable;
        if (!FindValidSpawnable(out currentSpawnable))
        {
            return;
        }

        // Get current anchor's information
        Vector3 position = transform.position;
        Quaternion rotation = transform.rotation;
        Vector3 localScale = transform.localScale;

        var plane = _sceneAnchor.GetComponent<OVRScenePlane>();
        var volume = _sceneAnchor.GetComponent<OVRSceneVolume>();

        var dimensions = volume ? volume.Dimensions : Vector3.one;

        if (_classification && plane)
        {
	        dimensions = plane.Dimensions;
	        dimensions.z = 1;

	        // Special case 01: Has only top plane
	        if (_classification.Contains(OVRSceneManager.Classification.Desk) ||
	            _classification.Contains(OVRSceneManager.Classification.Couch))
	        {
	            GetVolumeFromTopPlane(
	                transform,
	                plane.Dimensions,
	                transform.position.y,
	                out position,
	                out rotation,
	                out localScale);

	            dimensions = localScale;
	            // The pivot for the resizer is at the top
	            position.y += localScale.y / 2.0f;
	        }

	        // Special case 02: Set wall thickness to something small instead of default value (1.0m)
	        if (_classification.Contains(OVRSceneManager.Classification.WallFace) ||
	            _classification.Contains(OVRSceneManager.Classification.Ceiling) ||
	            _classification.Contains(OVRSceneManager.Classification.Floor))
	        {
	            dimensions.z = 0.01f;
	        }
	    }

        GameObject root = new GameObject("Root");
        root.transform.parent = transform;
        root.transform.SetPositionAndRotation(position, rotation);

        SimpleResizer resizer = new SimpleResizer();
        resizer.CreateResizedObject(dimensions, root, currentSpawnable.ResizablePrefab);
    }

    private bool FindValidSpawnable(out Spawnable currentSpawnable)
    {
	    currentSpawnable = null;

	    if (!_classification) return false;

        var sceneManager = FindObjectOfType<OVRSceneManager>();
        if (!sceneManager) return false;

        foreach (var spawnable in SpawnablePrefabs)
        {
            if(_classification.Contains(spawnable.ClassificationLabel))
            {
                currentSpawnable = spawnable;
                return true;
            }
        }

        return false;
    }

    private void AddRoomLight()
    {
        if (!RoomLightPrefab) return;
        if (_classification && _classification.Contains(OVRSceneManager.Classification.Ceiling) &&
            !_roomLightRef)
        {
            _roomLightRef = Instantiate(RoomLightPrefab, _sceneAnchor.transform);
        }
    }

    private void GetVolumeFromTopPlane(
        Transform plane,
        Vector2 dimensions,
        float height,
        out Vector3 position,
        out Quaternion rotation,
        out Vector3 localScale)
    {
        float halfHeight = height / 2.0f;
        position = plane.position - Vector3.up * halfHeight;
        rotation = Quaternion.LookRotation(-plane.up, Vector3.up);
        localScale = new Vector3(dimensions.x, halfHeight * 2.0f, dimensions.y);
    }
}
