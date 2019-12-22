using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeshRooter : MonoBehaviour {
    void Reset() {
        Bounds bounds = Utility.GetBounds(gameObject);
        transform.position = new Vector3(transform.position.x, -bounds.min.y*transform.lossyScale.y, transform.position.z);
    }
}
