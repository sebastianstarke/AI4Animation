using UnityEngine;
using System.Collections.Generic;
using VoxelSystem;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

[ExecuteInEditMode]
public class VoxelCollider : MonoBehaviour {

    public bool Combine = true;
    public bool Optimise = true;
    public int Resolution = 10;

	public bool ShowBounds = false;
	public bool ShowGeometry = false;

    [SerializeField] private Vector3 BoundsMin = Vector3.zero;
    [SerializeField] private Vector3 BoundsMax = Vector3.zero;

    private Transform Colliders = null;

    void Reset() {
        Generate();
    }

    public BoxCollider[] GetColliders() {
        return GetContainer().GetComponentsInChildren<BoxCollider>();
    }

    public void Clear() {
        if(Colliders != null) {
            Utility.Destroy(Colliders.gameObject);
        }
    }

    public Transform GetContainer() {
        if(Colliders != null) {
            return Colliders;
        }
        Colliders = transform.Find("Colliders");
        if(Colliders != null) {
            return Colliders;
        }
        Colliders = new GameObject("Colliders").transform;
        Colliders.SetParent(transform);
        Colliders.transform.localPosition = Vector3.zero;
        Colliders.transform.localRotation = Quaternion.identity;
        Colliders.transform.localScale = Vector3.one;
        Colliders.gameObject.layer = gameObject.layer;
        return Colliders;
    }

    void OnDestroy() {
        #if UNITY_EDITOR
        if(EditorApplication.isCompiling) {
            return;
        }
        if(UnityEditor.SceneManagement.PrefabStageUtility.GetPrefabStage(gameObject) != null) {
            return;
        }
        if(EditorSceneManager.GetActiveScene() != gameObject.scene) {
            return;
        }
        if(EditorApplication.isPlayingOrWillChangePlaymode) {
            return;
        }
        #endif
        Clear();
    }

    public Vector3 GetCenter() {
        return transform.position + transform.rotation * (0.5f * (GetBoundsMin() + GetBoundsMax()));
    }

    public Vector3 GetExtents() {
        return GetBoundsMax() - GetBoundsMin();
    }

    public Vector3 GetBoundsMin() {
        return Vector3.Scale(transform.lossyScale, BoundsMin);
    }

    public Vector3 GetBoundsMax() {
        return Vector3.Scale(transform.lossyScale, BoundsMax);
    }

    public void Generate() {
        Clear();
        Setup(transform);
        if(Resolution == 0) {
            return;
        }
        if(Combine) {
            List<Voxel_t> voxels;
            float unit;
            CPUVoxelizer.Voxelize(ComputeMesh(), Resolution, out voxels, out unit);
            List<BoxCollider> colliders = new List<BoxCollider>();
            for(int i=0; i<voxels.Count; i++) {
                BoxCollider c = GetContainer().gameObject.AddComponent<BoxCollider>();
                c.center = voxels[i].position;
                c.size = new Vector3(unit, unit, unit);
                colliders.Add(c);
            }
            if(Optimise) {
                FilterVoxels(colliders, Axis.XPositive);
                FilterVoxels(colliders, Axis.YPositive);
                FilterVoxels(colliders, Axis.ZPositive);
            }
        } else {
            foreach(MeshFilter filter in GetComponentsInChildren<MeshFilter>()) {
                List<Voxel_t> voxels;
                float unit;
                CPUVoxelizer.Voxelize(filter.sharedMesh, Resolution, out voxels, out unit);
                List<BoxCollider> colliders = new List<BoxCollider>();
                for(int i=0; i<voxels.Count; i++) {
                    BoxCollider c = GetContainer().gameObject.AddComponent<BoxCollider>();
                    c.center = (transform.worldToLocalMatrix * filter.transform.localToWorldMatrix).MultiplyPoint3x4(voxels[i].position);
                    c.size = new Vector3(unit, unit, unit);
                    colliders.Add(c);
                }
                if(Optimise) {
                    FilterVoxels(colliders, Axis.XPositive);
                    FilterVoxels(colliders, Axis.YPositive);
                    FilterVoxels(colliders, Axis.ZPositive);
                }
            }
        }
        {
            BoundsMin = Vector3.zero;
            BoundsMax = Vector3.zero;
            BoxCollider[] colliders = GetColliders();
            foreach(BoxCollider c in colliders) {
                BoundsMin = Vector3.Min(BoundsMin, c.center - 0.5f*c.size);
                BoundsMax = Vector3.Max(BoundsMax, c.center + 0.5f*c.size);
            }
        }
    }
    
    void OnRenderObject() {
        UltiDraw.Begin();

		if(ShowGeometry) {
			foreach(BoxCollider c in GetColliders()) {
				UltiDraw.DrawCuboid(c.transform.position + c.transform.rotation * Vector3.Scale(c.center, c.transform.lossyScale), c.transform.rotation, Vector3.Scale(c.size, c.transform.lossyScale), UltiDraw.Green.Opacity(0.5f));
			}
		}
        
		if(ShowBounds) {
			UltiDraw.DrawWireCuboid(GetCenter(), transform.rotation, GetExtents(), UltiDraw.Black);
		}

        UltiDraw.End();
    }

    private Vector3 GetCenter(BoxCollider c) {
        return transform.position + transform.rotation * Vector3.Scale(c.center, transform.lossyScale);
    }

    private Vector3 GetIncrement(BoxCollider c, Axis axis) {
        return transform.rotation * Vector3.Scale(GetSize(c), axis.GetAxis());
    }

    private Vector3 GetSize(BoxCollider c) {
        return Vector3.Scale(c.size, transform.lossyScale);
    }

    private Voxel GetConnection(Voxel[] voxels, Voxel reference, int step, Axis axis) {
        Vector3 pivot = reference.Center + step*GetIncrement(reference.Collider, axis);
        foreach(Voxel v in voxels) {
            if(!v.Masked && v.Center == pivot && v.Size.Zero(axis) == reference.Size.Zero(axis)) {
                return v;
            }
        }
        return null;
    }

    private void FilterVoxels(List<BoxCollider> colliders, Axis axis) {
        List<BoxCollider> instances = new List<BoxCollider>();
        Voxel[] voxels = new Voxel[colliders.Count];
        for(int i=0; i<colliders.Count; i++) {
            voxels[i] = new Voxel(colliders[i], GetCenter(colliders[i]), GetSize(colliders[i]));
        }
        foreach(Voxel v in voxels) {
            if(v.Masked) {
                continue;
            }
            BoxCollider c = GetContainer().gameObject.AddComponent<BoxCollider>();
            c.enabled = false;
            int step = 1;
            while(true) {
                Voxel connection = GetConnection(voxels, v, step, axis);
                if(connection != null) {
                    c.center = v.Collider.center + Vector3.Scale(v.Collider.size, 0.5f*step*axis.GetAxis());
                    c.size = Vector3.Scale(v.Collider.size, Vector3.one + step*axis.GetAxis());
                    connection.Masked = true;
                    c.enabled = true;
                    step += 1;
                } else {
                    break;
                }
            }
            if(!c.enabled) {
                Utility.Destroy(c);
            } else {
                v.Masked = true;
                instances.Add(c);
            }
        }
        foreach(Voxel v in voxels) {
            if(v.Masked) {
                colliders.Remove(v.Collider);
                Utility.Destroy(v.Collider);
            }
        }
        foreach(BoxCollider c in instances) {
            colliders.Add(c);
        }
    }

    private class Voxel {
        public BoxCollider Collider;
        public Vector3 Center;
        public Vector3 Size;
        public bool Masked;
        public Voxel(BoxCollider collider, Vector3 center, Vector3 size) {
            Collider = collider;
            Center = center;
            Size = size;
            Masked = false;
        }
    }

    private Mesh ComputeMesh() {
        MeshFilter[] filters = GetComponentsInChildren<MeshFilter>();
        CombineInstance[] combine = new CombineInstance[filters.Length];
        for(int i=0; i<combine.Length; i++) {
            combine[i].mesh = filters[i].sharedMesh;
            combine[i].transform = transform.worldToLocalMatrix * filters[i].transform.localToWorldMatrix;
        }
        Mesh mesh = new Mesh();
        mesh.CombineMeshes(combine, true, true);
        return mesh;
    }

    private void Setup(Transform t) {
        t.gameObject.isStatic = false;
        for(int i=0; i<t.childCount; i++) {
            Setup(t.GetChild(i));
        }
    }

	#if UNITY_EDITOR
	[CustomEditor(typeof(VoxelCollider))]
	public class VoxelCollider_Editor : Editor {

		public VoxelCollider Target;

		void Awake() {
			Target = (VoxelCollider)target;
            Target.GetContainer().gameObject.hideFlags = HideFlags.None;
		}

		public override void OnInspectorGUI() {
			Undo.RecordObject(Target, Target.name);

            if(Utility.GUIButton("Generate", UltiDraw.DarkGrey, UltiDraw.White)) {
                Target.Generate();
            }
            Target.Combine = EditorGUILayout.Toggle("Combine Meshes", Target.Combine);
            Target.Optimise = EditorGUILayout.Toggle("Optimise", Target.Optimise);
            Target.Resolution = Mathf.Clamp(EditorGUILayout.IntField("Resolution", Target.Resolution), 1, 50);
            EditorGUILayout.HelpBox("Colliders: " + Target.GetColliders().Length, MessageType.None);
            EditorGUILayout.HelpBox("Bounds Min: " + Target.BoundsMin.ToString("F3"), MessageType.None);
            EditorGUILayout.HelpBox("Bounds Max: " + Target.BoundsMax.ToString("F3"), MessageType.None);
            EditorGUILayout.HelpBox("Bounds: " + Target.GetExtents().ToString("F3"), MessageType.None);
			if(Utility.GUIButton("Show Bounds", Target.ShowBounds ? UltiDraw.Cyan : UltiDraw.Grey, UltiDraw.Black)) {
				Target.ShowBounds = !Target.ShowBounds;
			}
			if(Utility.GUIButton("Show Geometry", Target.ShowGeometry ? UltiDraw.Cyan : UltiDraw.Grey, UltiDraw.Black)) {
				Target.ShowGeometry = !Target.ShowGeometry;
			}

			if(GUI.changed) {
				EditorUtility.SetDirty(Target);
			}
		}

	}
	#endif

}