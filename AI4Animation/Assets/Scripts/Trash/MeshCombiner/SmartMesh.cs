
using UnityEngine;
using UnityEngine.Assertions;
using System.Linq;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace SmartCombine {
	public class SmartMeshData {
		public Mesh mesh {get; private set;}
		public Matrix4x4 transform {get; private set;}

		Material[] _materials;
		public IList<Material> materials {get {return new ReadOnlyCollection<Material>(_materials);}}

		public SmartMeshData(Mesh inMesh, Material[] inMaterials, Matrix4x4 inTransform) {
			Assert.IsTrue(inMesh, "Mesh cannot be null");
			Assert.IsNotNull<Material[]>(inMaterials, "Materials array cannot be null");

			mesh = inMesh;
			_materials = inMaterials;
			transform = inTransform;

			if (_materials.Length != mesh.subMeshCount) {
				Debug.LogWarning("SmartMeshData has incorrect number of materials. Resizing to match submesh count");
				Material[] resizedMaterials = new Material[mesh.subMeshCount];
				for (int i = 0; i < _materials.Length; i++) {
					if (i < _materials.Length) {
						resizedMaterials[i] = _materials[i];
					} else {
						resizedMaterials[i] = null;
					}
				}
				_materials = resizedMaterials;
			}
		}

		public SmartMeshData(Mesh inputMesh, Material[] inputMaterials)
			: this(inputMesh, inputMaterials, Matrix4x4.identity) {
		}

		public SmartMeshData(Mesh inputMesh, Material[] inputMaterials, Vector3 position)
			: this(inputMesh, inputMaterials, Matrix4x4.TRS(position, Quaternion.identity, Vector3.one)) {
		}

		public SmartMeshData(Mesh inputMesh, Material[] inputMaterials, Vector3 position, Quaternion rotation)
			: this(inputMesh, inputMaterials, Matrix4x4.TRS(position, rotation, Vector3.one)) {
		}

		public SmartMeshData(Mesh inputMesh, Material[] inputMaterials, Vector3 position, Quaternion rotation, Vector3 scale)
			: this(inputMesh, inputMaterials, Matrix4x4.TRS(position, rotation, scale)) {
		}
	}

	public static class SmartCombineUtilities {
		private class SmartSubmeshData {
			public Mesh mesh {get; private set;}
			public IList<CombineInstance> combineInstances {get; private set;}

			public SmartSubmeshData() {
				combineInstances = new List<CombineInstance>();
			}

			public void CombineSubmeshes() {
				if (mesh == null) mesh = new Mesh();
				else mesh.Clear();

				mesh.CombineMeshes(combineInstances.ToArray(), true, true);
			}
		}

		public static void CombineMeshesSmart(this Mesh mesh, SmartMeshData[] meshData, out Material[] materials) {
			IDictionary<Material, SmartSubmeshData> materialTable = new Dictionary<Material, SmartSubmeshData>();
			IList<CombineInstance> submeshCombineInstances = new List<CombineInstance>();

			foreach (SmartMeshData data in meshData) {
				IList<Material> meshMaterials = data.materials;
				for (int subMeshIndex = 0; subMeshIndex < data.mesh.subMeshCount; subMeshIndex++) {
					SmartSubmeshData submeshData = null;
					if (materialTable.ContainsKey(meshMaterials[subMeshIndex])) {
						submeshData = materialTable[meshMaterials[subMeshIndex]];
					} else {
						submeshData = new SmartSubmeshData();
						materialTable.Add(meshMaterials[subMeshIndex], submeshData);
					}

					CombineInstance combineInstance = new CombineInstance();
					combineInstance.mesh = data.mesh;
					combineInstance.subMeshIndex = subMeshIndex;
					combineInstance.transform = data.transform;

					submeshData.combineInstances.Add(combineInstance);
				}
			}

			foreach (SmartSubmeshData subMeshData in materialTable.Values) {
				subMeshData.CombineSubmeshes();

				CombineInstance combineInstance = new CombineInstance();
				combineInstance.mesh = subMeshData.mesh;
				combineInstance.subMeshIndex = 0;

				submeshCombineInstances.Add(combineInstance);
			}

			mesh.Clear();
			mesh.CombineMeshes(submeshCombineInstances.ToArray(), false, false);
			//mesh.Optimize();

			materials = materialTable.Keys.ToArray();
		}
	}
}