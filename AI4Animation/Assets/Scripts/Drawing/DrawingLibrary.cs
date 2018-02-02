using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public static class DrawingLibrary {

	private static Material SceneMaterial;

	private static bool Drawing = false;

	private static bool Initialised = false;

	private static float GUIOffset = 0.001f;

	private static Vector3 ViewPosition = Vector3.zero;
	private static Quaternion ViewRotation = Quaternion.identity;

	private static Mesh QuadMesh;
	private static Mesh CubeMesh;
	private static Mesh SphereMesh;
	private static Mesh CylinderMesh;
	private static Mesh CapsuleMesh;

	private static Dictionary<PrimitiveType, Mesh> PrimitiveMeshes = new Dictionary<PrimitiveType, Mesh>();

	static void Initialise() {
		if(Initialised) {
			return;
		}

		Resources.UnloadUnusedAssets();

		Shader xrayShader = Shader.Find("UnityGL/Transparent");

		SceneMaterial = new Material(xrayShader);
		SceneMaterial.hideFlags = HideFlags.HideAndDontSave;
		SceneMaterial.SetFloat("_Power", 0.1f);

		//Meshes
		QuadMesh = GetPrimitiveMesh(PrimitiveType.Quad);
		CubeMesh = GetPrimitiveMesh(PrimitiveType.Cube);
		SphereMesh = GetPrimitiveMesh(PrimitiveType.Sphere);
		CylinderMesh = GetPrimitiveMesh(PrimitiveType.Cylinder);
		CapsuleMesh = GetPrimitiveMesh(PrimitiveType.Capsule);

		Initialised = true;
	}

	//------------------------------------------------------------------------------------------
	//2D SCENE DRAWING FUNCTIONS
	//------------------------------------------------------------------------------------------
	public static void DrawLine(Vector3 start, Vector3 end, Color color) {

	}


	public static void DrawTriangle(Vector3 a, Vector3 b, Vector3 c, Color color) {

	}

	public static void DrawQuad(Vector3 center, float width, float height, Color color) {
		Initialise();
		//SceneMaterial.color = color;
		//DrawMesh(CubeMesh, position, rotation, size*Vector3.one, SceneMaterial);
	}

	public static void DrawCircle(Vector3 center, float radius, Color color) {

	}

	//------------------------------------------------------------------------------------------
	//3D SCENE DRAWING FUNCTIONS
	//------------------------------------------------------------------------------------------
	public static void DrawCube(Vector3 position, Quaternion rotation, float size, Color color) {
		Initialise();
		SceneMaterial.color = color;
		DrawMesh(CubeMesh, position, rotation, size*Vector3.one, SceneMaterial);
	}

	public static void DrawCuboid(Vector3 position, Quaternion rotation, Vector3 scale, Color color) {
		Initialise();
		SceneMaterial.color = color;
		DrawMesh(CubeMesh, position, rotation, scale, SceneMaterial);
	}

	public static void DrawSphere(Vector3 position, float size, Color color) {
		Initialise();
		SceneMaterial.color = color;
		DrawMesh(SphereMesh, position, Quaternion.identity, size*Vector3.one, SceneMaterial);
	}

	public static void DrawCylinder(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		Initialise();
		SceneMaterial.color = color;
		DrawMesh(CylinderMesh, position, rotation, new Vector3(width, height/2f, width), SceneMaterial);
	}

	public static void DrawCapsule(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		Initialise();
		SceneMaterial.color = color;
		DrawMesh(CapsuleMesh, position, rotation, new Vector3(width, height/2f, width), SceneMaterial);
	}

	public static void DrawMesh(Mesh mesh, Vector3 position, Quaternion rotation, Vector3 scale, Material material) {
		Initialise();
		material.SetPass(0);
		Graphics.DrawMeshNow(mesh, Matrix4x4.TRS(position, rotation, scale));
	}

	//------------------------------------------------------------------------------------------
	//GUI DRAWING FUNCTIONS
	//------------------------------------------------------------------------------------------
	
	//------------------------------------------------------------------------------------------
	//UTILITY FUNCTIONS
	//------------------------------------------------------------------------------------------
	private static Camera GetCamera() {
		if(Camera.current != null) {
			return Camera.current;
		} else {
			return Camera.main;
		}
	}

	public static Mesh GetPrimitiveMesh(PrimitiveType type) {
		if(!primitiveMeshes.ContainsKey(type)) {
			CreatePrimitiveMesh(type);
		}
		return primitiveMeshes[type];
	}

	private static Mesh CreatePrimitiveMesh(PrimitiveType type) {
		GameObject gameObject = GameObject.CreatePrimitive(type);
		gameObject.GetComponent<MeshRenderer>().enabled = false;
		Mesh mesh = gameObject.GetComponent<MeshFilter>().sharedMesh;
		Destroy(gameObject);

		primitiveMeshes[type] = mesh;
		return mesh;
	}

}
