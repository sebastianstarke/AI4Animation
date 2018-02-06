using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public static class DrawingLibrary {

	private static Material SceneMaterial;

	private static bool Drawing = false;

	private static float GUIOffset = 0.001f;

	private static Vector3[] CircleWire;
	private static Vector3[] QuadWire;
	private static Vector3[] CubeWire;
	private static Vector3[] SphereWire;
	private static Vector3[] CylinderWire;
	private static Vector3[] CapsuleWire;

	private static Mesh CircleMesh;
	private static Mesh QuadMesh;
	private static Mesh CubeMesh;
	private static Mesh SphereMesh;
	private static Mesh CylinderMesh;
	private static Mesh CapsuleMesh;

	private static Dictionary<PrimitiveType, Mesh> PrimitiveMeshes = new Dictionary<PrimitiveType, Mesh>();

	private static int Resolution = 30;

	private static Mesh Initialised;

	//------------------------------------------------------------------------------------------
	//2D SCENE DRAWING FUNCTIONS
	//------------------------------------------------------------------------------------------
	public static void DrawLine(Vector3 start, Vector3 end, Color color) {
		Initialise();
		GL.Begin(GL.LINES);
		SceneMaterial.color = color;
		SceneMaterial.SetPass(0);
		GL.Vertex(start);
		GL.Vertex(end);
		GL.End();
	}

    public static void DrawLine(Vector3 start, Vector3 end, float width, Color color) {
		Initialise();
		Vector3 viewPosition = GetCamera().transform.position;
		GL.Begin(GL.QUADS);
		SceneMaterial.color = color;
		SceneMaterial.SetPass(0);
		Vector3 dir = (end-start).normalized;
		Vector3 orthoStart = width/2f * (Quaternion.AngleAxis(90f, (start - viewPosition)) * dir);
		Vector3 orthoEnd = width/2f * (Quaternion.AngleAxis(90f, (end - viewPosition)) * dir);
		GL.Vertex(end+orthoEnd);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(start-orthoStart);
		GL.Vertex(start+orthoStart);
		GL.End();
    }

    public static void DrawLine(Vector3 start, Vector3 end, float startWidth, float endWidth, Color color) {
		Initialise();
		Vector3 viewPosition = GetCamera().transform.position;
		GL.Begin(GL.QUADS);
		SceneMaterial.color = color;
		SceneMaterial.SetPass(0);
		Vector3 dir = (end-start).normalized;
		Vector3 orthoStart = startWidth/2f * (Quaternion.AngleAxis(90f, (start - viewPosition)) * dir);
		Vector3 orthoEnd = endWidth/2f * (Quaternion.AngleAxis(90f, (end - viewPosition)) * dir);
		GL.Vertex(end+orthoEnd);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(start-orthoStart);
		GL.Vertex(start+orthoStart);
		GL.End();
    }

	public static void DrawLines(Vector3[] points, Color color) {
		Initialise();
		GL.Begin(GL.LINE_STRIP);
		SceneMaterial.color = color;
		SceneMaterial.SetPass(0);
		for(int i=0; i<points.Length; i++) {
			GL.Vertex(points[i]);
		}
		GL.End();
	}

    public static void DrawLines(Vector3[] points, float width, Color color) {
		Initialise();
		Vector3 viewPosition = GetCamera().transform.position;
		GL.Begin(GL.QUADS);
		SceneMaterial.color = color;
		SceneMaterial.SetPass(0);
		for(int i=1; i<points.Length; i++) {
			Vector3 start = points[i-1];
			Vector3 end = points[i];
			Vector3 dir = (end-start).normalized;
			Vector3 orthoStart = width/2f * (Quaternion.AngleAxis(90f, (start - viewPosition)) * dir);
			Vector3 orthoEnd = width/2f * (Quaternion.AngleAxis(90f, (end - viewPosition)) * dir);
			GL.Vertex(end+orthoEnd);
			GL.Vertex(end-orthoEnd);
			GL.Vertex(start-orthoStart);
			GL.Vertex(start+orthoStart);
		}
		GL.End();
    }

	public static void DrawTriangle(Vector3 a, Vector3 b, Vector3 c, Color color) {
		Initialise();
		GL.Begin(GL.TRIANGLES);
		SceneMaterial.color = color;
		SceneMaterial.SetPass(0);
        GL.Vertex(a);
		GL.Vertex(b);
		GL.Vertex(c);
		GL.End();
	}

	public static void DrawCircle(Vector3 position, float size, Color color) {
		Initialise();
		DrawMesh(CircleMesh, position, GetCamera().transform.rotation, size*Vector3.one, SceneMaterial, color);
	}

	public static void DrawCircle(Vector3 position, Quaternion rotation, float size, Color color) {
		Initialise();
		DrawMesh(CircleMesh, position, rotation, size*Vector3.one, SceneMaterial, color);
	}

	public static void DrawWireCircle(Vector3 position, float size, Color color) {
		DrawWireStrip(CircleWire, position, GetCamera().transform.rotation, size*Vector3.one, SceneMaterial, color);
	}

	public static void DrawWireCircle(Vector3 position, Quaternion rotation, float size, Color color) {
		DrawWireStrip(CircleWire, position, rotation, size*Vector3.one, SceneMaterial, color);
	}

	public static void DrawWiredCircle(Vector3 position, float size, Color circleColor, Color wireColor) {
		DrawCircle(position, size, circleColor);
		DrawWireCircle(position, size, wireColor);
	}

	public static void DrawWiredCircle(Vector3 position, Quaternion rotation, float size, Color circleColor, Color wireColor) {
		DrawCircle(position, rotation, size, circleColor);
		DrawWireCircle(position, rotation, size, wireColor);
	}

	public static void DrawArrow(Vector3 start, Vector3 end, float tipPivot, float shaftWidth, float tipWidth, Color color) {
		Initialise();
		if(tipPivot < 0f || tipPivot > 1f) {
			Debug.Log("The tip pivot must be specified between 0 and 1.");
			tipPivot = Mathf.Clamp(tipPivot, 0f, 1f);
		}
		Vector3 pivot = start + tipPivot * (end-start);
		DrawLine(start, pivot, shaftWidth, color);
		DrawLine(pivot, end, tipWidth, 0f, color);
	}

	public static void DrawArrow(Vector3 start, Vector3 end, float tipPivot, float shaftWidth, float tipWidth, Color shaftColor, Color tipColor) {
		Initialise();
		if(tipPivot < 0f || tipPivot > 1f) {
			Debug.Log("The tip pivot must be specified between 0 and 1.");
			tipPivot = Mathf.Clamp(tipPivot, 0f, 1f);
		}
		Vector3 pivot = start + tipPivot * (end-start);
		DrawLine(start, pivot, shaftWidth, shaftColor);
		DrawLine(pivot, end, tipWidth, 0f, tipColor);
	}

	//------------------------------------------------------------------------------------------
	//3D SCENE DRAWING FUNCTIONS
	//------------------------------------------------------------------------------------------
	public static void DrawQuad(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		Initialise();
		DrawMesh(QuadMesh, position, rotation, new Vector3(width, height, 1f), SceneMaterial, color);
	}

	public static void DrawQuadWire(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		Initialise();
		DrawWireStrip(QuadWire, position, rotation, new Vector3(width, height, 1f), SceneMaterial, color);
	}

	public static void DrawWiredQuad(Vector3 position, Quaternion rotation, float width, float height, Color quadColor, Color wireColor) {
		DrawQuad(position, rotation, width, height, quadColor);
		DrawQuadWire(position, rotation, width, height, wireColor);
	}

	public static void DrawCube(Vector3 position, Quaternion rotation, float size, Color color) {
		Initialise();
		DrawMesh(CubeMesh, position, rotation, size*Vector3.one, SceneMaterial, color);
	}

	public static void DrawCubeWire(Vector3 position, Quaternion rotation, float size, Color color) {
		Initialise();
		DrawWireLines(CubeWire, position, rotation, size*Vector3.one, SceneMaterial, color);
	}

	public static void DrawWiredCube(Vector3 position, Quaternion rotation, float size, Color cubeColor, Color wireColor) {
		DrawCube(position, rotation, size, cubeColor);
		DrawCubeWire(position, rotation, size, wireColor);
	}

	public static void DrawCuboid(Vector3 position, Quaternion rotation, Vector3 size, Color color) {
		Initialise();
		DrawMesh(CubeMesh, position, rotation, size, SceneMaterial, color);
	}

	public static void DrawCuboidWire(Vector3 position, Quaternion rotation, Vector3 size, Color color) {
		Initialise();
		DrawWireLines(CubeWire, position, rotation, size, SceneMaterial, color);
	}

	public static void DrawWiredCuboid(Vector3 position, Quaternion rotation, Vector3 size, Color cuboidColor, Color wireColor) {
		DrawCuboid(position, rotation, size, cuboidColor);
		DrawCuboidWire(position, rotation, size, wireColor);
	}

	public static void DrawSphere(Vector3 position, float size, Color color) {
		Initialise();
		DrawMesh(SphereMesh, position, Quaternion.identity, size*Vector3.one, SceneMaterial, color);
	}

	public static void DrawWireSphere(Vector3 position, float size, Color color) {
		Initialise();
		DrawWireLines(SphereWire, position, Quaternion.identity, size*Vector3.one, SceneMaterial, color);
	}

	public static void DrawWiredSphere(Vector3 position, float size, Color sphereColor, Color wireColor) {
		DrawSphere(position, size, sphereColor);
		DrawWireSphere(position, size, wireColor);
	}

	public static void DrawCylinder(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		Initialise();
		DrawMesh(CylinderMesh, position, rotation, new Vector3(width, height/2f, width), SceneMaterial, color);
	}

	public static void DrawWireCylinder(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		Initialise();
		DrawWireLines(CylinderWire, position, rotation, new Vector3(width, height/2f, width), SceneMaterial, color);
	}

	public static void DrawWiredCylinder(Vector3 position, Quaternion rotation, float width, float height, Color cylinderColor, Color wireColor) {
		DrawCylinder(position, rotation, width, height, cylinderColor);
		DrawWireCylinder(position, rotation, width, height, wireColor);
	}

	public static void DrawCapsule(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		Initialise();
		DrawMesh(CapsuleMesh, position, rotation, new Vector3(width, height/2f, width), SceneMaterial, color);
	}

	public static void DrawWireCapsule(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		Initialise();
		DrawWireLines(CapsuleWire, position, rotation, new Vector3(width, height/2f, width), SceneMaterial, color);
	}

	public static void DrawWiredCapsule(Vector3 position, Quaternion rotation, float width, float height, Color capsuleColor, Color wireColor) {
		DrawCapsule(position, rotation, width, height, capsuleColor);
		DrawWireCapsule(position, rotation, width, height, wireColor);
	}

	public static void DrawMesh(Mesh mesh, Vector3 position, Quaternion rotation, Vector3 scale, Material material, Color color) {
		Initialise();
		material.color = color;
		material.SetPass(0);
		Graphics.DrawMeshNow(mesh, Matrix4x4.TRS(position, rotation, scale));
	}

	//------------------------------------------------------------------------------------------
	//GUI DRAWING FUNCTIONS
	//------------------------------------------------------------------------------------------
	
	//------------------------------------------------------------------------------------------
	//UTILITY FUNCTIONS
	//------------------------------------------------------------------------------------------
	static void Initialise() {
		if(Initialised != null) {
			return;
		}

		Resources.UnloadUnusedAssets();

		Shader xrayShader = Shader.Find("UnityGL/Transparent");

		SceneMaterial = new Material(xrayShader);
		SceneMaterial.hideFlags = HideFlags.HideAndDontSave;
		SceneMaterial.SetFloat("_Power", 0f);

		//Meshes
		CircleMesh = CreateCircle(Resolution);
		QuadMesh = GetPrimitiveMesh(PrimitiveType.Quad);
		CubeMesh = GetPrimitiveMesh(PrimitiveType.Cube);
		SphereMesh = GetPrimitiveMesh(PrimitiveType.Sphere);
		CylinderMesh = GetPrimitiveMesh(PrimitiveType.Cylinder);
		CapsuleMesh = GetPrimitiveMesh(PrimitiveType.Capsule);
		//

		//Wires
		CircleWire = CreateCircleWire(Resolution);
		QuadWire = CreateQuadWire();
		CubeWire = CreateCubeWire();
		SphereWire = CreateSphereWire(Resolution);
		CylinderWire = CreateCylinderWire(Resolution);
		CapsuleWire = CreateCapsuleWire(Resolution);
		//
		
		Initialised = new Mesh();
	}

	private static Camera GetCamera() {
		if(Camera.current != null) {
			return Camera.current;
		} else {
			return Camera.main;
		}
	}

	private static Mesh GetPrimitiveMesh(PrimitiveType type) {
		if(!PrimitiveMeshes.ContainsKey(type)) {
			GameObject gameObject = GameObject.CreatePrimitive(type);
			gameObject.GetComponent<MeshRenderer>().enabled = false;
			Mesh mesh = gameObject.GetComponent<MeshFilter>().sharedMesh;
			if(Application.isPlaying) {
				GameObject.Destroy(gameObject);
			} else {
				GameObject.DestroyImmediate(gameObject);
			}
			PrimitiveMeshes[type] = mesh;
			return mesh;
		}
		return PrimitiveMeshes[type];
	}
	
	private static Mesh CreateCircle(int resolution) {
		float step = 360.0f / (float)resolution;
		List<Vector3> vertexList = new List<Vector3>();
		List<int> triangleList = new List<int>();
		Quaternion quaternion = Quaternion.Euler(0.0f, 0.0f, step);
		// Make first triangle.
		vertexList.Add(new Vector3(0.0f, 0.0f, 0.0f));  // 1. Circle center.
		vertexList.Add(new Vector3(0.0f, 0.5f, 0.0f));  // 2. First vertex on circle outline (radius = 0.5f)
		vertexList.Add(quaternion * vertexList[1]);     // 3. First vertex on circle outline rotated by angle)
		// Add triangle indices.
		triangleList.Add(1);
		triangleList.Add(0);
		triangleList.Add(2);
		for(int i=0; i<resolution-1; i++) {
			triangleList.Add(vertexList.Count - 1);
			triangleList.Add(0);                      // Index of circle center.
			triangleList.Add(vertexList.Count);
			vertexList.Add(quaternion * vertexList[vertexList.Count - 1]);
		}
		Mesh mesh = new Mesh();
		mesh.vertices = vertexList.ToArray();
		mesh.triangles = triangleList.ToArray();        
		return mesh;
	}

	private static Vector3[] CreateCircleWire(int resolution) {
		List<Vector3> points = new List<Vector3>();
		float step = 360.0f / (float)resolution;
		for(int i=0; i<resolution; i++) {
			points.Add(Quaternion.Euler(0f, 0f, i*step) * new Vector3(0f, 0.5f, 0f));
			points.Add(Quaternion.Euler(0f, 0f, (i+1)*step) * new Vector3(0f, 0.5f, 0f));
		}
		return points.ToArray();
	}

	private static Vector3[] CreateQuadWire() {
		List<Vector3> points = new List<Vector3>();
		points.Add(new Vector3(-0.5f, -0.5f, 0f));
		points.Add(new Vector3(0.5f, -0.5f, 0f));
		points.Add(new Vector3(0.5f, 0.5f, 0f));
		points.Add(new Vector3(-0.5f, 0.5f, 0f));
		points.Add(new Vector3(-0.5f, -0.5f, 0f));
		return points.ToArray();
	}

	private static Vector3[] CreateCubeWire() {
		float size = 1f;
		Vector3 A = new Vector3(-size/2f, -size/2f, -size/2f);
		Vector3 B = new Vector3(size/2f, -size/2f, -size/2f);
		Vector3 C = new Vector3(-size/2f, -size/2f, size/2f);
		Vector3 D = new Vector3(size/2f, -size/2f, size/2f);
		Vector3 p1 = A; Vector3 p2 = B;
		Vector3 p3 = C; Vector3 p4 = D;
		Vector3 p5 = -D; Vector3 p6 = -C;
		Vector3 p7 = -B; Vector3 p8 = -A;

		List<Vector3> points = new List<Vector3>();
		points.Add(p1); points.Add(p2);
		points.Add(p2); points.Add(p4);
		points.Add(p4); points.Add(p3);
		points.Add(p3); points.Add(p1);
		
		points.Add(p5); points.Add(p6);
		points.Add(p6); points.Add(p8);
		points.Add(p5); points.Add(p7);
		points.Add(p7); points.Add(p8);

		points.Add(p1); points.Add(p5);
		points.Add(p2); points.Add(p6);
		points.Add(p3); points.Add(p7);
		points.Add(p4); points.Add(p8);
		return points.ToArray();
	}

	private static Vector3[] CreateSphereWire(int resolution) {
		List<Vector3> points = new List<Vector3>();
		float step = 360.0f / (float)resolution;
		for(int i=0; i<resolution; i++) {
			points.Add(Quaternion.Euler(0f, 0f, i*step) * new Vector3(0f, 0.5f, 0f));
			points.Add(Quaternion.Euler(0f, 0f, (i+1)*step) * new Vector3(0f, 0.5f, 0f));
		}
		for(int i=0; i<resolution; i++) {
			points.Add(Quaternion.Euler(0f, i*step, 0f) * new Vector3(0f, 0f, 0.5f));
			points.Add(Quaternion.Euler(0f, (i+1)*step, 0f) * new Vector3(0f, 0f, 0.5f));
		}
		for(int i=0; i<resolution; i++) {
			points.Add(Quaternion.Euler(i*step, 0f, i*step) * new Vector3(0f, 0f, 0.5f));
			points.Add(Quaternion.Euler((i+1)*step, 0f, (i+1)*step) * new Vector3(0f, 0f, 0.5f));
		}
		return points.ToArray();
	}

	private static Vector3[] CreateCylinderWire(int resolution) {
		List<Vector3> points = new List<Vector3>();
		float step = 360.0f / (float)resolution;
		for(int i=0; i<resolution; i++) {
			points.Add(Quaternion.Euler(0f, i*step, 0f) * new Vector3(0f, 0f, 0.5f) + new Vector3(0f, 1f, 0f));
			points.Add(Quaternion.Euler(0f, (i+1)*step, 0f) * new Vector3(0f, 0f, 0.5f) + new Vector3(0f, 1f, 0f));
		}
		for(int i=0; i<resolution; i++) {
			points.Add(Quaternion.Euler(0f, i*step, 0f) * new Vector3(0f, 0f, 0.5f) - new Vector3(0f, 1f, 0f));
			points.Add(Quaternion.Euler(0f, (i+1)*step, 0f) * new Vector3(0f, 0f, 0.5f) - new Vector3(0f, 1f, 0f));
		}
		points.Add(new Vector3(0f, -1f, -0.5f));
		points.Add(new Vector3(0f, 1f, -0.5f));
		points.Add(new Vector3(0f, -1f, 0.5f));
		points.Add(new Vector3(0f, 1f, 0.5f));
		points.Add(new Vector3(-0.5f, -1f, 0f));
		points.Add(new Vector3(-0.5f, 1f, 0f));
		points.Add(new Vector3(0.5f, -1f, 0f));
		points.Add(new Vector3(0.5f, 1f, 0));
		return points.ToArray();
	}

	private static Vector3[] CreateCapsuleWire(int resolution) {
		List<Vector3> points = new List<Vector3>();
		float step = 360.0f / (float)resolution;
		for(int i=-resolution/4-1; i<=resolution/4; i++) {
			points.Add(Quaternion.Euler(0f, 0f, i*step) * new Vector3(0f, 0.5f, 0f) + new Vector3(0f, 0.5f, 0f));
			points.Add(Quaternion.Euler(0f, 0f, (i+1)*step) * new Vector3(0f, 0.5f, 0f) + new Vector3(0f, 0.5f, 0f));
		}
		for(int i=resolution/2; i<resolution; i++) {
			points.Add(Quaternion.Euler(i*step, 0f, i*step) * new Vector3(0f, 0f, 0.5f) + new Vector3(0f, 0.5f, 0f));
			points.Add(Quaternion.Euler((i+1)*step, 0f, (i+1)*step) * new Vector3(0f, 0f, 0.5f) + new Vector3(0f, 0.5f, 0f));
		}
		for(int i=-resolution/4-1; i<=resolution/4; i++) {
			points.Add(Quaternion.Euler(0f, 0f, i*step) * new Vector3(0f, -0.5f, 0f) + new Vector3(0f, -0.5f, 0f));
			points.Add(Quaternion.Euler(0f, 0f, (i+1)*step) * new Vector3(0f, -0.5f, 0f) + new Vector3(0f, -0.5f, 0f));
		}
		for(int i=resolution/2; i<resolution; i++) {
			points.Add(Quaternion.Euler(i*step, 0f, i*step) * new Vector3(0f, 0f, -0.5f) + new Vector3(0f, -0.5f, 0f));
			points.Add(Quaternion.Euler((i+1)*step, 0f, (i+1)*step) * new Vector3(0f, 0f, -0.5f) + new Vector3(0f, -0.5f, 0f));
		}
		points.Add(new Vector3(0f, -0.5f, -0.5f));
		points.Add(new Vector3(0f, 0.5f, -0.5f));
		points.Add(new Vector3(0f, -0.5f, 0.5f));
		points.Add(new Vector3(0f, 0.5f, 0.5f));
		points.Add(new Vector3(-0.5f, -0.5f, 0f));
		points.Add(new Vector3(-0.5f, 0.5f, 0f));
		points.Add(new Vector3(0.5f, -0.5f, 0f));
		points.Add(new Vector3(0.5f, 0.5f, 0));
		return points.ToArray();
	}

	private static void DrawWireStrip(Vector3[] points, Vector3 position, Quaternion rotation, Vector3 scale, Material material, Color color) {
		Initialise();
		GL.Begin(GL.LINE_STRIP);
		material.color = color;
		material.SetPass(0);
		for(int i=0; i<points.Length; i++) {
			GL.Vertex(position + rotation * Vector3.Scale(scale, points[i]));
		}
		GL.End();
	}

	private static void DrawWireLines(Vector3[] points, Vector3 position, Quaternion rotation, Vector3 scale, Material material, Color color) {
		Initialise();
		GL.Begin(GL.LINES);
		material.color = color;
		material.SetPass(0);
		for(int i=0; i<points.Length; i+=2) {
			GL.Vertex(position + rotation * Vector3.Scale(scale, points[i]));
			GL.Vertex(position + rotation * Vector3.Scale(scale, points[i+1]));
		}
		GL.End();
	}

}
