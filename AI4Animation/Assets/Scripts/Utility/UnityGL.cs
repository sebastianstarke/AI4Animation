using UnityEngine;

public static class UnityGL {

	private static Material SceneMaterial;
	private static Material GUIMaterial;

	private static Vector3[] IsocelesTrianglePoints;
	private static int CircleResolution = 10;
	private static Vector3[] CirclePoints;
	private static int SphereResolution = 10;
	private static Vector3[] SpherePoints;

	private static bool Drawing = false;

	private static bool Initialised = false;

	private static PROGRAM Program = PROGRAM.NONE;
	private static SPACE Space = SPACE.NONE;

	private enum PROGRAM {NONE, LINES, TRIANGLES, TRIANGLE_STRIP, QUADS};
	private enum SPACE {NONE, SCENE, GUI}

	private static float GUIOffset = 0.00001f;

	private static Vector3 ViewPosition = Vector3.zero;
	private static Quaternion ViewRotation = Quaternion.identity;

	static void Initialise() {
		if(Initialised) {
			return;
		}

		Resources.UnloadUnusedAssets();

		Shader colorShader = Shader.Find("Hidden/Internal-Colored");

		SceneMaterial = new Material(colorShader);
		SceneMaterial.hideFlags = HideFlags.HideAndDontSave;
		SceneMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
		SceneMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
		SceneMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
		SceneMaterial.SetInt("_ZWrite", 1);

		GUIMaterial = new Material(colorShader);
		GUIMaterial.hideFlags = HideFlags.HideAndDontSave;
		GUIMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
		GUIMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
		GUIMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
		GUIMaterial.SetInt("_ZWrite", 0);

		IsocelesTrianglePoints = new Vector3[3];
		CirclePoints = new Vector3[CircleResolution];
		SpherePoints = new Vector3[4 * SphereResolution * SphereResolution];

		Initialised = true;
	}

	static void UpdateData() {
		//Camera
		ViewPosition = GetCamera().transform.position;
		ViewRotation = GetCamera().transform.rotation;

		//Triangle
		IsocelesTrianglePoints[0] = ViewRotation * new Vector3(-0.5f, 0.5f, 0f);
		IsocelesTrianglePoints[1] = ViewRotation * new Vector3(0.5f, 0.5f, 0f);
		IsocelesTrianglePoints[2] = ViewRotation * new Vector3(0f, -0.5f, 0f);

		//Circle
		for(int i=0; i<CircleResolution; i++) {
			float angle = 2f * Mathf.PI * (float)i / ((float)CircleResolution-1f);
			CirclePoints[i] = ViewRotation * new Vector3(Mathf.Cos(angle), Mathf.Sin(angle), 0f);
		}

		//Sphere
		float startU=0;
		float startV=0;
		float endU=Mathf.PI*2;
		float endV=Mathf.PI;
		float stepU=(endU-startU)/SphereResolution; // step size between U-points on the grid
		float stepV=(endV-startV)/SphereResolution; // step size between V-points on the grid
		int index = 0;
		for(int i=0; i<SphereResolution; i++){ // U-points
			for(int j=0; j<SphereResolution; j++){ // V-points
				float u=i*stepU+startU;
				float v=j*stepV+startV;
				float un=(i+1==SphereResolution) ? endU : (i+1)*stepU+startU;
				float vn=(j+1==SphereResolution) ? endV : (j+1)*stepV+startV;
				// Find the four points of the grid
				// square by evaluating the parametric
				// surface function
				SpherePoints[index+0] = SphereVertex(u, v);
				SpherePoints[index+1] = SphereVertex(u, vn);
				SpherePoints[index+2] = SphereVertex(un, v);
				SpherePoints[index+3] = SphereVertex(un, vn);
				index += 4;
			}
		}
	}

	private static Vector3 SphereVertex(float u, float v) {
		return new Vector3(Mathf.Cos(u)*Mathf.Sin(v), Mathf.Cos(v), Mathf.Sin(u)*Mathf.Sin(v));
	}

	private static void SetProgram(PROGRAM program, SPACE space) {
		if(Program != program || Space != space) {
			GL.End();
			if(program == PROGRAM.NONE || space == SPACE.NONE) {
				Program = PROGRAM.NONE;
				Space = SPACE.NONE;
			} else {
				switch(space) {
					case SPACE.SCENE:
					SceneMaterial.SetPass(0);
					break;
					case SPACE.GUI:
					GUIMaterial.SetPass(0);
					break;
				}
				switch(program) {
					case PROGRAM.LINES:
					GL.Begin(GL.LINES);
					break;
					case PROGRAM.TRIANGLES:
					GL.Begin(GL.TRIANGLES);
					break;
					case PROGRAM.TRIANGLE_STRIP:
					GL.Begin(GL.TRIANGLE_STRIP);
					break;
					case PROGRAM.QUADS:
					GL.Begin(GL.QUADS);
					break;
				}
				Program = program;
				Space = space;
			}
		}
	}

	public static void Start() {
		if(Drawing) {
			Debug.Log("Drawing has not been finished yet.");
		} else {
			Initialise();
			UpdateData();
			Drawing = true;
		}
	}

	public static void Finish() {
		if(Drawing) {
			SetProgram(PROGRAM.NONE, SPACE.NONE);
			Drawing = false;
		} else {
			ViewPosition = Vector3.zero;
			ViewRotation = Quaternion.identity;
			Debug.Log("Drawing has not been started yet.");
		}
	}

	public static void DrawLine(Vector3 start, Vector3 end, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.LINES, SPACE.SCENE);
		GL.Color(color);
		GL.Vertex(start);
		GL.Vertex(end);
	}

	public static void DrawLine(Vector3 start, Vector3 end, Color startColor, Color endColor) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.LINES, SPACE.SCENE);
		GL.Color(startColor);
		GL.Vertex(start);
		GL.Color(endColor);
		GL.Vertex(end);
	}

    public static void DrawLine(Vector3 start, Vector3 end, float startWidth, float endWidth, Color startColor, Color endColor) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.QUADS, SPACE.SCENE);
		Vector3 dir = end-start;
		Vector3 orthoStart = startWidth/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir).normalized;
		Vector3 orthoEnd = endWidth/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir).normalized;

		GL.Color(startColor);
        GL.Vertex(start+orthoStart);
		GL.Vertex(start-orthoStart);
		GL.Color(endColor);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(end+orthoEnd);
    }

    public static void DrawLine(Vector3 start, Vector3 end, float startWidth, float endWidth, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.QUADS, SPACE.SCENE);
		Vector3 dir = end-start;
		Vector3 orthoStart = startWidth/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir).normalized;
		Vector3 orthoEnd = endWidth/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir).normalized;

		GL.Color(color);
        GL.Vertex(start+orthoStart);
		GL.Vertex(start-orthoStart);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(end+orthoEnd);
    }

    public static void DrawLine(Vector3 start, Vector3 end, float width, Color startColor, Color endColor) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.QUADS, SPACE.SCENE);
		Vector3 dir = end-start;
		Vector3 orthoStart = width/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir).normalized;
		Vector3 orthoEnd = width/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir).normalized;

		GL.Color(startColor);
        GL.Vertex(start+orthoStart);
		GL.Vertex(start-orthoStart);
		GL.Color(endColor);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(end+orthoEnd);
    }

    public static void DrawLine(Vector3 start, Vector3 end, float width, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.QUADS, SPACE.SCENE);
		Vector3 dir = end-start;
		Vector3 orthoStart = width/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir).normalized;
		Vector3 orthoEnd = width/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir).normalized;

		GL.Color(color);
        GL.Vertex(start+orthoStart);
		GL.Vertex(start-orthoStart);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(end+orthoEnd);
    }

	public static void DrawTriangle(Vector3 a, Vector3 b, Vector3 c, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.TRIANGLES, SPACE.SCENE);
        GL.Color(color);
        GL.Vertex(a);
		GL.Vertex(b);
		GL.Vertex(c);
	}

	public static void DrawIsocelesTriangle(Vector3 center, float radius, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.TRIANGLES, SPACE.SCENE);
        GL.Color(color);
        GL.Vertex(center + radius*IsocelesTrianglePoints[0]);
		GL.Vertex(center + radius*IsocelesTrianglePoints[1]);
		GL.Vertex(center + radius*IsocelesTrianglePoints[2]);
	}

	public static void DrawCircle(Vector3 center, float radius, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.TRIANGLES, SPACE.SCENE);
        GL.Color(color);
		for(int i=0; i<CircleResolution-1; i++) {
			GL.Vertex(center);
			GL.Vertex(center + radius*CirclePoints[i]);
			GL.Vertex(center + radius*CirclePoints[i+1]);
		}
	}

	public static void DrawSphere(Vector3 center, float radius, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.TRIANGLES, SPACE.SCENE);
        GL.Color(color);
		int index = 0;
		for(int i=0; i<SphereResolution; i++) {
			for(int j=0; j<SphereResolution; j++) {
				GL.Vertex(center + radius*SpherePoints[index+0]);
				GL.Vertex(center + radius*SpherePoints[index+2]);
				GL.Vertex(center + radius*SpherePoints[index+1]);
				GL.Vertex(center + radius*SpherePoints[index+3]);
				GL.Vertex(center + radius*SpherePoints[index+1]);
				GL.Vertex(center + radius*SpherePoints[index+2]);
				index += 4;
			}
		}
	}

	public static void DrawArrow(Vector3 start, Vector3 end, float tipPivot, float shaftWidth, float tipWidth, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		if(tipPivot < 0f || tipPivot > 1f) {
			Debug.Log("The tip pivot must be specified between 0 and 1.");
			return;
		}
		Vector3 pivot = start + tipPivot * (end-start);
		DrawLine(start, pivot, shaftWidth, color);
		DrawLine(pivot, end, tipWidth, 0f, color);
	}

	public static void DrawMesh(Mesh mesh, Vector3 position, Quaternion rotation, Vector3 scale, Material material) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.NONE, SPACE.SCENE);
		material.SetPass(0);
		Graphics.DrawMeshNow(mesh, Matrix4x4.TRS(position, rotation, scale));
	}

	//TODO FASTER
	public static void DrawGUILine(float xStart, float yStart, float xEnd, float yEnd, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		xStart *= Screen.width;
		yStart *= Screen.height;
		xEnd *= Screen.width;
		yEnd *= Screen.height;
		SetProgram(PROGRAM.LINES, SPACE.GUI);
		GL.Color(color);
		GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(xStart, yStart, GetCamera().nearClipPlane + GUIOffset)));
		GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(xEnd, yEnd, GetCamera().nearClipPlane + GUIOffset)));
	}

	//TODO FASTER
	public static void DrawGUIQuad(float x, float y, float width, float height, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		x *= Screen.width;
		y *= Screen.height;
		width *= Screen.width;
		height *= Screen.height;
		SetProgram(PROGRAM.QUADS, SPACE.GUI);
		GL.Color(color);
		GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(x, y, GetCamera().nearClipPlane + GUIOffset)));
		GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(x+width, y, GetCamera().nearClipPlane + GUIOffset)));
		GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(x+width, y+height, GetCamera().nearClipPlane + GUIOffset)));
		GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(x, y+height, GetCamera().nearClipPlane + GUIOffset)));
	}

	//TODO FASTER
	public static void DrawGUITriangle(float xA, float yA, float xB, float yB, float xC, float yC, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		xA *= Screen.width;
		yA *= Screen.height;
		xB *= Screen.width;
		yB *= Screen.height;
		xC *= Screen.width;
		yC *= Screen.height;
		SetProgram(PROGRAM.TRIANGLES, SPACE.GUI);
		GL.Color(color);
		GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(xA, yA, GetCamera().nearClipPlane + GUIOffset)));
		GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(xB, yB, GetCamera().nearClipPlane + GUIOffset)));
		GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(xC, yC, GetCamera().nearClipPlane + GUIOffset)));
	}

	//TODO FASTER
	public static void DrawGUICircle(float xCenter, float yCenter, float radius, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		xCenter *= Screen.width;
		yCenter *= Screen.height;
		radius *= Screen.height;
		int resolution = 30;
		SetProgram(PROGRAM.TRIANGLES, SPACE.GUI);
        GL.Color(color);
		Vector3 center = GetCamera().ScreenToWorldPoint(new Vector3(xCenter, yCenter, GetCamera().nearClipPlane + GUIOffset));
		for(int i=0; i<resolution-1; i++) {
			GL.Vertex(center);
			float a = 2f * Mathf.PI * (float)i / ((float)resolution-1f);
			GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(xCenter + Mathf.Cos(a)*radius, yCenter + Mathf.Sin(a)*radius, GetCamera().nearClipPlane + GUIOffset)));
			float b = 2f * Mathf.PI * (float)(i+1) / ((float)resolution-1f);
			GL.Vertex(GetCamera().ScreenToWorldPoint(new Vector3(xCenter + Mathf.Cos(b)*radius, yCenter + Mathf.Sin(b)*radius, GetCamera().nearClipPlane + GUIOffset)));

		}
	}
	
	private static Camera GetCamera() {
		if(Camera.current != null) {
			return Camera.current;
		} else {
			return Camera.main;
		}
	}

}