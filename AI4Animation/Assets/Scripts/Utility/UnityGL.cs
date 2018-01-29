using UnityEngine;

public static class UnityGL {

	private static Material SceneMaterial;
	private static Material GUIMaterial;

	private static bool Drawing = false;

	private static bool Initialised = false;

	private static PROGRAM Program = PROGRAM.NONE;

	private enum PROGRAM {NONE, LINES, TRIANGLES, TRIANGLE_STRIP, QUADS};
	private enum SPACE {NONE, SCENE, GUI}

	private static float GUIOffset = 0.001f;

	private static Vector3 ViewPosition = Vector3.zero;
	private static Quaternion ViewRotation = Quaternion.identity;

	private static Vector3[] IsocelesTrianglePoints;
	private static int CircleResolution = 10;
	private static Vector3[] CirclePoints;
	private static int SphereResolution = 10;
	private static Vector3[] SpherePoints;

	private static Material CurrentMaterial;

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

		//Isoceles Triangle
		IsocelesTrianglePoints = new Vector3[3];
		IsocelesTrianglePoints[0] = new Vector3(-0.5f, 0.5f, 0f);
		IsocelesTrianglePoints[1] = new Vector3(0.5f, 0.5f, 0f);
		IsocelesTrianglePoints[2] = new Vector3(0f, -0.5f, 0f);

		//Circle
		CirclePoints = new Vector3[CircleResolution];
		for(int i=0; i<CircleResolution; i++) {
			float angle = 2f * Mathf.PI * (float)i / ((float)CircleResolution-1f);
			CirclePoints[i] = CircleVertex(angle);
		}

		//Sphere
		SpherePoints = new Vector3[4 * SphereResolution * SphereResolution];
		float startU=0;
		float startV=0;
		float endU=Mathf.PI*2;
		float endV=Mathf.PI;
		float stepU=(endU-startU)/SphereResolution;
		float stepV=(endV-startV)/SphereResolution;
		int index = 0;
		for(int i=0; i<SphereResolution; i++) {
			for(int j=0; j<SphereResolution; j++) {
				float u=i*stepU+startU;
				float v=j*stepV+startV;
				float un=(i+1==SphereResolution) ? endU : (i+1)*stepU+startU;
				float vn=(j+1==SphereResolution) ? endV : (j+1)*stepV+startV;
				SpherePoints[index+0] = SphereVertex(u, v);
				SpherePoints[index+1] = SphereVertex(u, vn);
				SpherePoints[index+2] = SphereVertex(un, v);
				SpherePoints[index+3] = SphereVertex(un, vn);
				index += 4;
			}
		}

		Initialised = true;
	}

	private static Vector3 CircleVertex(float angle) {
 		return new Vector3(Mathf.Cos(angle), Mathf.Sin(angle), 0f);
	}

	private static Vector3 SphereVertex(float u, float v) {
		return new Vector3(Mathf.Cos(u)*Mathf.Sin(v), Mathf.Cos(v), Mathf.Sin(u)*Mathf.Sin(v));
	}

	private static void SetProgram(PROGRAM program, Material material) {
		if(Program != program || CurrentMaterial != material) {
			GL.End();
			Program = program;
			CurrentMaterial = material;
			if(Program != PROGRAM.NONE) {
				CurrentMaterial.SetPass(0);
				switch(Program) {
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
			}
		}
	}

	public static void Start() {
		if(Drawing) {
			Debug.Log("Drawing has not been finished yet.");
		} else {
			Initialise();
			ViewPosition = GetCamera().transform.position;
			ViewRotation = GetCamera().transform.rotation;
			Drawing = true;
		}
	}

	public static void Finish() {
		if(Drawing) {
			SetProgram(PROGRAM.NONE, null);
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
		SetProgram(PROGRAM.LINES, SceneMaterial);
		GL.Color(color);
		GL.Vertex(start);
		GL.Vertex(end);
	}

	public static void DrawLine(Vector3 start, Vector3 end, Color startColor, Color endColor) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.LINES, SceneMaterial);
		GL.Color(startColor);
		GL.Vertex(start);
		GL.Color(endColor);
		GL.Vertex(end);
	}

    public static void DrawLine(Vector3 start, Vector3 end, float width, Color color, Material material = null) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.QUADS, material == null ? SceneMaterial : material);
		Vector3 dir = (end-start).normalized;
		Vector3 orthoStart = width/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir);
		Vector3 orthoEnd = width/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir);
		
		GL.Color(color);
        GL.Vertex(start+orthoStart);
		GL.Vertex(start-orthoStart);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(end+orthoEnd);
    }

    public static void DrawLine(Vector3 start, Vector3 end, float startWidth, float endWidth, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.QUADS, SceneMaterial);
		Vector3 dir = (end-start).normalized;
		Vector3 orthoStart = startWidth/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir);
		Vector3 orthoEnd = endWidth/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir);

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
		SetProgram(PROGRAM.QUADS, SceneMaterial);
		Vector3 dir = (end-start).normalized;
		Vector3 orthoStart = width/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir);
		Vector3 orthoEnd = width/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir);

		GL.Color(startColor);
        GL.Vertex(start+orthoStart);
		GL.Vertex(start-orthoStart);
		GL.Color(endColor);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(end+orthoEnd);
    }

    public static void DrawLine(Vector3 start, Vector3 end, float startWidth, float endWidth, Color startColor, Color endColor) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.QUADS, SceneMaterial);
		Vector3 dir = (end-start).normalized;
		Vector3 orthoStart = startWidth/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir);
		Vector3 orthoEnd = endWidth/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir);

		GL.Color(startColor);
        GL.Vertex(start+orthoStart);
		GL.Vertex(start-orthoStart);
		GL.Color(endColor);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(end+orthoEnd);
    }


	public static void DrawTriangle(Vector3 a, Vector3 b, Vector3 c, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.TRIANGLES, SceneMaterial);
        GL.Color(color);
        GL.Vertex(a);
		GL.Vertex(b);
		GL.Vertex(c);
	}

	public static void DrawQuad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.QUADS, SceneMaterial);
        GL.Color(color);
        GL.Vertex(a);
		GL.Vertex(c);
		GL.Vertex(d);
		GL.Vertex(b);
	}

	public static void DrawIsocelesTriangle(Vector3 center, float radius, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.TRIANGLES, SceneMaterial);
        GL.Color(color);
        GL.Vertex(center + radius*(ViewRotation*IsocelesTrianglePoints[0]));
		GL.Vertex(center + radius*(ViewRotation*IsocelesTrianglePoints[1]));
		GL.Vertex(center + radius*(ViewRotation*IsocelesTrianglePoints[2]));
	}

	public static void DrawCircle(Vector3 center, float radius, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.TRIANGLES, SceneMaterial);
        GL.Color(color);
		for(int i=0; i<CircleResolution-1; i++) {
			GL.Vertex(center);
			GL.Vertex(center + radius*(ViewRotation*CirclePoints[i]));
			GL.Vertex(center + radius*(ViewRotation*CirclePoints[i+1]));
		}
	}

	public static void DrawSphere(Vector3 center, float radius, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.TRIANGLES, SceneMaterial);
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
		SetProgram(PROGRAM.NONE, SceneMaterial);
		material.SetPass(0);
		Graphics.DrawMeshNow(mesh, Matrix4x4.TRS(position, rotation, scale));
	}

	public static void DrawGUILine(float xStart, float yStart, float xEnd, float yEnd, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		xStart *= Screen.width;
		yStart *= Screen.height;
		xEnd *= Screen.width;
		yEnd *= Screen.height;
		Camera camera = GetCamera();
		Vector3 start = camera.ScreenToWorldPoint(new Vector3(xStart, yStart, camera.nearClipPlane + GUIOffset));
		Vector3 end = camera.ScreenToWorldPoint(new Vector3(xEnd, yEnd, camera.nearClipPlane + GUIOffset));
		SetProgram(PROGRAM.LINES, GUIMaterial);
		GL.Color(color);
		GL.Vertex(start);
		GL.Vertex(end);
	}

    public static void DrawGUILine(float xStart, float yStart, float xEnd, float yEnd, float width, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		xStart *= Screen.width;
		yStart *= Screen.height;
		xEnd *= Screen.width;
		yEnd *= Screen.height;
		width /= Screen.width;
		Camera camera = GetCamera();
		Vector3 start = camera.ScreenToWorldPoint(new Vector3(xStart, yStart, camera.nearClipPlane + GUIOffset));
		Vector3 end = camera.ScreenToWorldPoint(new Vector3(xEnd, yEnd, camera.nearClipPlane + GUIOffset));
		SetProgram(PROGRAM.QUADS, GUIMaterial);
		Vector3 dir = end-start;
		Vector3 ortho = width/2f * (Quaternion.AngleAxis(90f, camera.transform.forward) * dir).normalized;
		GL.Color(color);
        GL.Vertex(start+ortho);
		GL.Vertex(start-ortho);
		GL.Vertex(end-ortho);
		GL.Vertex(end+ortho);
    }

	public static void DrawGUIQuad(float x, float y, float width, float height, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		x *= Screen.width;
		y *= Screen.height;
		width *= Screen.width;
		height *= Screen.height;
		Camera camera = GetCamera();
		SetProgram(PROGRAM.QUADS, GUIMaterial);
		GL.Color(color);
		GL.Vertex(camera.ScreenToWorldPoint(new Vector3(x, y, camera.nearClipPlane + GUIOffset)));
		GL.Vertex(camera.ScreenToWorldPoint(new Vector3(x+width, y, camera.nearClipPlane + GUIOffset)));
		GL.Vertex(camera.ScreenToWorldPoint(new Vector3(x+width, y+height, camera.nearClipPlane + GUIOffset)));
		GL.Vertex(camera.ScreenToWorldPoint(new Vector3(x, y+height, camera.nearClipPlane + GUIOffset)));
	}

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
		Camera camera = GetCamera();
		SetProgram(PROGRAM.TRIANGLES, GUIMaterial);
		GL.Color(color);
		GL.Vertex(camera.ScreenToWorldPoint(new Vector3(xA, yA, camera.nearClipPlane + GUIOffset)));
		GL.Vertex(camera.ScreenToWorldPoint(new Vector3(xB, yB, camera.nearClipPlane + GUIOffset)));
		GL.Vertex(camera.ScreenToWorldPoint(new Vector3(xC, yC, camera.nearClipPlane + GUIOffset)));
	}

	public static void DrawGUICircle(float xCenter, float yCenter, float radius, Color color) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		xCenter *= Screen.width;
		yCenter *= Screen.height;
		radius *= Screen.height;
		int resolution = 30;
		Camera camera = GetCamera();
		SetProgram(PROGRAM.TRIANGLES, GUIMaterial);
        GL.Color(color);
		Vector3 center = camera.ScreenToWorldPoint(new Vector3(xCenter, yCenter, camera.nearClipPlane + GUIOffset));
		for(int i=0; i<resolution-1; i++) {
			GL.Vertex(center);
			float a = 2f * Mathf.PI * (float)i / ((float)resolution-1f);
			GL.Vertex(camera.ScreenToWorldPoint(new Vector3(xCenter + Mathf.Cos(a)*radius, yCenter + Mathf.Sin(a)*radius, camera.nearClipPlane + GUIOffset)));
			float b = 2f * Mathf.PI * (float)(i+1) / ((float)resolution-1f);
			GL.Vertex(camera.ScreenToWorldPoint(new Vector3(xCenter + Mathf.Cos(b)*radius, yCenter + Mathf.Sin(b)*radius, camera.nearClipPlane + GUIOffset)));
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