using UnityEngine;

public static class UnityGL {

	private static Material ColorMaterial;
	private static Vector3[] IsocelesTrianglePoints;
	private static int CircleResolution = 10;
	private static Vector3[] CirclePoints;
	private static int SphereResolution = 10;
	private static Vector3[] SpherePoints;

	private static bool Drawing = false;

	private static bool Initialised = false;

	private static PROGRAM Program = PROGRAM.NONE;

	private enum PROGRAM {NONE, LINES, TRIANGLES, TRIANGLE_STRIP, QUADS};

	private static Vector3 ViewPosition = Vector3.zero;
	private static Quaternion ViewRotation = Quaternion.identity;

	static void Initialise() {
		if(Initialised) {
			return;
		}

		//Create Material
		Resources.UnloadUnusedAssets();
		Shader colorShader = Shader.Find("Hidden/Internal-Colored");
		ColorMaterial = new Material(colorShader);
		ColorMaterial.hideFlags = HideFlags.HideAndDontSave;
		// Turn on alpha blending
		ColorMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
		ColorMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
		// Turn backface culling off
		ColorMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
		// Turn off depth writes
		ColorMaterial.SetInt("_ZWrite", 1);

		//Create Triangle
		IsocelesTrianglePoints = new Vector3[3];

		//Create Circle
		CirclePoints = new Vector3[CircleResolution];

		//Create Sphere
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

	private static void SetProgram(PROGRAM program) {
		if(Program != program) {
			GL.End();
			switch(program) {
				case PROGRAM.NONE:
				break;
				case PROGRAM.LINES:
				ColorMaterial.SetPass(0);
				GL.Begin(GL.LINES);
				break;
				case PROGRAM.TRIANGLES:
				ColorMaterial.SetPass(0);
				GL.Begin(GL.TRIANGLES);
				break;
				case PROGRAM.TRIANGLE_STRIP:
				ColorMaterial.SetPass(0);
				GL.Begin(GL.TRIANGLE_STRIP);
				break;
				case PROGRAM.QUADS:
				ColorMaterial.SetPass(0);
				GL.Begin(GL.QUADS);
				break;
			}
			Program = program;
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
			SetProgram(PROGRAM.NONE);
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
		SetProgram(PROGRAM.LINES);
		GL.Color(color);
		GL.Vertex(start);
		GL.Vertex(end);
	}

	public static void DrawLine(Vector3 start, Vector3 end, Color startColor, Color endColor) {
		if(!Drawing) {
			Debug.Log("Drawing has not yet been started.");
			return;
		}
		SetProgram(PROGRAM.LINES);
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
		SetProgram(PROGRAM.QUADS);
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
		SetProgram(PROGRAM.QUADS);
		Vector3 dir = end-start;
		Vector3 orthoStart = startWidth/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir).normalized;
		Vector3 orthoEnd = endWidth/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir).normalized;

		/*
		Vector3 center = 0.5f*(end+start);
		Vector3 ctoA = start+orthoStart-center;
		Vector3 ctoB = start-orthoStart-center;
		Vector3 ctoC = end-orthoEnd-center;
		Vector3 ctoD = end+orthoEnd-center;

		GL.Color(Color.black);
        GL.Vertex(center+ctoA+0.01f*(start-ViewPosition).normalized);
		GL.Vertex(center+ctoB+0.01f*(start-ViewPosition).normalized);
		GL.Vertex(center+ctoC+0.01f*(end-ViewPosition).normalized);
		GL.Vertex(center+ctoD+0.01f*(end-ViewPosition).normalized);
		*/

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
		SetProgram(PROGRAM.QUADS);
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
		SetProgram(PROGRAM.QUADS);
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
		SetProgram(PROGRAM.TRIANGLES);
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
		SetProgram(PROGRAM.TRIANGLES);
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
		SetProgram(PROGRAM.TRIANGLES);
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
		SetProgram(PROGRAM.TRIANGLES);
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
		SetProgram(PROGRAM.NONE);
		material.SetPass(0);
		Graphics.DrawMeshNow(Utility.GetPrimitiveMesh(PrimitiveType.Sphere), Matrix4x4.TRS(position, rotation, scale));
	}

	private static Camera GetCamera() {
		if(Camera.current != null) {
			return Camera.current;
		} else {
			return Camera.main;
		}
	}

}