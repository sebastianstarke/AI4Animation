using UnityEngine;

public static class UnityGL {

	private static bool Initialised = false;

	private static Material ColorMaterial;

	private static int CircleResolution = 10;
	private static Vector3[] CirclePoints;

	private static int SphereResolution = 10;
	private static Vector3[] SpherePoints;

	static void Initialise() {
		if(!Initialised) {
			CreateMaterial();
			CreateCircleData();
			CreateSphereData();
			Initialised = true;
		}
	}

    static void CreateMaterial() {
		Shader colorShader = Shader.Find("Hidden/Internal-Colored");
		ColorMaterial = new Material(colorShader);
		ColorMaterial.hideFlags = HideFlags.HideAndDontSave;
		// Turn on alpha blending
		ColorMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
		ColorMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
		// Turn backface culling off
		ColorMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
		// Turn off depth writes
		ColorMaterial.SetInt("_ZWrite", 0);
    }

	static void CreateCircleData() {
		CirclePoints = new Vector3[CircleResolution];
		for(int i=0; i<CircleResolution; i++) {
			float angle = 2f * Mathf.PI * ((float)i-1f) / ((float)CircleResolution-2f);
			CirclePoints[i] = new Vector3(Mathf.Cos(angle), Mathf.Sin(angle), 0f);
		}
	}

	static void CreateSphereData() {
		SpherePoints = new Vector3[4 * SphereResolution * SphereResolution];
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

    public static void DrawLine(Vector3 start, Vector3 end, float startWidth, float endWidth, Color color) {
       	Initialise();

		ColorMaterial.SetPass(0);

		Vector3 dir = (end-start).normalized;
		Vector3 orthoStart = Quaternion.AngleAxis(90f, (start - GetCamera().transform.position).normalized) * dir;
		Vector3 orthoEnd = Quaternion.AngleAxis(90f, (end - GetCamera().transform.position).normalized) * dir;
		Vector3 a = start+startWidth/2f*orthoStart;
		Vector3 b = start-startWidth/2f*orthoStart;
		Vector3 c = end+endWidth/2f*orthoEnd;
		Vector3 d = end-endWidth/2f*orthoEnd;

        //GL.PushMatrix();
        //GL.LoadProjectionMatrix(GetComponent<Camera>().projectionMatrix);
        GL.Begin(GL.QUADS);
        GL.Color(color);
        GL.Vertex(a);
		GL.Vertex(b);
		GL.Vertex(d);
		GL.Vertex(c);
        GL.End();
		//GL.PopMatrix();
    }

    public static void DrawLine(Vector3 start, Vector3 end, float width, Color color) {
		DrawLine(start, end, width, width, color);
    }

	public static void DrawTriangle(Vector3 a, Vector3 b, Vector3 c, Color color) {
       	Initialise();

		ColorMaterial.SetPass(0);

        GL.Begin(GL.TRIANGLE_STRIP);
        GL.Color(color);
        GL.Vertex(a);
		GL.Vertex(b);
		GL.Vertex(c);
        GL.End();
	}

	public static void DrawCircle(Vector3 center, float radius, Color color) {
		Initialise();

		ColorMaterial.SetPass(0);

        GL.Begin(GL.TRIANGLE_STRIP);
        GL.Color(color);
		for(int i=0; i<CircleResolution; i++) {
			GL.Vertex(center);
			GL.Vertex(center + radius * (GetCamera().transform.rotation * CirclePoints[i]));
		}
        GL.End();
	}

	public static void DrawSphere(Vector3 center, float radius, Color color) {
		Initialise();

		ColorMaterial.SetPass(0);

        GL.Begin(GL.TRIANGLE_STRIP);
        GL.Color(color);

		int index = 0;
		for(int i=0; i<SphereResolution; i++){ // U-points
			for(int j=0; j<SphereResolution; j++){ // V-points
				// Output the first triangle of this grid square
				GL.Vertex(center + radius*SpherePoints[index+0]);
				GL.Vertex(center + radius*SpherePoints[index+2]);
				GL.Vertex(center + radius*SpherePoints[index+1]);
				// Output the other triangle of this grid square
				GL.Vertex(center + radius*SpherePoints[index+3]);
				GL.Vertex(center + radius*SpherePoints[index+1]);
				GL.Vertex(center + radius*SpherePoints[index+2]);
				index += 4;
			}
		}

        GL.End();
	}

	public static void DrawArrow(Vector3 start, Vector3 end, float tipPivot, float shaftWidth, float tipWidth, Color color) {
		if(tipPivot < 0f || tipPivot > 1f) {
			Debug.Log("The tip pivot must be specified between 0 and 1.");
			return;
		}
		Vector3 pivot = start + tipPivot * (end-start);
		DrawLine(start, pivot, shaftWidth, color);
		DrawLine(pivot, end, tipWidth, 0f, color);
	}

	public static void DrawMesh(Mesh mesh, Vector3 position, Quaternion rotation, Vector3 scale, Material material) {
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