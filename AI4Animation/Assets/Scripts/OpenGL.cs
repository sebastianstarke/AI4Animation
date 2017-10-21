using UnityEngine;

public static class OpenGL {


	private static bool Initialised = false;

	private static Material ColorMaterial;
	private static int CircleResolution = 10;
	private static Vector3[] CirclePoints;

	static void Initialise() {
		if(!Initialised) {
			CreateMaterial();
			CreateCircleData();
			Initialised = true;
		}
	}

    static void CreateMaterial() {
		// Unity has a built-in shader that is useful for drawing
		// simple colored things.
		Shader shader = Shader.Find("Hidden/Internal-Colored");
		ColorMaterial = new Material(shader);
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

    public static void DrawLine(Vector3 start, Vector3 end, float width, Color color) {
       	Initialise();

        ColorMaterial.SetPass(0);

		Vector3 dir = (end-start).normalized;
		Vector3 orthoStart = width / 2f * (Quaternion.AngleAxis(90f, (start - Camera.main.transform.position).normalized) * dir);
		Vector3 orthoEnd = width / 2f * (Quaternion.AngleAxis(90f, (end - Camera.main.transform.position).normalized) * dir);
		Vector3 a = start+orthoStart;
		Vector3 b = start-orthoStart;
		Vector3 c = end+orthoEnd;
		Vector3 d = end-orthoEnd;

        GL.Begin(GL.TRIANGLE_STRIP);
        GL.Color(color);
        GL.Vertex(a);
		GL.Vertex(b);
		GL.Vertex(c);
		GL.Vertex(d);
        GL.End();
    }

	public static void DrawCircle(Vector3 center, float radius, Color color) {
		Initialise();

        ColorMaterial.SetPass(0);

        GL.Begin(GL.TRIANGLE_STRIP);
        GL.Color(color);
		for(int i=0; i<CircleResolution; i++) {
			GL.Vertex(center);
			GL.Vertex(center + radius * (Camera.main.transform.rotation * CirclePoints[i]));
		}
        GL.End();
	}
}