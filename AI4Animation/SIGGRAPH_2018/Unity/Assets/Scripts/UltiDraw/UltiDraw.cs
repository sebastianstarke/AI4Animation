using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public static class UltiDraw {

	[System.Serializable]
	public class GUIRect {
		[Range(0f, 1f)] public float X = 0.5f;
		[Range(0f, 1f)] public float Y = 0.5f;
		[Range(0f, 1f)] public float W = 0.5f;
		[Range(0f, 1f)] public float H = 0.5f;

		public GUIRect(float x, float y, float w, float h) {
			X = x;
			Y = y;
			W = w;
			H = h;
		}

		public Vector2 GetPosition() {
			return new Vector2(X, Y);
		}

		public Vector2 GetSize() {
			return new Vector2(W, H);
		}
	}

	public static Color White = Color.white;
	public static Color Black = Color.black;
	public static Color Red = Color.red;
	public static Color DarkRed = new Color(0.75f, 0f, 0f, 1f);
	public static Color Green = Color.green;
	public static Color DarkGreen = new Color(0f, 0.75f, 0f, 1f);
	public static Color Blue = Color.blue;
	public static Color Cyan = Color.cyan;
	public static Color Magenta = Color.magenta;
	public static Color Yellow = Color.yellow;
	public static Color Grey = Color.grey;
	public static Color LightGrey = new Color(0.75f, 0.75f, 0.75f, 1f);
	public static Color DarkGrey = new Color(0.25f, 0.25f, 0.25f, 1f);
	public static Color BlackGrey = new Color(0.125f, 0.125f, 0.125f, 1f);
	public static Color Orange = new Color(1f, 0.5f, 0f, 1f);
	public static Color Brown = new Color(0.5f, 0.25f, 0f, 1f);
	public static Color Mustard = new Color(1f, 0.75f, 0.25f, 1f);
	public static Color Teal = new Color(0f, 0.75f, 0.75f, 1f);
	public static Color Purple = new Color(0.5f, 0f, 0.5f, 1f);
	public static Color DarkBlue = new Color(0f, 0f, 0.75f, 1f);
	public static Color IndianRed = new Color(205f/255f, 92f/255f, 92f/255f, 1f);
	public static Color Gold = new Color(212f/255f, 175f/255f, 55f/255f, 1f);

	private static int Resolution = 30;

	private static Mesh Initialised;

	private static bool Active;

	private static Material GLMaterial;
	private static Material MeshMaterial;

	private static float GUIOffset = 0.001f;

	private static Camera Camera;
	private static Vector3 ViewPosition;
	private static Quaternion ViewRotation;

	private static PROGRAM Program = PROGRAM.NONE;
	private enum PROGRAM {NONE, LINES, TRIANGLES, TRIANGLE_STRIP, QUADS};

	private static Mesh CircleMesh;
	private static Mesh QuadMesh;
	private static Mesh CubeMesh;
	private static Mesh SphereMesh;
	private static Mesh CylinderMesh;
	private static Mesh CapsuleMesh;
	private static Mesh ConeMesh;
	private static Mesh PyramidMesh;
	private static Mesh BoneMesh;

	private static Vector3[] CircleWire;
	private static Vector3[] QuadWire;
	private static Vector3[] CubeWire;
	private static Vector3[] SphereWire;
	private static Vector3[] CylinderWire;
	private static Vector3[] CapsuleWire;
	private static Vector3[] ConeWire;
	private static Vector3[] PyramidWire;
	private static Vector3[] BoneWire;

	//------------------------------------------------------------------------------------------
	//CONTROL FUNCTIONS
	//------------------------------------------------------------------------------------------
	public static void Begin() {
		if(Active) {
			Debug.Log("Drawing is still active. Call 'End()' to stop.");
		} else {
			Initialise();
			Camera = GetCamera();
			ViewPosition = Camera.transform.position;
			ViewRotation = Camera.transform.rotation;
			Active = true;
		}
	}

	public static void End() {
		if(Active) {
			SetProgram(PROGRAM.NONE);
			Camera = null;
			ViewPosition = Vector3.zero;
			ViewRotation = Quaternion.identity;
			Active = false;
		} else {
			Debug.Log("Drawing is not active. Call 'Begin()' to start.");
		}
	}

	public static void SetDepthRendering(bool enabled) {
		Initialise();
		SetProgram(PROGRAM.NONE);
		GLMaterial.SetInt("_ZWrite", enabled ? 1 : 0);
		GLMaterial.SetInt("_ZTest", enabled ? (int)UnityEngine.Rendering.CompareFunction.LessEqual : (int)UnityEngine.Rendering.CompareFunction.Always);
		MeshMaterial.SetInt("_ZWrite", enabled ? 1 : 0);
		MeshMaterial.SetInt("_ZTest", enabled ? (int)UnityEngine.Rendering.CompareFunction.LessEqual : (int)UnityEngine.Rendering.CompareFunction.Always);
	}

	public static void SetCurvature(float value) {
		Initialise();
		SetProgram(PROGRAM.NONE);
		MeshMaterial.SetFloat("_Power", value);
	}

	public static void SetFilling(float value) {
		value = Mathf.Clamp(value, 0f, 1f);
		Initialise();
		SetProgram(PROGRAM.NONE);
		MeshMaterial.SetFloat("_Filling", value);
	}

	//------------------------------------------------------------------------------------------
	//2D SCENE DRAWING FUNCTIONS
	//------------------------------------------------------------------------------------------
	public static void DrawLine(Vector3 start, Vector3 end, Color color) {
		if(Return()) {return;};
		SetProgram(PROGRAM.LINES);
		GL.Color(color);
		GL.Vertex(start);
		GL.Vertex(end);
	}

    public static void DrawLine(Vector3 start, Vector3 end, float thickness, Color color) {
		DrawLine(start, end, thickness, thickness, color);
    }

    public static void DrawLine(Vector3 start, Vector3 end, float startThickness, float endThickness, Color color) {
		if(Return()) {return;};
		SetProgram(PROGRAM.QUADS);
		GL.Color(color);
		Vector3 dir = (end-start).normalized;
		Vector3 orthoStart = startThickness/2f * (Quaternion.AngleAxis(90f, (start - ViewPosition)) * dir);
		Vector3 orthoEnd = endThickness/2f * (Quaternion.AngleAxis(90f, (end - ViewPosition)) * dir);
		GL.Vertex(end+orthoEnd);
		GL.Vertex(end-orthoEnd);
		GL.Vertex(start-orthoStart);
		GL.Vertex(start+orthoStart);
    }

	public static void DrawTriangle(Vector3 a, Vector3 b, Vector3 c, Color color) {
		if(Return()) {return;};
		SetProgram(PROGRAM.TRIANGLES);
		GL.Color(color);
        GL.Vertex(b);
		GL.Vertex(a);
		GL.Vertex(c);
	}

	public static void DrawCircle(Vector3 position, float size, Color color) {
		DrawMesh(CircleMesh, position, ViewRotation, size*Vector3.one, color);
	}

	public static void DrawCircle(Vector3 position, Quaternion rotation, float size, Color color) {
		DrawMesh(CircleMesh, position, rotation, size*Vector3.one, color);
	}

	public static void DrawWireCircle(Vector3 position, float size, Color color) {
		DrawWire(CircleWire, position, ViewRotation, size*Vector3.one, color);
	}

	public static void DrawWireCircle(Vector3 position, Quaternion rotation, float size, Color color) {
		DrawWire(CircleWire, position, rotation, size*Vector3.one, color);
	}

	public static void DrawWiredCircle(Vector3 position, float size, Color circleColor, Color wireColor) {
		DrawCircle(position, size, circleColor);
		DrawWireCircle(position, size, wireColor);
	}

	public static void DrawWiredCircle(Vector3 position, Quaternion rotation, float size, Color circleColor, Color wireColor) {
		DrawCircle(position, rotation, size, circleColor);
		DrawWireCircle(position, rotation, size, wireColor);
	}

	public static void DrawEllipse(Vector3 position, float width, float height, Color color) {
		DrawMesh(CircleMesh, position, ViewRotation, new Vector3(width, height, 1f), color);
	}

	public static void DrawEllipse(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawMesh(CircleMesh, position, rotation, new Vector3(width, height, 1f), color);
	}

	public static void DrawWireEllipse(Vector3 position, float width, float height, Color color) {
		DrawWire(CircleWire, position, ViewRotation, new Vector3(width, height, 1f), color);
	}

	public static void DrawWireEllipse(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawWire(CircleWire, position, rotation, new Vector3(width, height, 1f), color);
	}

	public static void DrawWiredEllipse(Vector3 position, float width, float height, Color ellipseColor, Color wireColor) {
		DrawEllipse(position, ViewRotation, width, height, ellipseColor);
		DrawWireEllipse(position, ViewRotation, width, height, wireColor);
	}

	public static void DrawWiredEllipse(Vector3 position, Quaternion rotation, float width, float height, Color ellipseColor, Color wireColor) {
		DrawEllipse(position, rotation, width, height, ellipseColor);
		DrawWireEllipse(position, rotation, width, height, wireColor);
	}

	public static void DrawArrow(Vector3 start, Vector3 end, float tipPivot, float shaftWidth, float tipWidth, Color color) {
		tipPivot = Mathf.Clamp(tipPivot, 0f, 1f);
		Vector3 pivot = start + tipPivot * (end-start);
		DrawLine(start, pivot, shaftWidth, color);
		DrawLine(pivot, end, tipWidth, 0f, color);
	}

	public static void DrawArrow(Vector3 start, Vector3 end, float tipPivot, float shaftWidth, float tipWidth, Color shaftColor, Color tipColor) {
		tipPivot = Mathf.Clamp(tipPivot, 0f, 1f);
		Vector3 pivot = start + tipPivot * (end-start);
		DrawLine(start, pivot, shaftWidth, shaftColor);
		DrawLine(pivot, end, tipWidth, 0f, tipColor);
	}

	public static void DrawGrid(Vector3 center, Quaternion rotation, int cellsX, int cellsY, float sizeX, float sizeY, Color color) {
		if(Return()) {return;}
		float width = cellsX * sizeX;
		float height = cellsY * sizeY;
		Vector3 start = center - width/2f * (rotation * Vector3.right) - height/2f * (rotation * Vector3.forward);
		Vector3 dirX = rotation * Vector3.right;
		Vector3 dirY = rotation * Vector3.forward;
		for(int i=0; i<cellsX+1; i++) {
			DrawLine(start + i*sizeX*dirX, start + i*sizeX*dirX + height*dirY, color);
		}
		for(int i=0; i<cellsY+1; i++) {
			DrawLine(start + i*sizeY*dirY, start + i*sizeY*dirY + width*dirX, color);
		}
	}

	public static void DrawQuad(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawMesh(QuadMesh, position, rotation, new Vector3(width, height, 1f), color);
	}

	public static void DrawWireQuad(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawWire(QuadWire, position, rotation, new Vector3(width, height, 1f), color);
	}

	public static void DrawWiredQuad(Vector3 position, Quaternion rotation, float width, float height, Color quadColor, Color wireColor) {
		DrawQuad(position, rotation, width, height, quadColor);
		DrawWireQuad(position, rotation, width, height, wireColor);
	}

	public static void DrawCube(Vector3 position, Quaternion rotation, float size, Color color) {
		DrawMesh(CubeMesh, position, rotation, size*Vector3.one, color);
	}

	public static void DrawWireCube(Vector3 position, Quaternion rotation, float size, Color color) {
		DrawWire(CubeWire, position, rotation, size*Vector3.one, color);
	}

	public static void DrawWiredCube(Vector3 position, Quaternion rotation, float size, Color cubeColor, Color wireColor) {
		DrawCube(position, rotation, size, cubeColor);
		DrawWireCube(position, rotation, size, wireColor);
	}

	public static void DrawCuboid(Vector3 position, Quaternion rotation, Vector3 size, Color color) {
		DrawMesh(CubeMesh, position, rotation, size, color);
	}

	public static void DrawWireCuboid(Vector3 position, Quaternion rotation, Vector3 size, Color color) {
		DrawWire(CubeWire, position, rotation, size, color);
	}

	public static void DrawWiredCuboid(Vector3 position, Quaternion rotation, Vector3 size, Color cuboidColor, Color wireColor) {
		DrawCuboid(position, rotation, size, cuboidColor);
		DrawWireCuboid(position, rotation, size, wireColor);
	}

	public static void DrawSphere(Vector3 position, Quaternion rotation, float size, Color color) {
		DrawMesh(SphereMesh, position, rotation, size*Vector3.one, color);
	}

	public static void DrawWireSphere(Vector3 position, Quaternion rotation, float size, Color color) {
		DrawWire(SphereWire, position, rotation, size*Vector3.one, color);
	}

	public static void DrawWiredSphere(Vector3 position, Quaternion rotation, float size, Color sphereColor, Color wireColor) {
		DrawSphere(position, rotation, size, sphereColor);
		DrawWireSphere(position, rotation, size, wireColor);
	}

	public static void DrawEllipsoid(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawMesh(SphereMesh, position, rotation, new Vector3(width, height, width), color);
	}

	public static void DrawWireEllipsoid(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawWire(SphereWire, position, rotation, new Vector3(width, height, width), color);
	}

	public static void DrawWiredEllipsoid(Vector3 position, Quaternion rotation, float width, float height, Color ellipsoidColor, Color wireColor) {
		DrawEllipsoid(position, rotation, width, height, ellipsoidColor);
		DrawWireEllipsoid(position, rotation, width, height, wireColor);
	}

	public static void DrawCylinder(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawMesh(CylinderMesh, position, rotation, new Vector3(width, height/2f, width), color);
	}

	public static void DrawWireCylinder(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawWire(CylinderWire, position, rotation, new Vector3(width, height/2f, width), color);
	}

	public static void DrawWiredCylinder(Vector3 position, Quaternion rotation, float width, float height, Color cylinderColor, Color wireColor) {
		DrawCylinder(position, rotation, width, height, cylinderColor);
		DrawWireCylinder(position, rotation, width, height, wireColor);
	}

	public static void DrawCapsule(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawMesh(CapsuleMesh, position, rotation, new Vector3(width, height/2f, width), color);
	}

	public static void DrawWireCapsule(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawWire(CapsuleWire, position, rotation, new Vector3(width, height/2f, width), color);
	}

	public static void DrawWiredCapsule(Vector3 position, Quaternion rotation, float width, float height, Color capsuleColor, Color wireColor) {
		DrawCapsule(position, rotation, width, height, capsuleColor);
		DrawWireCapsule(position, rotation, width, height, wireColor);
	}

	public static void DrawCone(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawMesh(ConeMesh, position, rotation, new Vector3(width, height, width), color);
	}

	public static void DrawWireCone(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawWire(ConeWire, position, rotation, new Vector3(width, height, width), color);
	}

	public static void DrawWiredCone(Vector3 position, Quaternion rotation, float width, float height, Color coneColor, Color wireColor) {
		DrawCone(position, rotation, width, height, coneColor);
		DrawWireCone(position, rotation, width, height, wireColor);
	}

	public static void DrawPyramid(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawMesh(PyramidMesh, position, rotation, new Vector3(width, height, width), color);
	}

	public static void DrawWirePyramid(Vector3 position, Quaternion rotation, float width, float height, Color color) {
		DrawWire(PyramidWire, position, rotation, new Vector3(width, height, width), color);
	}

	public static void DrawWiredPyramid(Vector3 position, Quaternion rotation, float width, float height, Color pyramidColor, Color wireColor) {
		DrawPyramid(position, rotation, width, height, pyramidColor);
		DrawWirePyramid(position, rotation, width, height, wireColor);
	}

	public static void DrawBone(Vector3 position, Quaternion rotation, float width, float length, Color color) {
		DrawMesh(BoneMesh, position, rotation, new Vector3(width, width, length), color);
	}

	public static void DrawWireBone(Vector3 position, Quaternion rotation, float width, float length, Color color) {
		DrawWire(BoneWire, position, rotation, new Vector3(width, width, length), color);
	}

	public static void DrawWiredBone(Vector3 position, Quaternion rotation, float width, float length, Color boneColor, Color wireColor) {
		DrawBone(position, rotation, width, length, boneColor);
		DrawWireBone(position, rotation, width, length, wireColor);
	}

	public static void DrawTranslateGizmo(Vector3 position, Quaternion rotation, float size) {
		if(Return()) {return;}
		DrawLine(position, position + 0.8f*size*(rotation*Vector3.right), Red);
		DrawCone(position + 0.8f*size*(rotation*Vector3.right), rotation*Quaternion.Euler(0f, 0f, -90f), 0.15f*size, 0.2f*size, Red);
		DrawLine(position, position + 0.8f*size*(rotation*Vector3.up), Green);
		DrawCone(position + 0.8f*size*(rotation*Vector3.up), rotation*Quaternion.Euler(0f, 0f, 0f), 0.15f*size, 0.2f*size, Green);
		DrawLine(position, position + 0.8f*size*(rotation*Vector3.forward), Blue);
		DrawCone(position + 0.8f*size*(rotation*Vector3.forward), rotation*Quaternion.Euler(90f, 0f, 0f), 0.15f*size, 0.2f*size, Blue);
	}

	public static void DrawRotateGizmo(Vector3 position, Quaternion rotation, float size) {
		if(Return()) {return;}
		SetProgram(PROGRAM.NONE);
		DrawWireCircle(position, rotation*Quaternion.Euler(0f, 90f, 0f), 2f*size, Red);
		SetProgram(PROGRAM.NONE);
		DrawWireCircle(position, rotation*Quaternion.Euler(90f, 0f, 90f), 2f*size, Green);
		SetProgram(PROGRAM.NONE);
		DrawWireCircle(position, rotation*Quaternion.Euler(0f, 0f, 0f), 2f*size, Blue);
		SetProgram(PROGRAM.NONE);
	}

	public static void DrawScaleGizmo(Vector3 position, Quaternion rotation, float size) {
		if(Return()) {return;}
		DrawLine(position, position + 0.85f*size*(rotation*Vector3.right), Red);
		DrawCube(position + 0.925f*size*(rotation*Vector3.right), rotation, 0.15f, Red);
		DrawLine(position, position + 0.85f*size*(rotation*Vector3.up), Green);
		DrawCube(position + 0.925f*size*(rotation*Vector3.up), rotation, 0.15f, Green);
		DrawLine(position, position + 0.85f*size*(rotation*Vector3.forward), Blue);
		DrawCube(position + 0.925f*size*(rotation*Vector3.forward), rotation, 0.15f, Blue);
	}

	public static void DrawMesh(Mesh mesh, Vector3 position, Quaternion rotation, Vector3 scale, Color color) {
		if(Return()) {return;}
		SetProgram(PROGRAM.NONE);
		MeshMaterial.color = color;
		MeshMaterial.SetPass(0);
		Graphics.DrawMeshNow(mesh, Matrix4x4.TRS(position, rotation, scale));
	}

	//------------------------------------------------------------------------------------------
	//GUI DRAWING FUNCTIONS
	//------------------------------------------------------------------------------------------
	public static void DrawGUILine(Vector2 start, Vector2 end, Color color) {
		if(Camera != Camera.main) {return;}
		if(Return()) {return;}
		SetProgram(PROGRAM.LINES);
		GL.Color(color);
		start.x *= Screen.width;
		start.y *= Screen.height;
		end.x *= Screen.width;
		end.y *= Screen.height;
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(start.x, start.y, Camera.nearClipPlane + GUIOffset)));
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(end.x, end.y, Camera.nearClipPlane + GUIOffset)));
	}

    public static void DrawGUILine(Vector2 start, Vector2 end, float thickness, Color color) {
		if(Camera != Camera.main) {return;}
		if(Return()) {return;}
		SetProgram(PROGRAM.QUADS);
		GL.Color(color);
		start.x *= Screen.width;
		start.y *= Screen.height;
		end.x *= Screen.width;
		end.y *= Screen.height;
		thickness *= Screen.width;
		Vector3 p1 = new Vector3(start.x, start.y, Camera.nearClipPlane + GUIOffset);
		Vector3 p2 = new Vector3(end.x, end.y, Camera.nearClipPlane + GUIOffset);
		Vector3 dir = end-start;
		Vector3 ortho = thickness/2f * (Quaternion.AngleAxis(90f, Vector3.forward) * dir).normalized;
        GL.Vertex(Camera.ScreenToWorldPoint(p1-ortho));
		GL.Vertex(Camera.ScreenToWorldPoint(p1+ortho));
		GL.Vertex(Camera.ScreenToWorldPoint(p2+ortho));
		GL.Vertex(Camera.ScreenToWorldPoint(p2-ortho));
    }

    public static void DrawGUILine(Vector2 start, Vector2 end, float startThickness, float endThickness, Color color) {
		if(Camera != Camera.main) {return;}
		if(Return()) {return;}
		SetProgram(PROGRAM.QUADS);
		GL.Color(color);
		start.x *= Screen.width;
		start.y *= Screen.height;
		end.x *= Screen.width;
		end.y *= Screen.height;
		startThickness *= Screen.width;
		endThickness *= Screen.width;
		Vector3 p1 = new Vector3(start.x, start.y, Camera.nearClipPlane + GUIOffset);
		Vector3 p2 = new Vector3(end.x, end.y, Camera.nearClipPlane + GUIOffset);
		Vector3 dir = end-start;
		Vector3 orthoStart = startThickness/2f * (Quaternion.AngleAxis(90f, Vector3.forward) * dir).normalized;
		Vector3 orthoEnd = endThickness/2f * (Quaternion.AngleAxis(90f, Vector3.forward) * dir).normalized;
        GL.Vertex(Camera.ScreenToWorldPoint(p1-orthoStart));
		GL.Vertex(Camera.ScreenToWorldPoint(p1+orthoStart));
		GL.Vertex(Camera.ScreenToWorldPoint(p2+orthoEnd));
		GL.Vertex(Camera.ScreenToWorldPoint(p2-orthoEnd));
    }

	public static void DrawGUIRectangle(Vector2 center, Vector2 size, Color color) {
		if(Camera != Camera.main) {return;}
		if(Return()) {return;}
		SetProgram(PROGRAM.QUADS);
		GL.Color(color);
		center.x *= Screen.width;
		center.y *= Screen.height;
		size.x *= Screen.width;
		size.y *= Screen.height;
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x+size.x/2f, center.y-size.y/2f, Camera.nearClipPlane + GUIOffset)));
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x-size.x/2f, center.y-size.y/2f, Camera.nearClipPlane + GUIOffset)));
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x+-size.x/2f, center.y+size.y/2f, Camera.nearClipPlane + GUIOffset)));
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x+size.x/2f, center.y+size.y/2f, Camera.nearClipPlane + GUIOffset)));
	}

	public static void DrawGUIRectangle(Vector2 center, Vector2 size, Color color, float borderWidth, Color borderColor) {
		if(Camera != Camera.main) {return;}
		if(Return()) {return;}
		SetProgram(PROGRAM.QUADS);
		GL.Color(color);
		center.x *= Screen.width;
		center.y *= Screen.height;
		size.x *= Screen.width;
		size.y *= Screen.height;
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x+size.x/2f, center.y-size.y/2f, Camera.nearClipPlane + GUIOffset)));
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x-size.x/2f, center.y-size.y/2f, Camera.nearClipPlane + GUIOffset)));
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x+-size.x/2f, center.y+size.y/2f, Camera.nearClipPlane + GUIOffset)));
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x+size.x/2f, center.y+size.y/2f, Camera.nearClipPlane + GUIOffset)));
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
	}

	public static void DrawGUITriangle(Vector2 a, Vector2 b, Vector2 c, Color color) {
		//TODO: There is some dependency here on the order of triangles, need to fix...
		if(Camera != Camera.main) {return;}
		if(Return()) {return;}
		SetProgram(PROGRAM.TRIANGLES);
		GL.Color(color);
		a.x *= Screen.width;
		a.y *= Screen.height;
		b.x *= Screen.width;
		b.y *= Screen.height;
		c.x *= Screen.width;
		c.y *= Screen.height;
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(a.x, a.y, Camera.nearClipPlane + GUIOffset)));
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(b.x, b.y, Camera.nearClipPlane + GUIOffset)));
		GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(c.x, c.y, Camera.nearClipPlane + GUIOffset)));
	}

	public static void DrawGUICircle(Vector2 center, float size, Color color) {
		if(Camera != Camera.main) {return;}
		if(Return()) {return;}
		SetProgram(PROGRAM.TRIANGLES);
		GL.Color(color);
		center.x *= Screen.width;
		center.y *= Screen.height;
		for(int i=0; i<CircleWire.Length-1; i++) {
			GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x + size*CircleWire[i].x*Screen.width, center.y + size*CircleWire[i].y*Screen.width, Camera.nearClipPlane + GUIOffset)));
			GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x, center.y, Camera.nearClipPlane + GUIOffset)));
			GL.Vertex(Camera.ScreenToWorldPoint(new Vector3(center.x + size*CircleWire[i+1].x*Screen.width, center.y + size*CircleWire[i+1].y*Screen.width, Camera.nearClipPlane + GUIOffset)));
		}
	}

	public static void DrawGUITexture(Vector2 center, float size, Texture texture, Color color) {
		Vector2 area = size * Screen.width * new Vector2(texture.width, texture.height) / texture.width;
		Vector2 pos = new Vector2(center.x*Screen.width - area.x/2f, center.y*Screen.height - area.y/2f);
		GUI.DrawTexture(new Rect(pos.x, pos.y, area.x, area.y), texture, ScaleMode.StretchToFill, true, 1f, color, 0f, 100f);
	}

	public static void DrawGUIRectangleFrame(Vector2 center, Vector2 size, float thickness, Color color) {
		DrawGUIRectangle(new Vector2(center.x - size.x/2f - thickness/2f, center.y), new Vector2(thickness, size.y), color);
		DrawGUIRectangle(new Vector2(center.x + size.x/2f + thickness/2f, center.y), new Vector2(thickness, size.y), color);
		DrawGUIRectangle(new Vector2(center.x, center.y - size.y/2f - thickness*Screen.width/Screen.height/2f), new Vector2(size.x + 2f*thickness, thickness*Screen.width/Screen.height), color);
		DrawGUIRectangle(new Vector2(center.x, center.y + size.y/2f + thickness*Screen.width/Screen.height/2f), new Vector2(size.x + 2f*thickness, thickness*Screen.width/Screen.height), color);
	}

	//------------------------------------------------------------------------------------------
	//PLOTTING FUNCTIONS
	//------------------------------------------------------------------------------------------
	public static void DrawGUIFunction(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, Color background, Color line) {
		DrawGUIRectangle(center, size, background);
		float x = center.x - size.x/2f;
		float y = center.y - size.y/2f;
		//float scale = yMax - yMin;
		for(int i=0; i<values.Length-1; i++) {
			DrawGUILine(
					new Vector2(x + (float)i/(float)(values.Length-1)*size.x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					new Vector2(x + (float)(i+1)/(float)(values.Length-1)*size.x, y + Mathf.Clamp(Normalise(values[i+1], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
				line
			);
		}
	}

	public static void DrawGUIFunction(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, float thickness, Color background, Color line) {
		DrawGUIRectangle(center, size, background);
		float x = center.x - size.x/2f;
		float y = center.y - size.y/2f;
		//float scale = yMax - yMin;
		for(int i=0; i<values.Length-1; i++) {
			DrawGUILine(
					new Vector2(x + (float)i/(float)(values.Length-1)*size.x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					new Vector2(x + (float)(i+1)/(float)(values.Length-1)*size.x, y + Mathf.Clamp(Normalise(values[i+1], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
				thickness,
				line
			);
		}
	}

	public static void DrawGUIFunction(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, Color background, Color line, float borderWidth, Color borderColor) {
		DrawGUIRectangle(center, size, background);
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
		float x = center.x - size.x/2f;
		float y = center.y - size.y/2f;
		//float scale = yMax - yMin;
		for(int i=0; i<values.Length-1; i++) {
			DrawGUILine(
					new Vector2(x + (float)i/(float)(values.Length-1)*size.x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					new Vector2(x + (float)(i+1)/(float)(values.Length-1)*size.x, y + Mathf.Clamp(Normalise(values[i+1], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
				line
			);
		}
	}

	public static void DrawGUIFunction(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, float thickness, Color background, Color line, float borderWidth, Color borderColor) {
		DrawGUIRectangle(center, size, background);
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
		float x = center.x - size.x/2f;
		float y = center.y - size.y/2f;
		//float scale = yMax - yMin;
		for(int i=0; i<values.Length-1; i++) {
			DrawGUILine(
					new Vector2(x + (float)i/(float)(values.Length-1)*size.x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					new Vector2(x + (float)(i+1)/(float)(values.Length-1)*size.x, y + Mathf.Clamp(Normalise(values[i+1], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
				thickness,
				line
			);
		}
	}

	public static void DrawGUIFunctions(Vector2 center, Vector2 size, List<float[]> values, float yMin, float yMax, Color background, Color[] lines) {
		DrawGUIRectangle(center, size, background);
		float x = center.x - size.x/2f;
		float y = center.y - size.y/2f;
		//float scale = yMax - yMin;
		for(int k=0; k<values.Count; k++) {
			for(int i=0; i<values[k].Length-1; i++) {
				DrawGUILine(
					new Vector2(x + (float)i/(float)(values[k].Length-1)*size.x, y + Mathf.Clamp(Normalise(values[k][i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					new Vector2(x + (float)(i+1)/(float)(values[k].Length-1)*size.x, y + Mathf.Clamp(Normalise(values[k][i+1], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					lines[k]
				);
			}
		}
	}

	public static void DrawGUIFunctions(Vector2 center, Vector2 size, List<float[]> values, float yMin, float yMax, float thickness, Color background, Color[] lines) {
		DrawGUIRectangle(center, size, background);
		float x = center.x - size.x/2f;
		float y = center.y - size.y/2f;
		//float scale = yMax - yMin;
		for(int k=0; k<values.Count; k++) {
			for(int i=0; i<values[k].Length-1; i++) {
				DrawGUILine(
					new Vector2(x + (float)i/(float)(values[k].Length-1)*size.x, y + Mathf.Clamp(Normalise(values[k][i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					new Vector2(x + (float)(i+1)/(float)(values[k].Length-1)*size.x, y + Mathf.Clamp(Normalise(values[k][i+1], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					thickness,
					lines[k]
				);
			}
		}
	}

	public static void DrawGUIFunctions(Vector2 center, Vector2 size, List<float[]> values, float yMin, float yMax, Color background, Color[] lines, float borderWidth, Color borderColor) {
		DrawGUIRectangle(center, size, background);
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
		float x = center.x - size.x/2f;
		float y = center.y - size.y/2f;
		//float scale = yMax - yMin;
		for(int k=0; k<values.Count; k++) {
			for(int i=0; i<values[k].Length-1; i++) {
				DrawGUILine(
					new Vector2(x + (float)i/(float)(values[k].Length-1)*size.x, y + Mathf.Clamp(Normalise(values[k][i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					new Vector2(x + (float)(i+1)/(float)(values[k].Length-1)*size.x, y + Mathf.Clamp(Normalise(values[k][i+1], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					lines[k]
				);
			}
		}
	}

	public static void DrawGUIFunctions(Vector2 center, Vector2 size, List<float[]> values, float yMin, float yMax, float thickness, Color background, Color[] lines, float borderWidth, Color borderColor) {
		DrawGUIRectangle(center, size, background);
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
		float x = center.x - size.x/2f;
		float y = center.y - size.y/2f;
		//float scale = yMax - yMin;
		for(int k=0; k<values.Count; k++) {
			for(int i=0; i<values[k].Length-1; i++) {
				DrawGUILine(
					new Vector2(x + (float)i/(float)(values[k].Length-1)*size.x, y + Mathf.Clamp(Normalise(values[k][i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					new Vector2(x + (float)(i+1)/(float)(values[k].Length-1)*size.x, y + Mathf.Clamp(Normalise(values[k][i+1], yMin, yMax, 0f, 1f), 0f, 1f)*size.y),
					thickness,
					lines[k]
				);
			}
		}
	}

	public static void DrawGUIBars(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, Color background, Color color) {
		DrawGUIRectangle(center, size, background);
		for(int i=0; i<values.Length; i++) {
			float x = center.x - size.x/2f + (float)i / (float)(values.Length-1) * size.x;
			float y = center.y - size.y/2f;
			Vector3 pivot = new Vector2(x, y + Utility.Normalise(0f, yMin, yMax, 0f, 1f)*size.y);
			Vector2 tip = new Vector2(x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y);
			DrawGUILine(pivot, tip, color);
		}
	}

	public static void DrawGUIBars(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, Color background, Color[] colors) {
		DrawGUIRectangle(center, size, background);
		for(int i=0; i<values.Length; i++) {
			float x = center.x - size.x/2f + (float)i / (float)(values.Length-1) * size.x;
			float y = center.y - size.y/2f;
			Vector3 pivot = new Vector2(x, y + Utility.Normalise(0f, yMin, yMax, 0f, 1f)*size.y);
			Vector2 tip = new Vector2(x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y);
			DrawGUILine(pivot, tip, colors[i]);
		}
	}

	public static void DrawGUIBars(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, float thickness, Color background, Color color) {
		DrawGUIRectangle(center, size, background);
		for(int i=0; i<values.Length; i++) {
			float x = center.x - size.x/2f + Utility.Normalise((float)i / (float)(values.Length-1), 0f, 1f, thickness, 1f-thickness) * size.x;
			float y = center.y - size.y/2f;
			Vector3 pivot = new Vector2(x, y + Utility.Normalise(0f, yMin, yMax, 0f, 1f)*size.y);
			Vector2 tip = new Vector2(x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y);
			DrawGUILine(pivot, tip, thickness, color);
		}
	}

	public static void DrawGUIBars(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, float thickness, Color background, Color[] colors) {
		DrawGUIRectangle(center, size, background);
		for(int i=0; i<values.Length; i++) {
			float x = center.x - size.x/2f + Utility.Normalise((float)i / (float)(values.Length-1), 0f, 1f, thickness, 1f-thickness) * size.x;
			float y = center.y - size.y/2f;
			Vector3 pivot = new Vector2(x, y + Utility.Normalise(0f, yMin, yMax, 0f, 1f)*size.y);
			Vector2 tip = new Vector2(x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y);
			DrawGUILine(pivot, tip, thickness, colors[i]);
		}
	}


	public static void DrawGUIBars(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, Color background, Color color, float borderWidth, Color borderColor) {
		DrawGUIRectangle(center, size, background);
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
		for(int i=0; i<values.Length; i++) {
			float x = center.x - size.x/2f + (float)i / (float)(values.Length-1) * size.x;
			float y = center.y - size.y/2f;
			Vector3 pivot = new Vector2(x, y + Utility.Normalise(0f, yMin, yMax, 0f, 1f)*size.y);
			Vector2 tip = new Vector2(x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y);
			DrawGUILine(pivot, tip, color);
		}
	}

	public static void DrawGUIBars(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, Color background, Color[] colors, float borderWidth, Color borderColor) {
		DrawGUIRectangle(center, size, background);
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
		for(int i=0; i<values.Length; i++) {
			float x = center.x - size.x/2f + (float)i / (float)(values.Length-1) * size.x;
			float y = center.y - size.y/2f;
			Vector3 pivot = new Vector2(x, y + Utility.Normalise(0f, yMin, yMax, 0f, 1f)*size.y);
			Vector2 tip = new Vector2(x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y);
			DrawGUILine(pivot, tip, colors[i]);
		}
	}

	public static void DrawGUIBars(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, float thickness, Color background, Color color, float borderWidth, Color borderColor) {
		DrawGUIRectangle(center, size, background);
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
		for(int i=0; i<values.Length; i++) {
			float x = center.x - size.x/2f + (float)i / (float)(values.Length-1) * size.x;
			float y = center.y - size.y/2f;
			Vector3 pivot = new Vector2(x, y + Utility.Normalise(0f, yMin, yMax, 0f, 1f)*size.y);
			Vector2 tip = new Vector2(x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y);
			DrawGUILine(pivot, tip, color);
		}
	}

	public static void DrawGUIBars(Vector2 center, Vector2 size, float[] values, float yMin, float yMax, float thickness, Color background, Color[] colors, float borderWidth, Color borderColor) {
		DrawGUIRectangle(center, size, background);
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
		for(int i=0; i<values.Length; i++) {
			float x = center.x - size.x/2f + (float)i / (float)(values.Length-1) * size.x;
			float y = center.y - size.y/2f;
			Vector3 pivot = new Vector2(x, y + Utility.Normalise(0f, yMin, yMax, 0f, 1f)*size.y);
			Vector2 tip = new Vector2(x, y + Mathf.Clamp(Normalise(values[i], yMin, yMax, 0f, 1f), 0f, 1f)*size.y);
			DrawGUILine(pivot, tip, colors[i]);
		}
	}

	public static void DrawGUIHorizontalBar(Vector2 center, Vector2 size, Color backgroundColor, float borderWidth, Color borderColor, float fillAmount, Color fillColor) {
		fillAmount = Mathf.Clamp(fillAmount, 0f, 1f);
		DrawGUIRectangleFrame(center, size, borderWidth, borderColor);
		DrawGUIRectangle(new Vector2(center.x - size.x/2f + fillAmount * size.x/2f, center.y), new Vector2(fillAmount * size.x, size.y), fillColor);
	}

	public static void DrawGUIHorizontalPivot(Vector2 center, Vector2 size, Color backgroundColor, float borderWidth, Color borderColor, float pivot, float pivotWidth, Color pivotColor) {
		DrawGUIRectangleFrame(center, size,  borderWidth, borderColor);
		DrawGUIRectangle(new Vector2(center.x - size.x/2f + Normalise(pivot * size.x, 0f, size.x, pivotWidth/2f, size.x - pivotWidth/2f), center.y), new Vector2(pivotWidth, size.y), pivotColor);
	}

	public static void DrawGUIVerticalPivot(Vector2 center, Vector2 size, Color backgroundColor, float borderWidth, Color borderColor, float pivot, float pivotHeight, Color pivotColor) {
		DrawGUIRectangleFrame(center, size,  borderWidth, borderColor);
		DrawGUIRectangle(new Vector2(center.x, center.y - size.y/2f + Normalise(pivot * size.y, 0f, size.y, pivotHeight/2f, size.y - pivotHeight/2f)), new Vector2(size.x, pivotHeight), pivotColor);
	}

	public static void DrawGUICircularPivot(Vector2 center, float size, Color backgroundColor, float degrees, float length, Color pivotColor) {
		degrees = Mathf.Repeat(degrees, 360f);
		DrawGUICircle(center, size, backgroundColor);
		Vector2 end = length * size * (Quaternion.AngleAxis(-degrees, Vector3.forward) * Vector2.up);
		end.x = end.x / Screen.width * Screen.height;
		DrawGUILine(center, center + end, Mathf.Abs(length*size/5f), 0f, pivotColor);
	}

	public static void DrawGUICircularPoint(Vector2 center, float size, Vector2 position, Color backgroundColor, Color pointColor) {
		DrawGUICircle(center, size, backgroundColor);
		Vector2 point = size * position;
		point.x = point.x / Screen.width * Screen.height;
		DrawGUICircle(center + point, size/10f, pointColor);
	}

	//------------------------------------------------------------------------------------------
	//UTILITY FUNCTIONS
	//------------------------------------------------------------------------------------------
	public static Color Transparent(this Color color, float opacity) {
		return new Color(color.r, color.g, color.b, Mathf.Clamp(opacity, 0f, 1f));
	}

	public static Color Lighten(this Color color, float amount) {
		return Color.Lerp(color, Color.white, amount);
	}

	public static Color Darken(this Color color, float amount) {
		return Color.Lerp(color, Color.black, amount);
	}

	public static Color[] GetRainbowColors(int number) {
		Color[] colors = new Color[number];
		for(int i=0; i<number; i++) {
			float frequency = 5f/number;
			colors[i].r = Normalise(Mathf.Sin(frequency*i + 0f) * (127f) + 128f, 0f, 255f, 0f, 1f);
			colors[i].g = Normalise(Mathf.Sin(frequency*i + 2f) * (127f) + 128f, 0f, 255f, 0f, 1f);
			colors[i].b = Normalise(Mathf.Sin(frequency*i + 4f) * (127f) + 128f, 0f, 255f, 0f, 1f);
			colors[i].a = 1f;
		}
		return colors;
	}

	public static Color GetRandomColor() {
		return new Color(Random.value, Random.value, Random.value, 1f);
	}

	public static Rect GetGUIRect(float x, float y, float w, float h) {
		return new Rect(x*Screen.width, y*Screen.height, w*Screen.width, h*Screen.height);
	}

	public static float Normalise(float value, float valueMin, float valueMax, float resultMin, float resultMax) {
		if(valueMax-valueMin != 0f) {
			return (value-valueMin)/(valueMax-valueMin)*(resultMax-resultMin) + resultMin;
		} else {
			//Not possible to normalise input value.
			return value;
		}
	}

	//------------------------------------------------------------------------------------------
	//INTERNAL FUNCTIONS
	//------------------------------------------------------------------------------------------
	private static bool Return() {
		if(!Active) {
			Debug.Log("Drawing is not active. Call 'Begin()' first.");
		}
		return !Active;
	}

	static void Initialise() {
		if(Initialised != null) {
			return;
		}

		Resources.UnloadUnusedAssets();

		GLMaterial = new Material(Shader.Find("Hidden/Internal-Colored"));
		GLMaterial.hideFlags = HideFlags.HideAndDontSave;
		GLMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
		GLMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
		GLMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Back);
		GLMaterial.SetInt("_ZWrite", 1);
		GLMaterial.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.Always);

		MeshMaterial = new Material(Shader.Find("UltiDraw"));
		MeshMaterial.hideFlags = HideFlags.HideAndDontSave;
		MeshMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Back);
		MeshMaterial.SetInt("_ZWrite", 1);
		MeshMaterial.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.Always);
		MeshMaterial.SetFloat("_Power", 0.25f);

		//Meshes
		CircleMesh = CreateCircleMesh(Resolution);
		QuadMesh = GetPrimitiveMesh(PrimitiveType.Quad);
		CubeMesh = GetPrimitiveMesh(PrimitiveType.Cube);
		SphereMesh = GetPrimitiveMesh(PrimitiveType.Sphere);
		CylinderMesh = GetPrimitiveMesh(PrimitiveType.Cylinder);
		CapsuleMesh = GetPrimitiveMesh(PrimitiveType.Capsule);
		ConeMesh = CreateConeMesh(Resolution);
		PyramidMesh = CreatePyramidMesh();
		BoneMesh = CreateBoneMesh();
		//

		//Wires
		CircleWire = CreateCircleWire(Resolution);
		QuadWire = CreateQuadWire();
		CubeWire = CreateCubeWire();
		SphereWire = CreateSphereWire(Resolution);
		CylinderWire = CreateCylinderWire(Resolution);
		CapsuleWire = CreateCapsuleWire(Resolution);
		ConeWire = CreateConeWire(Resolution);
		PyramidWire = CreatePyramidWire();
		BoneWire = CreateBoneWire();
		//
		
		Initialised = new Mesh();
	}

	private static void SetProgram(PROGRAM program) {
		if(Program != program) {
			Program = program;
			GL.End();
			if(Program != PROGRAM.NONE) {
				GLMaterial.SetPass(0);
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

	private static void DrawWire(Vector3[] points, Vector3 position, Quaternion rotation, Vector3 scale, Color color) {
		if(Return()) {return;};
		SetProgram(PROGRAM.LINES);
		GL.Color(color);
		for(int i=0; i<points.Length; i+=2) {
			GL.Vertex(position + rotation * Vector3.Scale(scale, points[i]));
			GL.Vertex(position + rotation * Vector3.Scale(scale, points[i+1]));
		}
	}

	private static Camera GetCamera() {
		if(Camera.current != null) {
			return Camera.current;
		} else {
			return Camera.main;
		}
	}

	private static Mesh GetPrimitiveMesh(PrimitiveType type) {
		GameObject gameObject = GameObject.CreatePrimitive(type);
		gameObject.hideFlags = HideFlags.HideInHierarchy;
		gameObject.GetComponent<MeshRenderer>().enabled = false;
		Mesh mesh = gameObject.GetComponent<MeshFilter>().sharedMesh;
		if(Application.isPlaying) {
			GameObject.Destroy(gameObject);
		} else {
			GameObject.DestroyImmediate(gameObject);
		}
		return mesh;
	}
	
	private static Mesh CreateCircleMesh(int resolution) {
		List<Vector3> vertices = new List<Vector3>();
		List<int> triangles = new List<int>();
		float step = 360.0f / (float)resolution;
		Quaternion quaternion = Quaternion.Euler(0f, 0f, step);
		vertices.Add(new Vector3(0f, 0f, 0f));
		vertices.Add(new Vector3(0f, 0.5f, 0f));
		vertices.Add(quaternion * vertices[1]);
		triangles.Add(1);
		triangles.Add(0);
		triangles.Add(2);
		for(int i=0; i<resolution-1; i++) {
			triangles.Add(vertices.Count - 1);
			triangles.Add(0);
			triangles.Add(vertices.Count);
			vertices.Add(quaternion * vertices[vertices.Count - 1]);
		}
		Mesh mesh = new Mesh();
		mesh.vertices = vertices.ToArray();
		mesh.triangles = triangles.ToArray();     
		return mesh;
	}

	private static Mesh CreateConeMesh(int resolution) {
		List<Vector3> vertices = new List<Vector3>();
		List<int> triangles = new List<int>();
		float step = 360.0f / (float)resolution;
		Quaternion quaternion = Quaternion.Euler(0f, step, 0f);
		vertices.Add(new Vector3(0f, 1f, 0f));
		vertices.Add(new Vector3(0f, 0f, 0f));
		vertices.Add(new Vector3(0f, 0f, 0.5f));
		vertices.Add(quaternion * vertices[2]);
		triangles.Add(2);
		triangles.Add(1);
		triangles.Add(3);
		triangles.Add(2);
		triangles.Add(3);
		triangles.Add(0);
		for(int i=0; i<resolution-1; i++) {
			triangles.Add(vertices.Count-1);
			triangles.Add(1);
			triangles.Add(vertices.Count);
			triangles.Add(vertices.Count-1);
			triangles.Add(vertices.Count);
			triangles.Add(0);
			vertices.Add(quaternion * vertices[vertices.Count-1]);
		}
		Mesh mesh = new Mesh();
		mesh.vertices = vertices.ToArray();
		mesh.triangles = triangles.ToArray();
		mesh.RecalculateNormals();
		return mesh;
	}

	private static Mesh CreatePyramidMesh() {
		List<Vector3> vertices = new List<Vector3>();
		List<int> triangles = new List<int>();
		vertices.Add(new Vector3(-0.5f, 0f, -0.5f));
		vertices.Add(new Vector3(0.5f, 0f, -0.5f));
		vertices.Add(new Vector3(0.5f, 0f, 0.5f));
		vertices.Add(new Vector3(-0.5f, 0f, -0.5f));
		vertices.Add(new Vector3(0.5f, 0f, 0.5f));
		vertices.Add(new Vector3(-0.5f, 0f, 0.5f));
		vertices.Add(new Vector3(-0.5f, 0f, -0.5f));
		vertices.Add(new Vector3(0f, 1f, 0f));
		vertices.Add(new Vector3(0.5f, 0f, -0.5f));
		vertices.Add(new Vector3(0.5f, 0f, -0.5f));
		vertices.Add(new Vector3(0f, 1f, 0f));
		vertices.Add(new Vector3(0.5f, 0f, 0.5f));
		vertices.Add(new Vector3(0.5f, 0f, 0.5f));
		vertices.Add(new Vector3(0f, 1f, 0f));
		vertices.Add(new Vector3(-0.5f, 0f, 0.5f));
		vertices.Add(new Vector3(-0.5f, 0f, 0.5f));
		vertices.Add(new Vector3(0f, 1f, 0f));
		vertices.Add(new Vector3(-0.5f, 0f, -0.5f));
		for(int i=0; i<18; i++) {
			triangles.Add(i);
		}
		Mesh mesh = new Mesh();
		mesh.vertices = vertices.ToArray();
		mesh.triangles = triangles.ToArray();
		mesh.RecalculateNormals();
		return mesh;
	}

	private static Mesh CreateBoneMesh() {
		float size = 1f/7f;
		List<Vector3> vertices = new List<Vector3>();
		List<int> triangles = new List<int>();
		vertices.Add(new Vector3(-size, -size, 0.200f));
		vertices.Add(new Vector3(-size, size, 0.200f));
		vertices.Add(new Vector3(0.000f, 0.000f, 0.000f));
		vertices.Add(new Vector3(size, size, 0.200f));
		vertices.Add(new Vector3(0.000f, 0.000f, 1.000f));
		vertices.Add(new Vector3(size, -size, 0.200f));
		vertices.Add(new Vector3(-size, size, 0.200f));
		vertices.Add(new Vector3(-size, -size, 0.200f));
		vertices.Add(new Vector3(0.000f, 0.000f, 1.000f));
		vertices.Add(new Vector3(size, -size, 0.200f));
		vertices.Add(new Vector3(0.000f, 0.000f, 1.000f));
		vertices.Add(new Vector3(-size, -size, 0.200f));
		vertices.Add(new Vector3(size, size, 0.200f));
		vertices.Add(new Vector3(-size, size, 0.200f));
		vertices.Add(new Vector3(0.000f, 0.000f, 1.000f));
		vertices.Add(new Vector3(size, size, 0.200f));
		vertices.Add(new Vector3(size, -size, 0.200f));
		vertices.Add(new Vector3(0.000f, 0.000f, 0.000f));
		vertices.Add(new Vector3(size, size, 0.200f));
		vertices.Add(new Vector3(0.000f, 0.000f, 0.000f));
		vertices.Add(new Vector3(-size, size, 0.200f));
		vertices.Add(new Vector3(size, -size, 0.200f));
		vertices.Add(new Vector3(-size, -size, 0.200f));
		vertices.Add(new Vector3(0.000f, 0.000f, 0.000f));
		for(int i=0; i<24; i++) {
			triangles.Add(i);
		}
		Mesh mesh = new Mesh();
		mesh.vertices = vertices.ToArray();
		mesh.triangles = triangles.ToArray();
		mesh.RecalculateNormals();
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

		points.Add(new Vector3(0.5f, -0.5f, 0f));
		points.Add(new Vector3(0.5f, 0.5f, 0f));

		points.Add(new Vector3(0.5f, 0.5f, 0f));
		points.Add(new Vector3(-0.5f, 0.5f, 0f));

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

	private static Vector3[] CreateConeWire(int resolution) {
		List<Vector3> points = new List<Vector3>();
		float step = 360.0f / (float)resolution;
		for(int i=0; i<resolution; i++) {
			points.Add(Quaternion.Euler(0f, i*step, 0f) * new Vector3(0f, 0f, 0.5f));
			points.Add(Quaternion.Euler(0f, (i+1)*step, 0f) * new Vector3(0f, 0f, 0.5f));
		}
		points.Add(new Vector3(-0.5f, 0f, 0f));
		points.Add(new Vector3(0f, 1f, 0f));
		points.Add(new Vector3(0.5f, 0f, 0f));
		points.Add(new Vector3(0f, 1f, 0f));
		points.Add(new Vector3(0f, 0f, -0.5f));
		points.Add(new Vector3(0f, 1f, 0f));
		points.Add(new Vector3(0f, 0f, 0.5f));
		points.Add(new Vector3(0f, 1f, 0f));
		return points.ToArray();
	}

	private static Vector3[] CreatePyramidWire() {
		List<Vector3> points = new List<Vector3>();
		points.Add(new Vector3(-0.5f, 0f, -0.5f));
		points.Add(new Vector3(0.5f, 0f, -0.5f));
		points.Add(new Vector3(0.5f, 0f, -0.5f));
		points.Add(new Vector3(0.5f, 0f, 0.5f));
		points.Add(new Vector3(0.5f, 0f, 0.5f));
		points.Add(new Vector3(-0.5f, 0f, 0.5f));
		points.Add(new Vector3(-0.5f, 0f, 0.5f));
		points.Add(new Vector3(-0.5f, 0f, -0.5f));
		points.Add(new Vector3(-0.5f, 0f, -0.5f));
		points.Add(new Vector3(0f, 1f, 0f));
		points.Add(new Vector3(0.5f, 0f, -0.5f));
		points.Add(new Vector3(0f, 1f, 0f));
		points.Add(new Vector3(-0.5f, 0f, 0.5f));
		points.Add(new Vector3(0f, 1f, 0f));
		points.Add(new Vector3(0.5f, 0f, 0.5f));
		points.Add(new Vector3(0f, 1f, 0f));
		return points.ToArray();
	}

	private static Vector3[] CreateBoneWire() {
		float size = 1f/7f;
		List<Vector3> points = new List<Vector3>();
		points.Add(new Vector3(0.000f, 0.000f, 0.000f));
		points.Add(new Vector3(-size, -size, 0.200f));
		points.Add(new Vector3(0.000f, 0.000f, 0.000f));
		points.Add(new Vector3(size, -size, 0.200f));
		points.Add(new Vector3(0.000f, 0.000f, 0.000f));
		points.Add(new Vector3(-size, size, 0.200f));
		points.Add(new Vector3(0.000f, 0.000f, 0.000f));
		points.Add(new Vector3(size, size, 0.200f));
		points.Add(new Vector3(-size, -size, 0.200f));
		points.Add(new Vector3(0.000f, 0.000f, 1.000f));
		points.Add(new Vector3(size, -size, 0.200f));
		points.Add(new Vector3(0.000f, 0.000f, 1.000f));
		points.Add(new Vector3(-size, size, 0.200f));
		points.Add(new Vector3(0.000f, 0.000f, 1.000f));
		points.Add(new Vector3(size, size, 0.200f));
		points.Add(new Vector3(0.000f, 0.000f, 1.000f));
		points.Add(new Vector3(-size, -size, 0.200f));
		points.Add(new Vector3(size, -size, 0.200f));
		points.Add(new Vector3(size, -size, 0.200f));
		points.Add(new Vector3(size, size, 0.200f));
		points.Add(new Vector3(size, size, 0.200f));
		points.Add(new Vector3(-size, size, 0.200f));
		points.Add(new Vector3(-size, size, 0.200f));
		points.Add(new Vector3(-size, -size, 0.200f));
		return points.ToArray();
	}

}
