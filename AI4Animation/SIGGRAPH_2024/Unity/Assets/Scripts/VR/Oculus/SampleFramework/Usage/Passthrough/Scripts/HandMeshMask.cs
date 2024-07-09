using UnityEngine;

/*
 * This script creates a custom mesh, specifically for hand masking in the Passthrough SDK:
 * 
 * 1. Created with 2D screen space in mind, since it's 2D triangles facing the camera.
 *    Not advised to use this mesh in any other way.
 * 2. The look of it should be coupled with maskMaterial, which defines the falloff of the fade and
 *    also how it blends Passthrough.
 * 3. The mesh has UV.x of 0 where the hand is fully opaque, and UV.x of 1 where faded out.
 *    This way, the gradient curve can be tuned in a user-friendly way (instead of in shader).
 *    There may be an optimization of avoiding a texture read, if the fade is all in shader (remapping UV.x)
 * 4. Requires knowledge of the hand bones and indices
 */

public class HandMeshMask : MonoBehaviour
{
    public OVRSkeleton referenceHand;
    public Material maskMaterial;

    // these feel like good defaults
    [Tooltip("The segments around the tip of a finger")]
    public int radialDivisions = 9;
    [Tooltip("The fade range (finger width is 2x this)")]
    public float borderSize = 0.2f;
    [Tooltip("Along the fingers, each knuckle scales down by this amount.  Default is zero for uniform width along entire finger.")]
    public float fingerTaper = 0.13f;
    [Tooltip("Shorten the last bone of each finger; need this to account for bone structure (end bone is at finger tip instead of center). Default is 1.")]
    public float fingerTipLength = 0.8f;
    [Tooltip("Move the base of the 4 main fingers towards the tips, to avoid a visible mesh crack between finger webbing. Default is 0.")]
    public float webOffset = 0.25f;

    // retrieved by OVRHands at runtime
    float handScale = 1.0f;

    // mesh information
    GameObject maskMeshObject;
    Mesh maskMesh;
    Vector3[] handVertices;
    Vector2[] handUVs;
    Color32[] handColors;
    int[] handTriangles;
    int vertCounter = 0;
    int triCounter = 0;

    void Awake()
    {
        // this object must be at the origin for the vertex positions to work
        transform.position = Vector3.zero;

        maskMesh = new Mesh();
        maskMeshObject = new GameObject("MeshMask");
        maskMeshObject.transform.parent = this.transform;
        maskMeshObject.transform.localPosition = Vector3.zero;
        maskMeshObject.AddComponent<MeshFilter>();
        maskMeshObject.AddComponent<MeshRenderer>();
        maskMeshObject.GetComponent<MeshFilter>().mesh = maskMesh;
        maskMeshObject.GetComponent<MeshRenderer>().material = maskMaterial;
    }

    private void Update()
    {
        // must have a minimum amount for math to work out
        radialDivisions = Mathf.Max(2, radialDivisions);

        if (referenceHand)
        {
            // make sure all math accounts for hand scale
            handScale = referenceHand.GetComponent<OVRHand>().HandScale;
            CreateHandMesh();
        }

        bool handsActive = (
          OVRInput.GetActiveController() == OVRInput.Controller.Hands ||
          OVRInput.GetActiveController() == OVRInput.Controller.LHand ||
          OVRInput.GetActiveController() == OVRInput.Controller.RHand);
        maskMeshObject.SetActive(handsActive);
    }

    // do not edit any numbers below unless you like pain
    // (it's EASY to break mesh creation and HARD to debug it)
    // if you do, you need to become intimately aware of the OVRSkeleton.Bone numbers
    void CreateHandMesh()
    {
        int knuckleVerts = 8 + (radialDivisions - 2) * 2;
        int knuckleCount = 25; // five fingers, three knuckles per finger, then 10 more as palm borders

        int palmVerts = 12;
        int palmTriIndices = 66; // 22 triangles, each triangle has three verts

        handVertices = new Vector3[knuckleVerts * knuckleCount + palmVerts];
        handUVs = new Vector2[handVertices.Length];
        handColors = new Color32[handVertices.Length];
        handTriangles = new int[handVertices.Length * 3 + palmTriIndices];
        // these counters are incremented during mesh construction, at each step, to ensure a valid mesh
        vertCounter = 0;
        triCounter = 0;

        // make knuckle meshes for each finger
        for (int i = 0; i < 5; i++)
        {
            int pinkyOffset = i < 4 ? 0 : 1; // pinky bone numbering is slightly different than other fingers
            int baseId = 3 + i * 3 + pinkyOffset;
            int tipId = 19 + i;
            float k0taper = 1.0f;
            float k1taper = 1.0f - fingerTaper;
            float k2taper = 1.0f - (fingerTaper * 2);
            float k3taper = 1.0f - (fingerTaper * 3);
            // thumb also gets a bit wider thickness at the base
            if (i == 0)
            {
                k0taper *= 1.2f;
                k1taper *= 1.1f;
            }
            AddKnuckleMesh(knuckleVerts, k0taper, k1taper, referenceHand.Bones[baseId].Transform.position, referenceHand.Bones[baseId + 1].Transform.position);
            AddKnuckleMesh(knuckleVerts, k1taper, k2taper, referenceHand.Bones[baseId + 1].Transform.position, referenceHand.Bones[baseId + 2].Transform.position);

            // for the tip of the finger, the mask needs to be a bit different:
            // the final joint of the skeleton's finger is at the tip, but
            // we need the final joint to be somewhat inside the finger tip, so the radial mask matches
            Vector3 lastKnucklePos = referenceHand.Bones[baseId + 2].Transform.position;
            Vector3 tipPos = referenceHand.Bones[tipId].Transform.position;
            Vector3 offsetTipPos = (tipPos - lastKnucklePos) * fingerTipLength + lastKnucklePos;
            AddKnuckleMesh(knuckleVerts, k2taper, k3taper, lastKnucklePos, offsetTipPos);
        }

        // add palm mesh, which is very different than fingers
        // it uses the same concept and parameters ("fade out from center")
        AddPalmMesh(knuckleVerts);

        // final step: combine all the arrays to make the mesh object
        maskMesh.Clear();
        maskMesh.name = "HandMeshMask";
        maskMesh.vertices = handVertices;
        maskMesh.uv = handUVs;
        maskMesh.colors32 = handColors;
        maskMesh.triangles = handTriangles;
    }

    // a "knuckle" is a camera-facing mesh from two bones
    // the first two verts are at the bones, the rest border them like a 2D capsule
    void AddKnuckleMesh(int knuckleVerts, float point1scale, float point2scale, Vector3 point1, Vector3 point2)
    {
        int baseVertId = vertCounter;

        Vector3 camPos = Camera.main.transform.position;
        Vector3 fwdVec = (point1 + point2) * 0.5f - camPos; // use the center of the two points
        Vector3 upVec = point2 - point1;
        Vector3.OrthoNormalize(ref fwdVec, ref upVec);
        Vector3 rightVec = Vector3.Cross(upVec, fwdVec);

        AddVertex(point2, Vector2.zero, Color.black);
        AddVertex(point1, Vector2.zero, Color.black);

        int fanVerts = radialDivisions + 1;

        // rotate this vector counter clockwise, making verts along the way
        Vector3 windingVec = rightVec;

        for (int i = 0; i < fanVerts * 2; i++)
        {
            int basePoint = (i / fanVerts) + baseVertId;
            Vector3 vertPos = handVertices[basePoint] + windingVec * borderSize * handScale * (basePoint != baseVertId ? point1scale : point2scale);
            AddVertex(vertPos, new Vector2(1, 0), Color.black);
            if (i != radialDivisions) // after making the first fan, don't wind for one vert
            {
                windingVec = Quaternion.AngleAxis(180.0f / radialDivisions, fwdVec) * windingVec;
            }
        }

        // after vertices have been made, assign their indices to create triangles
        handTriangles[triCounter++] = baseVertId + 0;
        handTriangles[triCounter++] = baseVertId + 1;
        handTriangles[triCounter++] = baseVertId + radialDivisions + 3;
        handTriangles[triCounter++] = baseVertId + 0;
        handTriangles[triCounter++] = baseVertId + radialDivisions + 3;
        handTriangles[triCounter++] = baseVertId + radialDivisions + 2;

        handTriangles[triCounter++] = baseVertId + 2;
        handTriangles[triCounter++] = baseVertId + knuckleVerts - 1;
        handTriangles[triCounter++] = baseVertId + 0;
        handTriangles[triCounter++] = baseVertId + 0;
        handTriangles[triCounter++] = baseVertId + knuckleVerts - 1;
        handTriangles[triCounter++] = baseVertId + 1;

        for (int i = 0; i < radialDivisions; i++)
        {
            handTriangles[triCounter++] = baseVertId + 2 + i;
            handTriangles[triCounter++] = baseVertId + 0;
            handTriangles[triCounter++] = baseVertId + 3 + i;
        }

        for (int i = 0; i < radialDivisions; i++)
        {
            handTriangles[triCounter++] = baseVertId + fanVerts + 2 + i;
            handTriangles[triCounter++] = baseVertId + 1;
            handTriangles[triCounter++] = baseVertId + fanVerts + 3 + i;
        }
    }

    // make the palm section, append it to the mesh
    // study the bone indices and locations in OVRSkeleton.Bones to understand this
    void AddPalmMesh(int knuckleVerts)
    {
        int baseVertId = vertCounter;
        // make a few vertices that aren't bone positions

        // vertex between middle and ring fingers
        Vector3 customVert1 = (referenceHand.Bones[9].Transform.position + referenceHand.Bones[12].Transform.position) * 0.5f;
        // vertex at "saddle" between thumb and index
        Vector3 customVert2 = (referenceHand.Bones[4].Transform.position + referenceHand.Bones[6].Transform.position) * 0.5f;
        customVert2 = (customVert2 - referenceHand.Bones[15].Transform.position) * 0.9f + referenceHand.Bones[15].Transform.position;
        // vertex further up thumb, between bones 4 and 5
        Vector3 thumbPos = (referenceHand.Bones[5].Transform.position - referenceHand.Bones[4].Transform.position) * webOffset;
        thumbPos += referenceHand.Bones[4].Transform.position;
        // at knuckles - move the mesh down the fingers to avoid the ugly mesh gap at finger-webs
        Vector3 indexPos = (referenceHand.Bones[7].Transform.position - referenceHand.Bones[6].Transform.position) * webOffset;
        indexPos += referenceHand.Bones[6].Transform.position;
        Vector3 pinkyPos = (referenceHand.Bones[17].Transform.position - referenceHand.Bones[16].Transform.position) * webOffset;
        pinkyPos += referenceHand.Bones[16].Transform.position;
        Vector3 middlePos = (referenceHand.Bones[10].Transform.position - referenceHand.Bones[9].Transform.position) +
          (referenceHand.Bones[13].Transform.position - referenceHand.Bones[12].Transform.position);
        middlePos *= 0.5f * webOffset;
        middlePos += customVert1;

        // first, make solid low-poly palm
        AddVertex(referenceHand.Bones[0].Transform.position, Vector2.zero, Color.black);  // baseVertId + 0
        AddVertex(referenceHand.Bones[3].Transform.position, Vector2.zero, Color.black);  // +1
        AddVertex(referenceHand.Bones[4].Transform.position, Vector2.zero, Color.black);  // +2
        AddVertex(thumbPos, Vector2.zero, Color.black);                                   // +3
        AddVertex(customVert2, Vector2.zero, Color.black);                                // +4
        AddVertex(referenceHand.Bones[6].Transform.position, Vector2.zero, Color.black);  // +5
        AddVertex(customVert1, Vector2.zero, Color.black);                                // +6
        AddVertex(referenceHand.Bones[15].Transform.position, Vector2.zero, Color.black); // +7
        AddVertex(referenceHand.Bones[16].Transform.position, Vector2.zero, Color.black); // +8
        AddVertex(indexPos, Vector2.zero, Color.black);                                   // +9
        AddVertex(middlePos, Vector2.zero, Color.black);                                  // +10
        AddVertex(pinkyPos, Vector2.zero, Color.black);                                   // +11

        // then, assign triangles
        // unfortunately there's no elegant way to do this
        // this is literally making a low-poly mesh in code (YUCK)

        // palm side (on left hand)
        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 1;
        handTriangles[triCounter++] = baseVertId + 4;

        handTriangles[triCounter++] = baseVertId + 1;
        handTriangles[triCounter++] = baseVertId + 2;
        handTriangles[triCounter++] = baseVertId + 4;

        handTriangles[triCounter++] = baseVertId + 2;
        handTriangles[triCounter++] = baseVertId + 3;
        handTriangles[triCounter++] = baseVertId + 4;

        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 4;
        handTriangles[triCounter++] = baseVertId + 5;

        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 5;
        handTriangles[triCounter++] = baseVertId + 6;

        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 8;

        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 8;
        handTriangles[triCounter++] = baseVertId + 7;

        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 5;
        handTriangles[triCounter++] = baseVertId + 9;

        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 9;
        handTriangles[triCounter++] = baseVertId + 10;

        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 10;
        handTriangles[triCounter++] = baseVertId + 11;

        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 11;
        handTriangles[triCounter++] = baseVertId + 8;

        // back side - to make triangulation easier, it's the same as palm side but reversing the last two verts
        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 4;
        handTriangles[triCounter++] = baseVertId + 1;

        handTriangles[triCounter++] = baseVertId + 1;
        handTriangles[triCounter++] = baseVertId + 4;
        handTriangles[triCounter++] = baseVertId + 2;

        handTriangles[triCounter++] = baseVertId + 2;
        handTriangles[triCounter++] = baseVertId + 4;
        handTriangles[triCounter++] = baseVertId + 3;

        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 5;
        handTriangles[triCounter++] = baseVertId + 4;

        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 5;

        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 8;
        handTriangles[triCounter++] = baseVertId + 6;

        handTriangles[triCounter++] = baseVertId;
        handTriangles[triCounter++] = baseVertId + 7;
        handTriangles[triCounter++] = baseVertId + 8;

        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 9;
        handTriangles[triCounter++] = baseVertId + 5;

        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 10;
        handTriangles[triCounter++] = baseVertId + 9;

        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 11;
        handTriangles[triCounter++] = baseVertId + 10;

        handTriangles[triCounter++] = baseVertId + 6;
        handTriangles[triCounter++] = baseVertId + 8;
        handTriangles[triCounter++] = baseVertId + 11;

        // then, make camera-facing "fins" for the outer fade
        // fortunately these can be knuckle capsules
        // as an optimization, they could just be fins on silhouette edges

        // add fading border to palm mesh
        // between thumb and index
        AddKnuckleMesh(knuckleVerts, 1.0f, 1.0f, customVert2, referenceHand.Bones[6].Transform.position);
        AddKnuckleMesh(knuckleVerts, 1.0f - fingerTaper, 1.0f, thumbPos, customVert2);


        AddKnuckleMesh(knuckleVerts, 1.0f, 1.0f, indexPos, middlePos);
        AddKnuckleMesh(knuckleVerts, 1.0f, 1.0f, middlePos, pinkyPos);

        AddKnuckleMesh(knuckleVerts, 1.0f, 1.0f, referenceHand.Bones[6].Transform.position, customVert1);
        AddKnuckleMesh(knuckleVerts, 1.0f, 1.0f, customVert1, referenceHand.Bones[16].Transform.position);

        AddKnuckleMesh(knuckleVerts, 1.2f, 1.0f, referenceHand.Bones[15].Transform.position, referenceHand.Bones[16].Transform.position);
        AddKnuckleMesh(knuckleVerts, 1.3f, 1.2f, referenceHand.Bones[0].Transform.position, referenceHand.Bones[15].Transform.position);
        AddKnuckleMesh(knuckleVerts, 1.3f, 1.2f, referenceHand.Bones[0].Transform.position, referenceHand.Bones[3].Transform.position);
        AddKnuckleMesh(knuckleVerts, 1.3f, 1.0f, referenceHand.Bones[0].Transform.position, referenceHand.Bones[6].Transform.position);
    }

    void AddVertex(Vector3 position, Vector2 uv, Color color)
    {
        handVertices[vertCounter] = position;
        // UV.x will be the distance from center
        // this way, the shader could decide how best to remap: either with a gradient texture, or directly on UV.x
        handUVs[vertCounter] = uv;
        // using vertex alpha could also be an option for fading, if the UVs are needed elsewhere
        handColors[vertCounter] = color;
        vertCounter++;
    }
}
