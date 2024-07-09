/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Procedurally generated capsule mesh which matches the character controller size.
/// This was originally created for visualizing the capsule in the HMD but it could be adapted to other purposes.
/// </summary>
[ExecuteInEditMode]
public class CharacterCapsule : MonoBehaviour
{
    private CharacterController _character;
    private MeshFilter _meshFilter;

    private float _height;
    private float _radius;

    [Range(4,32)]
    public int SubdivisionsU;

    [Range(4, 32)]
    public int SubdivisionsV;

    private int _subdivisionU;
    private int _subdivisionV;

    private Vector3[] _vertices;
    private int[] _triangles;

    // Update is called once per frame
    void Update ()
	{
	    if (_character == null)
	    {
	        _character = GetComponentInParent<CharacterController>();
	        if (_character == null)
	        {
	            return;
	        }
	    }

	    if (_height == _character.height 
            && _radius == _character.radius 
            && _subdivisionU == SubdivisionsU 
            && _subdivisionV == SubdivisionsV)
	    {
	        return;
	    }

        _height = _character.height;
	    _radius = _character.radius;
	    _subdivisionU = SubdivisionsU;
	    _subdivisionV = SubdivisionsV;

        List<Vector3> verts = new List<Vector3>();

        var vector = new Vector3(1,0,0);

        // Generate the mesh
        var topOffset = new Vector3(0, _height/2.0f - _radius, 0);
        var bottomOffset = new Vector3(0, _radius - _height/2.0f, 0);

        // Add all the necessary vertices
        verts.Add(new Vector3(0, _height/2.0f, 0));

	    for (int u = SubdivisionsU - 1; u >= 0; u--)
	    {
	        float uf = (float) u / (float) SubdivisionsU;
            for (int v = 0; v < SubdivisionsV; v++)
            {
                float vf = (float) v / (float) SubdivisionsV;
                var q = Quaternion.Euler(0, vf * 360.0f, uf * 90.0f);
                var vert = q * vector;
                vert *= _radius;
                var v1 = vert + topOffset;
                verts.Add(v1);
	        }
        }

	    for (int u = 0; u < SubdivisionsU; u++)
	    {
            float uf = (float)u / (float)SubdivisionsU;
	        for (int v = 0; v < SubdivisionsV; v++)
	        {
	            float vf = (float)v / (float)SubdivisionsV;
	            var q = Quaternion.Euler(0, vf * 360.0f + 180.0f, uf * 90.0f);
	            var vert = q * vector;
	            vert *= _radius;
	            var v2 = bottomOffset - vert;
	            verts.Add(v2);
	        }
	    }
	    verts.Add(new Vector3(0, -_height / 2.0f, 0));


        // Setup all the triangles

        List<int> tris = new List<int>();
	    int index;
	    int i;

        // top cap
	    for (int v = 0; v < SubdivisionsV; v++)
	    {
	        i = 0;
	        tris.Add(i);
	        tris.Add(v);
	        tris.Add(v+1);
	    }
	    tris.Add(0);
        tris.Add(SubdivisionsV);
	    tris.Add(1);

        // top hemisphere
        for (int u = 0; u < SubdivisionsU - 1; u++)
	    {
	        index = u * SubdivisionsV + 1;
	        for (int v = 0; v < SubdivisionsV - 1; v++)
            {
                i = index + v;
                tris.Add(i);
                tris.Add(i + SubdivisionsV);
                tris.Add(i + 1);

                tris.Add(i + 1);
                tris.Add(i + SubdivisionsV);
                tris.Add(i + SubdivisionsV + 1);
           }
	        i = index + SubdivisionsV - 1;
	        tris.Add(i);
	        tris.Add(i + SubdivisionsV);
	        tris.Add(i + 1 - SubdivisionsV);

            tris.Add(i + 1 - SubdivisionsV);
            tris.Add(i + SubdivisionsV);
	        tris.Add(i + 1);
        }

        // center tube
        index = (SubdivisionsU - 1) * SubdivisionsV + 1;
	    for (int v = 0; v < SubdivisionsV - 1; v++)
	    {
	        i = index + v;
	        tris.Add(i);
	        tris.Add(i + SubdivisionsV);
	        tris.Add(i + 1);

	        tris.Add(i + 1);
	        tris.Add(i + SubdivisionsV);
	        tris.Add(i + SubdivisionsV + 1);
	    }
	    i = index + SubdivisionsV - 1;
	    tris.Add(i);
	    tris.Add(i + SubdivisionsV);
	    tris.Add(i + 1 - SubdivisionsV);

	    tris.Add(i + 1 - SubdivisionsV);
	    tris.Add(i + SubdivisionsV);
	    tris.Add(i + 1);

        // bottom hemisphere
        for (int u = 0; u < SubdivisionsU - 1; u++)
	    {
	        index = u * SubdivisionsV + (SubdivisionsU * SubdivisionsV) + 1;
	        for (int v = 0; v < SubdivisionsV - 1; v++)
	        {
	            i = index + v;
	            tris.Add(i);
	            tris.Add(i + SubdivisionsV);
	            tris.Add(i + 1);

	            tris.Add(i + 1);
	            tris.Add(i + SubdivisionsV);
	            tris.Add(i + SubdivisionsV + 1);
	        }
	        i = index + SubdivisionsV - 1;
	        tris.Add(i);
	        tris.Add(i + SubdivisionsV);
	        tris.Add(i + 1 - SubdivisionsV);

	        tris.Add(i + 1 - SubdivisionsV);
	        tris.Add(i + SubdivisionsV);
	        tris.Add(i + 1); 
	    }

        // bottom cap
	    var last = verts.Count - 1;
	    var lastRow = last - SubdivisionsV;
	    for (int v = 0; v < SubdivisionsV; v++)
	    {
	        i = 0;
	        tris.Add(last);
	        tris.Add(lastRow + v + 1);
	        tris.Add(lastRow + v);
	    }
	    tris.Add(last);
	    tris.Add(lastRow);
	    tris.Add(last - 1);


        _vertices = verts.ToArray();
	    _triangles = tris.ToArray();

	    _meshFilter = gameObject.GetComponent<MeshFilter>();
        _meshFilter.mesh = new Mesh();
        _meshFilter.sharedMesh.vertices = _vertices;
	    _meshFilter.sharedMesh.triangles = _triangles;
        _meshFilter.sharedMesh.RecalculateNormals();
	}
}

