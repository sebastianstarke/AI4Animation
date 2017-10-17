using UnityEngine;
using System.IO;
using System.Collections.Generic;

public class World : MonoBehaviour {

	private Terrain Terrain;
	private TerrainCollider Collider;
	private TerrainData Data;

	void Start() {
		Terrain = gameObject.AddComponent<Terrain>();
		Collider = gameObject.AddComponent<TerrainCollider>();
		
		Data = new TerrainData();
		float[,] heights = ReadHeights("../PFNN/demo/heightmaps/hmap_013_smooth_ao.txt");
		int resolution = 4096;
		Data.heightmapResolution = resolution;
		Data.size = new Vector3(heights.GetLength(0),1,heights.GetLength(1));
		Data.SetHeights(0,0,heights);
		Terrain.terrainData = Data;
		Collider.terrainData = Data;
	}

	private float[,] ReadHeights(string fn) {
		try {
			StreamReader reader = new StreamReader(fn);
			List<List<float>> data = new List<List<float>>();
			string line;
			while ((line = reader.ReadLine()) != null) {
				string[] entries = line.Split(' ');
				float[] values = new float[entries.Length];
				for(int i=0; i<values.Length; i++) {
					values[i] = float.Parse(entries[i], System.Globalization.CultureInfo.InvariantCulture);
				}
				data.Add(new List<float>(values));
			}
			reader.Close();
			int rows = data.Count;
			int cols = data[0].Count;
			float[,] heights = new float[rows,cols];
			for(int x=0; x<rows; x++) {
				for(int y=0; y<cols; y++) {
					heights[x,y] = data[x][y];
				}
			}
			return heights;
		} catch (System.Exception e) {
        	Debug.Log(e.Message);
			return null;
    	}
	}

}
