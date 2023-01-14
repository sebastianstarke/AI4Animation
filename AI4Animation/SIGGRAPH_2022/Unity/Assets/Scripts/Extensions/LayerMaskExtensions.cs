using UnityEngine;
using System.Collections.Generic;
 
public static class LayerMaskExtensions
{
	public static LayerMask Create(params string[] layerNames)
	{
		return NamesToMask(layerNames);
	}
 
	public static LayerMask Create(params int[] layerNumbers)
	{
		return LayerNumbersToMask(layerNumbers);
	}
 
	public static LayerMask NamesToMask(params string[] layerNames)
	{
		LayerMask ret = (LayerMask)0;
		foreach(var name in layerNames)
		{
			ret |= (1 << LayerMask.NameToLayer(name));
		}
		return ret;
	}
 
	public static LayerMask LayerNumbersToMask(params int[] layerNumbers)
	{
		LayerMask ret = (LayerMask)0;
		foreach(var layer in layerNumbers)
		{
			ret |= (1 << layer);
		}
		return ret;
	}
 
	public static LayerMask Inverse(this LayerMask original)
	{
		return ~original;
	}
 
	public static LayerMask AddToMask(this LayerMask original, params string[] layerNames)
	{
		return original | NamesToMask(layerNames);
	}
 
	public static LayerMask RemoveFromMask(this LayerMask original, params string[] layerNames)
	{
		LayerMask invertedOriginal = ~original;
		return ~(invertedOriginal | NamesToMask(layerNames));
	}
 
	public static string[] MaskToNames(this LayerMask original)
	{
		var output = new List<string>();
 
		for (int i = 0; i < 32; ++i)
		{
			int shifted = 1 << i;
			if ((original & shifted) == shifted)
			{
				string layerName = LayerMask.LayerToName(i);
				if (!string.IsNullOrEmpty(layerName))
				{
					output.Add(layerName);
				}
			}
		}
		return output.ToArray();
	}
 
	public static string MaskToString(this LayerMask original)
	{
		return MaskToString(original, ", ");
	}
 
	public static string MaskToString(this LayerMask original, string delimiter)
	{
		return string.Join(delimiter, MaskToNames(original));
	}

	public static bool Contains(this LayerMask mask, int layer) 
	{ 
		return mask == (mask | (1 << layer)); 
	}

	public static bool Contains(this LayerMask mask, string layer){ 
		return mask.Contains(LayerMask.NameToLayer(layer)); 
	}
}