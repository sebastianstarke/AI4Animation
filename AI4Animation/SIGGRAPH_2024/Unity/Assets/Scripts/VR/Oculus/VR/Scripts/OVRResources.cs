/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System.Collections.Generic;
using UnityEngine;

public class OVRResources : MonoBehaviour
{
	private static AssetBundle resourceBundle;
	private static List<string> assetNames;

	public static UnityEngine.Object Load(string path)
	{
		if (Debug.isDebugBuild)
		{
			if(resourceBundle == null)
			{
				Debug.Log("[OVRResources] Resource bundle was not loaded successfully");
				return null;
			}

			var result = assetNames.Find(s => s.Contains(path.ToLower()));
			return resourceBundle.LoadAsset(result);
		}
		return Resources.Load(path);
	}
	public static T Load<T>(string path) where T : UnityEngine.Object
	{
		if (Debug.isDebugBuild)
		{
			if (resourceBundle == null)
			{
				Debug.Log("[OVRResources] Resource bundle was not loaded successfully");
				return null;
			}

			var result = assetNames.Find(s => s.Contains(path.ToLower()));
			return resourceBundle.LoadAsset<T>(result);
		}
		return Resources.Load<T>(path);
	}

	public static void SetResourceBundle(AssetBundle bundle)
	{
		resourceBundle = bundle;
		assetNames = new List<string>();
		assetNames.AddRange(resourceBundle.GetAllAssetNames());
	}
}
