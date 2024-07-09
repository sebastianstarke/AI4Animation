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

using System;
using System.IO;
using System.Xml;
using UnityEngine;
using UnityEditor;

public class BuildAssetBundles : MonoBehaviour
{
	[MenuItem("Oculus/Tools/Build Mobile-Quest Expansion File", false, 100000)]
	public static void BuildBundles()
	{
		// Create expansion file directory and call build asset bundles
		string path = Application.dataPath + "/../Asset Bundles/";
		if (!System.IO.Directory.Exists(path))
		{
			System.IO.Directory.CreateDirectory(path);
		}
		BuildPipeline.BuildAssetBundles(path, BuildAssetBundleOptions.ChunkBasedCompression, BuildTarget.Android);

		// Rename asset bundle file to the proper obb string
		if (File.Exists(path + "Asset Bundles"))
		{
			string expansionName = "main." + PlayerSettings.Android.bundleVersionCode + "." + PlayerSettings.applicationIdentifier + ".obb";
			try
			{
				if (File.Exists(path + expansionName))
				{
					File.Delete(path + expansionName);
				}
				File.Move(path + "Asset Bundles", path + expansionName);
				UnityEngine.Debug.Log("OBB expansion file " + expansionName + " has been successfully created at " + path);

				UpdateAndroidManifest();
			}
			catch (Exception e)
			{
				UnityEngine.Debug.LogError(e.Message);
			}
		}
	}

	public static void UpdateAndroidManifest()
	{
		string manifestFolder = Application.dataPath + "/Plugins/Android";
		try
		{
			// Load android manfiest file
			XmlDocument doc = new XmlDocument();
			doc.Load(manifestFolder + "/AndroidManifest.xml");

			string androidNamepsaceURI;
			XmlElement element = (XmlElement)doc.SelectSingleNode("/manifest");
			if(element == null)
			{
				UnityEngine.Debug.LogError("Could not find manifest tag in android manifest.");
				return;
			}

			// Get android namespace URI from the manifest
			androidNamepsaceURI = element.GetAttribute("xmlns:android");
			if (!string.IsNullOrEmpty(androidNamepsaceURI))
			{
				// Check if the android manifest already has the read external storage permission
				XmlNodeList nodeList = doc.SelectNodes("/manifest/application/uses-permission");
				foreach (XmlElement e in nodeList)
				{
					string attr = e.GetAttribute("name", androidNamepsaceURI);
					if (attr == "android.permission.READ_EXTERNAL_STORAGE")
					{
						UnityEngine.Debug.Log("Android manifest already has the proper permissions.");
						return;
					}
				}

				element = (XmlElement)doc.SelectSingleNode("/manifest/application");
				if (element != null)
				{
					// Insert read external storage permission
					XmlElement newElement = doc.CreateElement("uses-permission");
					newElement.SetAttribute("name", androidNamepsaceURI, "android.permission.READ_EXTERNAL_STORAGE");
					element.AppendChild(newElement);

					doc.Save(manifestFolder + "/AndroidManifest.xml");
					UnityEngine.Debug.Log("Successfully modified android manifest with external storage permission.");
					return;
				}
			}

			UnityEngine.Debug.LogError("Could not find android naemspace URI in android manifest.");
		}
		catch (Exception e)
		{
			UnityEngine.Debug.LogError(e.Message);
		}
	}
}
