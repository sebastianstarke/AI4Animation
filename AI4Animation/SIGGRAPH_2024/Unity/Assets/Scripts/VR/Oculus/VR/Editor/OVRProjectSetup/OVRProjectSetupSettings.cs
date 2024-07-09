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

using UnityEngine;
using UnityEditor;
using System;

[System.Serializable]
public class OVRProjectSetupSettings : ScriptableObject
{
	[Serializable]
	public class BoolProperties : SerializableDictionary<string, bool> {}

	private const string AssetName = "OculusProjectSetupSettings.asset";

    [SerializeField] private BoolProperties boolProperties = new BoolProperties();

    private static OVRProjectSetupSettings _config = null;
    private static string _configPath = null;

    public bool GetProjectSetupBool(string key, bool defaultValue)
    {
        if (!boolProperties.TryGetValue(key, out var value))
        {
            // To avoid clutter, getter doesn't add to the dictionary
            value = defaultValue;
        }

        return value;
    }

    public void SetProjectSetupBool(string key, bool value)
    {
	    boolProperties[key] = value;
        EditorUtility.SetDirty(this);
    }

    public void RemoveProjectSetupBool(string key)
    {
	    boolProperties.Remove(key);
        EditorUtility.SetDirty(this);
    }

    private static string GetOculusProjectConfigAssetPath(bool refresh = false)
    {
	    if (_configPath != null && !refresh)
	    {
		    return _configPath;
	    }

	    // Using the same Path logic as OVRProjectConfig
	    _configPath = OVRProjectConfig.ComputeOculusProjectAssetPath(AssetName);
        return _configPath;
    }

    public static OVRProjectSetupSettings GetProjectConfig(bool refresh = false, bool create = true)
    {
	    if (_config != null && !refresh)
	    {
		    return _config;
	    }

        var oculusProjectConfigAssetPath = GetOculusProjectConfigAssetPath(refresh:false);
        try
        {
	        _config = AssetDatabase.LoadAssetAtPath(oculusProjectConfigAssetPath, typeof(OVRProjectSetupSettings)) as OVRProjectSetupSettings;
        }
        catch (System.Exception e)
        {
            Debug.LogWarningFormat("Unable to load ProjectSetupConfig from {0}, error {1}", oculusProjectConfigAssetPath, e.Message);
        }

        if (_config == null && create && !BuildPipeline.isBuildingPlayer)
        {
	        Debug.LogFormat("Creating ProjectSetupConfig at path {0}", oculusProjectConfigAssetPath);
	        _config = ScriptableObject.CreateInstance<OVRProjectSetupSettings>();
            AssetDatabase.CreateAsset(_config, oculusProjectConfigAssetPath);
        }

        return _config;
    }
}
