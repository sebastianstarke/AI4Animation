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

#if UNITY_EDITOR
using UnityEditor;

[InitializeOnLoadAttribute]
public class OculusSampleFrameworkUtil
{
  static OculusSampleFrameworkUtil()
  {
#if UNITY_2017_2_OR_NEWER
    EditorApplication.playModeStateChanged += HandlePlayModeState;
#else
    EditorApplication.playmodeStateChanged += () =>
    {
      if (EditorApplication.isPlaying)
      {
        OVRPlugin.SendEvent("load", OVRPlugin.wrapperVersion.ToString(), "sample_framework");
      }
    };
#endif
	}

#if UNITY_2017_2_OR_NEWER
	private static void HandlePlayModeState(PlayModeStateChange state)
  {
    if (state == PlayModeStateChange.EnteredPlayMode)
    {
      OVRPlugin.SendEvent("load", OVRPlugin.wrapperVersion.ToString(), "sample_framework");
    }
  }
#endif
}

#endif
