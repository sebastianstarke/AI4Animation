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

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace OculusSampleFramework
{
    public class DistanceGrabberSample : MonoBehaviour
    {

        bool useSpherecast = false;
        bool allowGrabThroughWalls = false;

        public bool UseSpherecast
        {
            get { return useSpherecast; }
            set
            {
                useSpherecast = value;
                for (int i = 0; i < m_grabbers.Length; ++i)
                {
                    m_grabbers[i].UseSpherecast = useSpherecast;
                }
            }
        }

        public bool AllowGrabThroughWalls
        {
            get { return allowGrabThroughWalls; }
            set
            {
                allowGrabThroughWalls = value;
                for (int i = 0; i < m_grabbers.Length; ++i)
                {
                    m_grabbers[i].m_preventGrabThroughWalls = !allowGrabThroughWalls;
                }
            }
        }

        [SerializeField]
        DistanceGrabber[] m_grabbers = null;

        // Use this for initialization
        void Start()
        {
            DebugUIBuilder.instance.AddLabel("Distance Grab Sample");
            DebugUIBuilder.instance.AddToggle("Use Spherecasting", ToggleSphereCasting, useSpherecast);
            DebugUIBuilder.instance.AddToggle("Grab Through Walls", ToggleGrabThroughWalls, allowGrabThroughWalls);
            DebugUIBuilder.instance.Show();

			// Forcing physics tick rate to match game frame rate, for improved physics in this sample.
			// See comment in OVRGrabber.Update for more information.
			float freq = OVRManager.display.displayFrequency;
			if(freq > 0.1f)
			{
				Debug.Log("Setting Time.fixedDeltaTime to: " + (1.0f / freq));
				Time.fixedDeltaTime = 1.0f / freq;
			}
        }

        public void ToggleSphereCasting(Toggle t)
        {
            UseSpherecast = !UseSpherecast;
        }

        public void ToggleGrabThroughWalls(Toggle t)
        {
            AllowGrabThroughWalls = !AllowGrabThroughWalls;
        }
    }
}
