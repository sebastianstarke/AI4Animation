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


/************************************************************************************

Quick implementation notes:

Grab candidates and grab range:
-------------------------------
The trigger volume on the OVRPlayerController, which also has a GrabManager component,
determines whether an object is in range or out of range.

Hands (via the DistanceGrabber component) determine the target object in one of two
ways, depending on bool m_useSpherecast:
true: cast a sphere of radius m_spherecastRadius at distance m_maxGrabDistance. Select
the first collision.
false: from all objects within the grab volume, select the closest object that can be 
hit by a ray from the player's hand.

IMPORTANT NOTE: if you change the radius of the trigger volume on the 
OVRPlayerController, you must ensure the spherecast or the grab volume on the grabbers
is big enough to reach all objects within that radius! Keep in mind the hand may be a
little behind or two the side of the player, so you need to make it somewhat larger
than the radius. There is no major concern with making it too large (aside from minor
performance questions), because if an object is not in range according to the
OVRPlayerController's trigger volume, it will not be considered for grabbing.

Crosshairs and Outlines:
------------------------

Objects with a DistanceGrabbable component draw their own in range / targeted
highlight. How these states are best presented is highly app-specific.

************************************************************************************/

using UnityEngine;

namespace OculusSampleFramework
{
    public class GrabManager : MonoBehaviour
    {
        Collider m_grabVolume;

        public Color OutlineColorInRange;
        public Color OutlineColorHighlighted;
        public Color OutlineColorOutOfRange;

        void OnTriggerEnter(Collider otherCollider)
        {
            DistanceGrabbable dg = otherCollider.GetComponentInChildren<DistanceGrabbable>();
            if(dg)
            {
                dg.InRange = true;
            }

        }
        
        void OnTriggerExit(Collider otherCollider)
        {
            DistanceGrabbable dg = otherCollider.GetComponentInChildren<DistanceGrabbable>();
            if(dg)
            {
                dg.InRange = false;
            }
        }
    }
}
