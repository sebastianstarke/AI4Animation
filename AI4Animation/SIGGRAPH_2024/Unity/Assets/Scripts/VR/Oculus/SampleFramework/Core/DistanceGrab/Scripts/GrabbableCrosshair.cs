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
using System.Collections;

namespace OculusSampleFramework
{
    public class GrabbableCrosshair : MonoBehaviour
    {
        public enum CrosshairState { Disabled, Enabled, Targeted }

        CrosshairState m_state = CrosshairState.Disabled;
        Transform m_centerEyeAnchor;

        [SerializeField]
        GameObject m_targetedCrosshair = null;
        [SerializeField]
        GameObject m_enabledCrosshair = null;

        private void Start()
        {
            m_centerEyeAnchor = GameObject.Find("CenterEyeAnchor").transform;
        }

        public void SetState(CrosshairState cs)
        {
            m_state = cs;
            if (cs == CrosshairState.Disabled)
            {
                m_targetedCrosshair.SetActive(false);
                m_enabledCrosshair.SetActive(false);
            }
            else if (cs == CrosshairState.Enabled)
            {
                m_targetedCrosshair.SetActive(false);
                m_enabledCrosshair.SetActive(true);
            }
            else if (cs == CrosshairState.Targeted)
            {
                m_targetedCrosshair.SetActive(true);
                m_enabledCrosshair.SetActive(false);
            }
        }

        private void Update()
        {
            if (m_state != CrosshairState.Disabled)
            {
                transform.LookAt(m_centerEyeAnchor);
            }
        }
    }
}
