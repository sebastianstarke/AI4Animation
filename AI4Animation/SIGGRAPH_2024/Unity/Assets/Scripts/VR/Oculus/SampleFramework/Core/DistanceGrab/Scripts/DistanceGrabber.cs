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
#if UNITY_EDITOR
using UnityEngine.SceneManagement;
#endif



namespace OculusSampleFramework
{
    /// <summary>
    /// Allows grabbing and throwing of objects with the DistanceGrabbable component on them.
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public class DistanceGrabber : OVRGrabber
    {
        // Radius of sphere used in spherecast from hand along forward ray to find target object.
        [SerializeField]
        float m_spherecastRadius = 0;

        // Distance below which no-snap objects won't be teleported, but will instead be left
        // where they are in relation to the hand.
        [SerializeField]
        float m_noSnapThreshhold = 0.05f;

        [SerializeField]
        bool m_useSpherecast;
        public bool UseSpherecast
        {
            get { return m_useSpherecast; }
            set 
            {
                m_useSpherecast = value;
                GrabVolumeEnable(!m_useSpherecast);
            }
        }

        // Public to allow changing in demo.
        [SerializeField]
        public bool m_preventGrabThroughWalls;

        [SerializeField]
        float m_objectPullVelocity = 10.0f;
        float m_objectPullMaxRotationRate = 360.0f; // max rotation rate in degrees per second

        bool m_movingObjectToHand = false;

        // Objects can be distance grabbed up to this distance from the hand.
        [SerializeField]
        float m_maxGrabDistance;

        // Only allow grabbing objects in this layer.
		// NOTE: you can use the value -1 to attempt to grab everything.
        [SerializeField]
        int m_grabObjectsInLayer = 0;
        [SerializeField]
        int m_obstructionLayer = 0;

        DistanceGrabber m_otherHand;

        protected DistanceGrabbable m_target;
        // Tracked separately from m_target, because we support child colliders of a DistanceGrabbable.
        protected Collider m_targetCollider;

        protected override void Start()
        {
            base.Start();

            // Basic hack to guess at max grab distance based on player size.
            // Note that there's no major downside to making this value too high, as objects
            // outside the player's grabbable trigger volume will not be eligible targets regardless.
            Collider sc = m_player.GetComponentInChildren<Collider>();
            if(sc != null)
            {
                m_maxGrabDistance = sc.bounds.size.z * 0.5f + 3.0f;
            }
            else
            {
                m_maxGrabDistance = 12.0f;
            }

            if(m_parentHeldObject == true)
            {
                Debug.LogError("m_parentHeldObject incompatible with DistanceGrabber. Setting to false.");
                m_parentHeldObject = false;
            }

            DistanceGrabber[] grabbers = FindObjectsOfType<DistanceGrabber>();
            for (int i = 0; i < grabbers.Length; ++i)
            {
                if (grabbers[i] != this) m_otherHand = grabbers[i];
            }
            Debug.Assert(m_otherHand != null);

#if UNITY_EDITOR
            OVRPlugin.SendEvent("distance_grabber", (SceneManager.GetActiveScene().name == "DistanceGrab").ToString(), "sample_framework");
#endif
    }

		public override void Update()
        {
            base.Update();

            Debug.DrawRay(transform.position, transform.forward, Color.red, 0.1f);
            
            DistanceGrabbable target;
            Collider targetColl;
            FindTarget(out target, out targetColl);

            if (target != m_target)
            {
                if (m_target != null)
                {
                    m_target.Targeted = m_otherHand.m_target == m_target;
                }
                m_target = target;
                m_targetCollider = targetColl;
                if (m_target != null)
                {
                    m_target.Targeted = true;
                }
            }
        }

        protected override void GrabBegin()
        {
            DistanceGrabbable closestGrabbable = m_target;
            Collider closestGrabbableCollider = m_targetCollider;

            GrabVolumeEnable(false);

            if (closestGrabbable != null)
            {
                if (closestGrabbable.isGrabbed)
                {
                    ((DistanceGrabber)closestGrabbable.grabbedBy).OffhandGrabbed(closestGrabbable);
                }

                m_grabbedObj = closestGrabbable;
                m_grabbedObj.GrabBegin(this, closestGrabbableCollider);
                SetPlayerIgnoreCollision(m_grabbedObj.gameObject, true);

                m_movingObjectToHand = true;
                m_lastPos = transform.position;
                m_lastRot = transform.rotation;

                // If it's within a certain distance respect the no-snap.
                Vector3 closestPointOnBounds = closestGrabbableCollider.ClosestPointOnBounds(m_gripTransform.position);
                if(!m_grabbedObj.snapPosition && !m_grabbedObj.snapOrientation && m_noSnapThreshhold > 0.0f && (closestPointOnBounds - m_gripTransform.position).magnitude < m_noSnapThreshhold)
                {
                    Vector3 relPos = m_grabbedObj.transform.position - transform.position;
                    m_movingObjectToHand = false;
                    relPos = Quaternion.Inverse(transform.rotation) * relPos;
                    m_grabbedObjectPosOff = relPos;
                    Quaternion relOri = Quaternion.Inverse(transform.rotation) * m_grabbedObj.transform.rotation;
                    m_grabbedObjectRotOff = relOri;
                }
                else
                {
                    // Set up offsets for grabbed object desired position relative to hand.
                    m_grabbedObjectPosOff = m_gripTransform.localPosition;
                    if (m_grabbedObj.snapOffset)
                    {
                        Vector3 snapOffset = m_grabbedObj.snapOffset.position;
                        if (m_controller == OVRInput.Controller.LTouch) snapOffset.x = -snapOffset.x;
                        m_grabbedObjectPosOff += snapOffset;
                    }

                    m_grabbedObjectRotOff = m_gripTransform.localRotation;
                    if (m_grabbedObj.snapOffset)
                    {
                        m_grabbedObjectRotOff = m_grabbedObj.snapOffset.rotation * m_grabbedObjectRotOff;
                    }
                }

            }
        }

        protected override void MoveGrabbedObject(Vector3 pos, Quaternion rot, bool forceTeleport = false)
        {
            if (m_grabbedObj == null)
            {
                return;
            }

            Rigidbody grabbedRigidbody = m_grabbedObj.grabbedRigidbody;
            Vector3 grabbablePosition = pos + rot * m_grabbedObjectPosOff;
            Quaternion grabbableRotation = rot * m_grabbedObjectRotOff;

            if (m_movingObjectToHand)
            {
                float travel = m_objectPullVelocity * Time.deltaTime;
                Vector3 dir = grabbablePosition - m_grabbedObj.transform.position;
                if(travel * travel * 1.1f > dir.sqrMagnitude)
                {
                    m_movingObjectToHand = false;
                }
                else
                {
                    dir.Normalize();
                    grabbablePosition = m_grabbedObj.transform.position + dir * travel;
                    grabbableRotation = Quaternion.RotateTowards(m_grabbedObj.transform.rotation, grabbableRotation, m_objectPullMaxRotationRate * Time.deltaTime);
                }
            }
            grabbedRigidbody.MovePosition(grabbablePosition);
            grabbedRigidbody.MoveRotation(grabbableRotation);
        }

        static private DistanceGrabbable HitInfoToGrabbable(RaycastHit hitInfo)
        {
            if (hitInfo.collider != null)
            {
                GameObject go = hitInfo.collider.gameObject;
                return go.GetComponent<DistanceGrabbable>() ?? go.GetComponentInParent<DistanceGrabbable>();
            }
            return null;
        }

        protected bool FindTarget(out DistanceGrabbable dgOut, out Collider collOut)
        {
            dgOut = null;
            collOut = null;
            float closestMagSq = float.MaxValue;

            // First test for objects within the grab volume, if we're using those.
            // (Some usage of DistanceGrabber will not use grab volumes, and will only 
            // use spherecasts, and that's supported.)
            foreach (OVRGrabbable cg in m_grabCandidates.Keys)
            {
                DistanceGrabbable grabbable = cg as DistanceGrabbable;
				bool canGrab = grabbable != null && grabbable.InRange && !(grabbable.isGrabbed && !grabbable.allowOffhandGrab);
				if (canGrab && m_grabObjectsInLayer >= 0) canGrab = grabbable.gameObject.layer == m_grabObjectsInLayer;
                if (!canGrab)
                {
                    continue;
                }

                for (int j = 0; j < grabbable.grabPoints.Length; ++j)
                {
                    Collider grabbableCollider = grabbable.grabPoints[j];
                    // Store the closest grabbable
                    Vector3 closestPointOnBounds = grabbableCollider.ClosestPointOnBounds(m_gripTransform.position);
                    float grabbableMagSq = (m_gripTransform.position - closestPointOnBounds).sqrMagnitude;

                    if (grabbableMagSq < closestMagSq)
                    {
                        bool accept = true;
                        if(m_preventGrabThroughWalls)
                        {
                            // NOTE: if this raycast fails, ideally we'd try other rays near the edges of the object, especially for large objects.
                            // NOTE 2: todo optimization: sort the objects before performing any raycasts.
                            Ray ray = new Ray();
                            ray.direction = grabbable.transform.position - m_gripTransform.position;
                            ray.origin = m_gripTransform.position;
                            RaycastHit obstructionHitInfo;
                            Debug.DrawRay(ray.origin, ray.direction, Color.red, 0.1f);

                            if (Physics.Raycast(ray, out obstructionHitInfo, m_maxGrabDistance, 1 << m_obstructionLayer, QueryTriggerInteraction.Ignore))
                            {
                                float distToObject = (grabbableCollider.ClosestPointOnBounds(m_gripTransform.position) - m_gripTransform.position).magnitude;
                                if(distToObject > obstructionHitInfo.distance * 1.1)
                                {
                                    accept = false;
                                }
                            }
                        }
                        if(accept)
                        {
                            closestMagSq = grabbableMagSq;
                            dgOut = grabbable;
                            collOut = grabbableCollider;
                        }
                    }
                }
            }

            if (dgOut == null && m_useSpherecast)
            {
                return FindTargetWithSpherecast(out dgOut, out collOut);
            }
            return dgOut != null;
        }

        protected bool FindTargetWithSpherecast(out DistanceGrabbable dgOut, out Collider collOut)
        {
            dgOut = null;
            collOut = null;
            Ray ray = new Ray(m_gripTransform.position, m_gripTransform.forward);
            RaycastHit hitInfo;

			// If no objects in grab volume, raycast.
			// Potential optimization: 
			// In DistanceGrabbable.RefreshCrosshairs, we could move the object between collision layers.
			// If it's in range, it would move into the layer DistanceGrabber.m_grabObjectsInLayer,
			// and if out of range, into another layer so it's ignored by DistanceGrabber's SphereCast.
			// However, we're limiting the SphereCast by m_maxGrabDistance, so the optimization doesn't seem
			// essential.
			int layer = (m_grabObjectsInLayer == -1) ? ~0 : 1 << m_grabObjectsInLayer;
            if (Physics.SphereCast(ray, m_spherecastRadius, out hitInfo, m_maxGrabDistance, layer))
            {
                DistanceGrabbable grabbable = null;
                Collider hitCollider = null;
                if (hitInfo.collider != null)
                {
                    grabbable = hitInfo.collider.gameObject.GetComponentInParent<DistanceGrabbable>();
                    hitCollider = grabbable == null ? null : hitInfo.collider;
                    if(grabbable)
                    {
                        dgOut = grabbable;
                        collOut = hitCollider;
                    }
                }

                if (grabbable != null && m_preventGrabThroughWalls)
                {
                    // Found a valid hit. Now test to see if it's blocked by collision.
                    RaycastHit obstructionHitInfo;
                    ray.direction = hitInfo.point - m_gripTransform.position;

                    dgOut = grabbable;
                    collOut = hitCollider;
                    if (Physics.Raycast(ray, out obstructionHitInfo, m_maxGrabDistance, 1 << m_obstructionLayer, QueryTriggerInteraction.Ignore))
                    {
                        DistanceGrabbable obstruction = null;
                        if(hitInfo.collider != null)
                        {
                            obstruction = obstructionHitInfo.collider.gameObject.GetComponentInParent<DistanceGrabbable>();
                        }
                        if (obstruction != grabbable && obstructionHitInfo.distance < hitInfo.distance)
                        {
                            dgOut = null;
                            collOut = null;
                        }
                    }
                }
            }
            return dgOut != null;
        }

        protected override void GrabVolumeEnable(bool enabled)
        {
            if(m_useSpherecast) enabled = false;
            base.GrabVolumeEnable(enabled);
        }

        // Just here to allow calling of a protected member function.
      protected override void OffhandGrabbed(OVRGrabbable grabbable)
        {
            base.OffhandGrabbed(grabbable);
        }
    }
}
