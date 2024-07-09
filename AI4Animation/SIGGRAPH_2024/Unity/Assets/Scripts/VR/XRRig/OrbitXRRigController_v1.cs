// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using AI4Animation;

// public class OrbitXRRigController_v1 : ManualXRRigController
// {
//     [Tooltip("Position offset on camera")]
//     public Vector3 SelfOffset = Vector3.zero;
//     [Tooltip("Position offset on focus target (HMD)")]
//     public Vector3 TargetOffset = Vector3.zero;
//     [Tooltip("Damping on camera motion")]
//     [Range(0f, 1f)] public float Damping = 0f;
//     protected const float OrbitingSensitivity = 90f;
//     protected Vector2 Orbiting = Vector2.up;
//     protected Actor Character = null;

//     public override void Setup(LiveTracking device, XRRig xrRig)
//     {
//         base.Setup(device, xrRig);

//         // Find target actor to orbit around
//         var liveTracking = device as LiveTracking;
//         if(liveTracking == null)
//         {
//             var actors = FindObjectsOfType<Actor>();
//             foreach(var actor in actors)
//             {
//                 if(actor.gameObject.activeSelf)
//                 {
//                     Character = actor;
//                     break;
//                 }
//             }
//         }
//         else
//         {
//             Character = liveTracking.Character;
//         }
//     }

//     protected override void UpdateHMD()
//     {
//         Transform camera = _xrRig.HMDAnchor.transform;
//         Matrix4x4 head = Device.GetTransformation(LiveTracking.TRACKER_TYPE.HEAD);

//         //Save previous coordinates
//         Vector3 previousPosition = camera.position;
//         Quaternion previousRotation = camera.rotation;

//         //Positioning the camera around the target and focusing at it
//         Vector3 selfOffset = SelfOffset;
//         Vector3 targetOffset = TargetOffset + new Vector3(0f, head.GetPosition().y, 0f);
//         Vector3 pivot = Character.transform.position + Character.transform.rotation * targetOffset;
//         camera.position = Character.transform.position + Character.transform.rotation * (Mathf.Abs(Orbiting.y) * selfOffset);
//         camera.rotation = head.GetRotation();
//         camera.RotateAround(pivot, Vector3.up, -Orbiting.x);
//         camera.LookAt(pivot);

//         //Lerp camera from previous to new coordinates
//         camera.position = Vector3.Lerp(previousPosition, camera.position, 1f - Damping);
//         camera.rotation = Quaternion.Slerp(previousRotation, camera.rotation, 1f - Damping);
//     }

//     public void Orbit(Vector2 axis)
//     {
//         Orbiting.x += OrbitingSensitivity * axis.x * Time.deltaTime;
//         Orbiting.y = Mathf.Max(0f, Orbiting.y - axis.y * Time.deltaTime);
//     }
// }
