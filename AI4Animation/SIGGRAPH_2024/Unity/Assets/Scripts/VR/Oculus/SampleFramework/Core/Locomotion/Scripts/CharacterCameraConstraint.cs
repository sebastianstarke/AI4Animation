/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific
language governing permissions and limitations under the license.

************************************************************************************/

using System;
using System.Collections;
using UnityEngine;

/// <summary>
/// This component is responsible for moving the character capsule to match the HMD, fading out the camera or blocking movement when
/// collisions occur, and adjusting the character capsule height to match the HMD's offset from the ground.
/// </summary>
public class CharacterCameraConstraint : MonoBehaviour
{
	// Distance in front of the camera at which fading begins.
	private const float FADE_RAY_LENGTH = 0.25f;
	// How much overlap before fully faded.
	private const float FADE_OVERLAP_MAXIMUM = 0.1f;
	// Fully faded amount, which can be adjusted for debugging purposes.
	private const float FADE_AMOUNT_MAXIMUM = 1.0f;

	/// <summary>
	/// This should be a reference to the OVRCameraRig that is usually a child of the PlayerController.
	/// </summary>
	[Tooltip("This should be a reference to the OVRCameraRig that is usually a child of the PlayerController.")]
	public OVRCameraRig CameraRig;

	/// <summary>
	/// Collision layers to be used for the purposes of fading out the screen when the HMD is inside world geometry and adjusting the capsule height.
	/// </summary>
	[Tooltip("Collision layers to be used for the purposes of fading out the screen when the HMD is inside world geometry and adjusting the capsule height.")]
	public LayerMask CollideLayers;

	/// <summary>
	/// Offset is added to camera's real world height, effectively treating it as though the player was taller/standing higher.
	/// </summary>
	[Tooltip("Offset is added to camera's real world height, effectively treating it as though the player was taller/standing higher.")]
	public float HeightOffset;

	/// <summary>
	/// Minimum height that the character capsule can shrink to.  To disable, set to capsule's height.
	/// </summary>
	[Tooltip("Minimum height that the character capsule can shrink to.  To disable, set to capsule's height.")]
	public float MinimumHeight;

	/// <summary>
	/// Maximum height that the character capsule can grow to.  To disable, set to capsule's height.
	/// </summary>
	[Tooltip("Maximum height that the character capsule can grow to.  To disable, set to capsule's height.")]
	public float MaximumHeight;

	private CapsuleCollider _character;
	private SimpleCapsuleWithStickMovement _simplePlayerController;

	CharacterCameraConstraint()
	{
	}

	private void Awake ()
	{
		_character = GetComponent<CapsuleCollider>();
		_simplePlayerController = GetComponent<SimpleCapsuleWithStickMovement>();
	}

	private void OnEnable()
	{
		_simplePlayerController.CameraUpdated += CameraUpdate;
	}

	private void OnDisable()
	{
		_simplePlayerController.CameraUpdated -= CameraUpdate;
	}

	/// <summary>
	/// This method is the handler for the PlayerController.CameraUpdated event, which is used
	/// to handle whether or not the screen should fade out due to the camera's position (such
	/// as in a wall), and update the character height based on camera position.
	///
	/// Future work: Have the character capsule attempt to match the camera's X,Z position so
	/// that hit detection and dodging work as expected.  Presently the capsule only grows up
	/// and down, but doesn't otherwise change based on camera position.
	/// </summary>
	private void CameraUpdate()
	{
		// Check whether the camera is inside of geometry or on the other side of geometry.
		// Then check if any geometry would be clipped.  Fade out screen accordingly.
		float clippingOverlap = 0.0f;
		if (CheckCameraOverlapped())
		{
			OVRScreenFade.instance.SetExplicitFade(FADE_AMOUNT_MAXIMUM);
		}
		else if (CheckCameraNearClipping(out clippingOverlap))
		{
			// Calculate a `t` value based on where in the interval [0.0, FADE_OVERLAP_MAXIMUM]
			// that our actual overlap lies, and then use that `t` value to calulate an
			// actual fade amount on the interval [0.0, FADE_AMOUNT_MAXIMUM].
			// Note: Both math helper functions clamp `t` to within [0.0, 1.0].
			float fadeParameter = Mathf.InverseLerp(0.0f, FADE_OVERLAP_MAXIMUM, clippingOverlap);
			float fadeAlpha = Mathf.Lerp(0.0f, FADE_AMOUNT_MAXIMUM, fadeParameter);
			OVRScreenFade.instance.SetExplicitFade(fadeAlpha);
		}
		else
		{
			OVRScreenFade.instance.SetExplicitFade(0.0f);
		}

		// Offset the camera into the capsule that's used so that it doesn't scrape any
		// overhanging geometry that the player can barely fit through.  Currently just
		// based on where we start fading.
		float capsuleOffset = FADE_RAY_LENGTH;

		// Calculate the current height by re-adding the height offset to the camera
		// position.
		float calculatedHeight = CameraRig.centerEyeAnchor.localPosition.y + HeightOffset + capsuleOffset;

		// Calculate the minimum allowable capsule height, based on the current value, and
		// the configured minimum value.  Current height is always a valid value to ensure
		// we don't change the capsule height for any reason other than the camera position
		// changing.
		float calculatedMinimumHeight = MinimumHeight;
		calculatedMinimumHeight = Mathf.Min(_character.height, calculatedMinimumHeight);

		// Calculate the maximum allowable capsule height, based on the current value, the
		// configured maximum value and whether there is something overhead.  Current
		// height is always a valid value to ensure we don't change the capsule height for
		// any reason other than the camera position changing.
		float calculatedMaximumHeight = MaximumHeight;
		RaycastHit heightHitInfo;
		if (Physics.SphereCast(_character.transform.position, _character.radius * 0.2f, Vector3.up, out heightHitInfo,
			MaximumHeight - _character.transform.position.y, CollideLayers, QueryTriggerInteraction.Ignore))
		{
			calculatedMaximumHeight = heightHitInfo.point.y;
		}
		calculatedMaximumHeight = Mathf.Max(_character.height, calculatedMaximumHeight);

		// Finally adjust capsule height based on camera position, clamped to our acceptable
		// minimum and maximum values.
		_character.height = Mathf.Clamp(calculatedHeight, calculatedMinimumHeight, calculatedMaximumHeight);

		// Offset the height of the camera to account for the changing capsule height.
		// We want to select a height offset that places the camera near the top of the
		// capsule.  We offset the camera down from the top of the capsule to prevent
		// the screen from fading while the capsule barely fits under an overhang.
		float cameraRigHeightOffset = HeightOffset - (_character.height * 0.5f) - capsuleOffset;
		CameraRig.transform.localPosition = new Vector3(0.0f, cameraRigHeightOffset, 0.0f);
	}

	/// <summary>
	/// This method checks whether the camera is inside of geometry or has geometry between it
	/// and the character's capsule.
	/// </summary>
	private bool CheckCameraOverlapped()
	{
		Camera camera = CameraRig.centerEyeAnchor.GetComponent<Camera>();

		// Use a ray from the capsule starting at the camera's height, but clamped
		// to make sure it comes from the capsule.  We clamp slightly inside of
		// the capsule to account for the sphere cast radius and a small offset
		// in case things would otherwise be touching.
		Vector3 origin = _character.transform.position;
		float yOffset = Mathf.Max(0.0f, (_character.height * 0.5f) - camera.nearClipPlane - 0.01f);
		origin.y = Mathf.Clamp(CameraRig.centerEyeAnchor.position.y, _character.transform.position.y - yOffset, _character.transform.position.y + yOffset);
		Vector3 delta = CameraRig.centerEyeAnchor.position - origin;
		float distance = delta.magnitude;
		Vector3 direction = delta / distance;
		RaycastHit hitInfo;
		return Physics.SphereCast(origin, camera.nearClipPlane, direction, out hitInfo, distance, CollideLayers, QueryTriggerInteraction.Ignore);
	}

	/// <summary>
	/// This method checks whether the camera is close enough to geometry to
	/// cause it to start clipping, and if so, return the overlap amount.
	/// </summary>
	private bool CheckCameraNearClipping(out float result)
	{
		Camera camera = CameraRig.centerEyeAnchor.GetComponent<Camera>();

		Vector3[] frustumCorners = new Vector3[4];
		camera.CalculateFrustumCorners(new Rect(0, 0, 1, 1), camera.nearClipPlane, Camera.MonoOrStereoscopicEye.Mono, frustumCorners);

		// Cast a ray through each corner of the frustum and the center, and take the
		// maximum overlap (if any) returned as the basis to decide how much to fade.
		Vector3 frustumBottomLeft = CameraRig.centerEyeAnchor.position + (Vector3.Normalize(CameraRig.centerEyeAnchor.TransformVector(frustumCorners[0])) * FADE_RAY_LENGTH);
		Vector3 frustumTopLeft = CameraRig.centerEyeAnchor.position + (Vector3.Normalize(CameraRig.centerEyeAnchor.TransformVector(frustumCorners[1])) * FADE_RAY_LENGTH);
		Vector3 frustumTopRight = CameraRig.centerEyeAnchor.position + (Vector3.Normalize(CameraRig.centerEyeAnchor.TransformVector(frustumCorners[2])) * FADE_RAY_LENGTH);
		Vector3 frustumBottomRight = CameraRig.centerEyeAnchor.position + (Vector3.Normalize(CameraRig.centerEyeAnchor.TransformVector(frustumCorners[3])) * FADE_RAY_LENGTH);
		Vector3 frustumCenter = (frustumTopLeft + frustumBottomRight) / 2.0f;

		bool hit = false;
		result = 0.0f;
		foreach (Vector3 frustumPoint in new Vector3[] { frustumBottomLeft, frustumTopLeft, frustumTopRight, frustumBottomRight, frustumCenter })
		{
			RaycastHit hitInfo;
			if (Physics.Linecast(CameraRig.centerEyeAnchor.position, frustumPoint, out hitInfo, CollideLayers, QueryTriggerInteraction.Ignore))
			{
				hit = true;
				result = Mathf.Max(result, Vector3.Distance(hitInfo.point, frustumPoint));
			}
		}
		return hit;
	}
}
