/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using System.Collections;

/// <summary>
/// This transition will move the player to the destination over the span of a fixed amount of time.
/// It will not adjust the orientation of the player because this is very uncomfortable.
/// Note there is custom editor for this behavior which is used to control the warp interpolation.
/// </summary>
public class TeleportTransitionWarp : TeleportTransition
{
	/// <summary>
	/// How much time the warp transition takes to complete.
	/// </summary>
	[Tooltip("How much time the warp transition takes to complete.")]
	[Range(0.01f, 1.0f)]
	public float TransitionDuration = 0.5f;

	/// <summary>
	/// Curve to control the position lerp between the current location and the destination.
	/// There is a custom editor for this field to avoid a problem where inspector curves don't update as expected. 
	/// The custom inspector code is here: .\Editor\OVRTeleportTransitionWarpInspector.cs
	/// </summary>
	[HideInInspector]
	public AnimationCurve PositionLerp = AnimationCurve.Linear(0, 0, 1, 1);

	/// <summary>
	/// When the teleport state is entered, quickly move the player to the new location
	/// over the duration of the teleport.
	/// </summary>
	protected override void LocomotionTeleportOnEnterStateTeleporting()
	{
		StartCoroutine(DoWarp());
	}

	/// <summary>
	/// This coroutine will be active during the teleport transition and will move the camera 
	/// according to the PositionLerp curve.
	/// </summary>
	/// <returns></returns>
	IEnumerator DoWarp()
	{
		LocomotionTeleport.IsTransitioning = true;
		var startPosition = LocomotionTeleport.GetCharacterPosition();  
		float elapsedTime = 0;
		while (elapsedTime < TransitionDuration)
		{
			elapsedTime += Time.deltaTime;
			var t = elapsedTime / TransitionDuration;
			var pLerp = PositionLerp.Evaluate(t);
			LocomotionTeleport.DoWarp(startPosition, pLerp);
			yield return null;
		}
		LocomotionTeleport.DoWarp(startPosition, 1.0f);
		LocomotionTeleport.IsTransitioning = false;
	}
}
