/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using System;
using UnityEngine;
using System.Collections;

public class TeleportAimVisualLaser : TeleportSupport
{
	/// <summary>
	/// This prefab will be instantiated when the aim visual is awakened, and will be set active when the 
	/// user is aiming, and deactivated when they are done aiming.
	/// </summary>
	[Tooltip("This prefab will be instantiated when the aim visual is awakened, and will be set active when the user is aiming, and deactivated when they are done aiming.")]
	public LineRenderer LaserPrefab;

	private readonly Action _enterAimStateAction;
	private readonly Action _exitAimStateAction;
	private readonly Action<LocomotionTeleport.AimData> _updateAimDataAction;
	private LineRenderer _lineRenderer;
	private Vector3[] _linePoints;

	public TeleportAimVisualLaser()
	{
		_enterAimStateAction = EnterAimState;
		_exitAimStateAction = ExitAimState;
		_updateAimDataAction = UpdateAimData;
	}

	private void EnterAimState()
	{
		_lineRenderer.gameObject.SetActive(true);
	}

	private void ExitAimState()
	{
		_lineRenderer.gameObject.SetActive(false);
	}

	void Awake()
	{
		LaserPrefab.gameObject.SetActive(false);
		_lineRenderer = Instantiate(LaserPrefab);
	}

	protected override void AddEventHandlers()
	{
		base.AddEventHandlers();
		LocomotionTeleport.EnterStateAim += _enterAimStateAction;
		LocomotionTeleport.ExitStateAim += _exitAimStateAction;
		LocomotionTeleport.UpdateAimData += _updateAimDataAction;
	}

	/// <summary>
	/// Derived classes that need to use event handlers need to override this method and
	/// call the base class to ensure all event handlers are removed as intended.
	/// </summary>
	protected override void RemoveEventHandlers()
	{
		LocomotionTeleport.EnterStateAim -= _enterAimStateAction;
		LocomotionTeleport.ExitStateAim -= _exitAimStateAction;
		LocomotionTeleport.UpdateAimData -= _updateAimDataAction;
		base.RemoveEventHandlers();
	}

	private void UpdateAimData(LocomotionTeleport.AimData obj)
	{
		_lineRenderer.sharedMaterial.color = obj.TargetValid ? Color.green : Color.red;

		var points = obj.Points;
        //        Debug.Log("AimVisualLaser: count: " + points.Count);
        _lineRenderer.positionCount = points.Count;
		//_lineRenderer.SetVertexCount(points.Count);
		for (int i = 0; i < points.Count; i++)
		{
			_lineRenderer.SetPosition(i, points[i]);
		}
	}
}
