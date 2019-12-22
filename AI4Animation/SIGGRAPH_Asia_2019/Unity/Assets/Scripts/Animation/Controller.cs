using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class Controller {

	public RaycastHit Projection;
	public bool ProjectionValid;
	public bool ProjectionActive;
	public Interaction ProjectionInteraction;
	public Interaction ActiveInteraction;
	public Interaction SelectedInteraction;

	public Signal[] Signals = new Signal[0];

	private enum OPERATION {Translate, Rotate, Scale};
	private OPERATION Operation = OPERATION.Translate;
	private float Sensitivity = 10f;
	private float Smoothing = 0.5f;
	private bool ProjectSurface = true;
	private bool LockX = true;
	private bool LockY = false;
	private bool LockZ = true;
	private Vector2 LastMousePosition;
	private Vector3 Offset;

	public void Update() {
		Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
		// if(ActiveInteraction == null) {
			RaycastHit[] hits = Physics.RaycastAll(ray, float.PositiveInfinity, LayerMask.GetMask("Default", "Ground", "Interaction"));
			ProjectionValid = false;
			float dMin = float.MaxValue;
			for(int i=0; i<hits.Length; i++) {
				float dNew = Vector3.Distance(ray.origin, hits[i].point);
				if(dNew < dMin) {
					dMin = dNew;
					Projection = hits[i];
					ProjectionValid = true;
				}
				// if(hits[i].transform == ActiveInteraction.transform || hits[i].transform.parent == ActiveInteraction.transform || hits[i].collider.isTrigger) {
				// } else {
				// 	float dNew = Vector3.Distance(ray.origin, hits[i].point);
				// 	if(dNew < dMin) {
				// 		dMin = dNew;
				// 		Projection = hits[i];
				// 		ProjectionValid = true;
				// 	}
				// }
			}
		// } else {
			// ProjectionValid = Physics.Raycast(ray, out Projection, float.PositiveInfinity, LayerMask.GetMask("Default", "Ground", "Interaction"));
		// }
		ProjectionActive = Input.GetMouseButton(1);
		ProjectionInteraction = !ProjectionValid ? null : Projection.transform.GetComponent<Interaction>();
		if(Input.GetMouseButtonDown(0) && ProjectionInteraction != null && SelectedInteraction == null) {
			SelectedInteraction = ProjectionInteraction;
			float screenDistance = Camera.main.WorldToScreenPoint(SelectedInteraction.transform.position).z;
			Offset = Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenDistance)) - SelectedInteraction.transform.position;
		}
		if(Input.GetMouseButtonUp(0)) {
			SelectedInteraction = null;
			Offset = Vector3.zero;
		}
		if(Input.GetMouseButton(0) && SelectedInteraction != null) {
			if(Input.GetKey(KeyCode.R)) {
				Operation = OPERATION.Rotate;
			} else if(Input.GetKey(KeyCode.T)) {
				Operation = OPERATION.Scale;
			} else {
				Operation = OPERATION.Translate;
			}
			Move();
		}
		LastMousePosition = Input.mousePosition;
	}

	private void Move() {
		if(Operation == OPERATION.Translate) {
			float screenDistance = Camera.main.WorldToScreenPoint(SelectedInteraction.transform.position).z;
			Vector3 current = SelectedInteraction.transform.position;
			Vector3 target = Camera.main.ScreenToWorldPoint(new Vector3(Input.mousePosition.x, Input.mousePosition.y, screenDistance)) - Offset;
			SelectedInteraction.transform.position = Vector3.Lerp(current, target, 1f-Smoothing);
			if(ProjectSurface) {
				SelectedInteraction.transform.position = Utility.ProjectGround(SelectedInteraction.transform.position, LayerMask.GetMask("Default", "Ground"));
			}
		}
		if(Operation == OPERATION.Rotate) {
			Vector2 deltaMousePosition = GetNormalizedMousePosition(Input.mousePosition) - GetNormalizedMousePosition(LastMousePosition);
			Vector3 prev = SelectedInteraction.transform.eulerAngles;
			SelectedInteraction.transform.Rotate(Camera.main.transform.right, Sensitivity/Time.deltaTime*deltaMousePosition.y, Space.World);
			SelectedInteraction.transform.Rotate(Camera.main.transform.up, -Sensitivity/Time.deltaTime*deltaMousePosition.x, Space.World);
			Vector3 next = SelectedInteraction.transform.eulerAngles;
			if(LockX) {
				next.x = prev.x;
			}
			if(LockY) {
				next.y = prev.y;
			}
			if(LockZ) {
				next.z = prev.z;
			}
			SelectedInteraction.transform.rotation = Quaternion.Slerp(Quaternion.Euler(prev), Quaternion.Euler(next), 1f-Smoothing);
		}
		if(Operation == OPERATION.Scale) {
			Vector2 deltaMousePosition = GetNormalizedMousePosition(Input.mousePosition) - GetNormalizedMousePosition(LastMousePosition);
			SelectedInteraction.transform.localScale += Sensitivity * new Vector3(deltaMousePosition.x, deltaMousePosition.y, 0.1f*Input.mouseScrollDelta.y*Time.deltaTime);
		}
	}

	public void Draw() {
		UltiDraw.Begin();

		if(ProjectionValid) {
			if(ProjectionActive) {
				UltiDraw.DrawWiredSphere(Projection.point, Quaternion.identity, 0.2f, UltiDraw.Black, UltiDraw.Cyan.Transparent(0.5f));
			} else {
				UltiDraw.DrawSphere(Projection.point, Quaternion.identity, 0.2f, UltiDraw.Black);
			}

			if(SelectedInteraction != null) {
				BoxCollider c = (BoxCollider)SelectedInteraction.GetComponent<BoxCollider>();
				UltiDraw.DrawWiredCuboid(c.transform.position + c.transform.rotation * Vector3.Scale(c.transform.lossyScale, c.center), c.transform.rotation, Vector3.Scale(c.transform.lossyScale, c.size), UltiDraw.Black.Transparent(0.25f), UltiDraw.Magenta.Transparent(0.5f));
				if(Operation == OPERATION.Translate) {
					UltiDraw.DrawTranslateGizmo(c.transform.position, c.transform.rotation, 1f);
				}
				if(Operation == OPERATION.Rotate) {
					UltiDraw.DrawRotateGizmo(c.transform.position, c.transform.rotation, 1f);
				}
				if(Operation == OPERATION.Scale) {
					UltiDraw.DrawScaleGizmo(c.transform.position, c.transform.rotation, 1f);
				}
			} else if(Projection.collider is BoxCollider && Projection.collider.isTrigger && Projection.transform.GetComponent<Interaction>() != null) {
				BoxCollider c = (BoxCollider)Projection.collider;
				UltiDraw.DrawWireCuboid(c.transform.position + c.transform.rotation * Vector3.Scale(c.transform.lossyScale, c.center), c.transform.rotation, Vector3.Scale(c.transform.lossyScale, c.size), UltiDraw.Black.Transparent(0.5f));
			}
		}

		if(ActiveInteraction != null) {
			UltiDraw.DrawWireCuboid(ActiveInteraction.GetCenter().GetPosition(), ActiveInteraction.transform.rotation, ActiveInteraction.GetExtents(), UltiDraw.Cyan);
		}

		UltiDraw.End();
	}

	private Vector2 GetNormalizedMousePosition(Vector2 mousePosition) {
		return Camera.main.ScreenToViewportPoint(mousePosition);
	}

	public Interaction GetClosestInteraction(Transform pivot) {
		Interaction[] interactions = GameObject.FindObjectsOfType<Interaction>();
		if(interactions.Length == 0) {
			return null;
		} else {
			List<Interaction> candidates = new List<Interaction>();
			for(int i=0; i<interactions.Length; i++) {
				candidates.Add(interactions[i]);
			}
			if(candidates.Count == 0) {
				return null;
			}
			Interaction closest = candidates[0];
			for(int i=1; i<candidates.Count; i++) {
				if(Vector3.Distance(pivot.position, candidates[i].transform.position) < Vector3.Distance(pivot.position, closest.transform.position)) {
					closest = candidates[i];
				}
			}
			return closest;
		}
	}

	public void SetDefault(Signal signal) {
		foreach(Signal s in Signals) {
			s.Default = false;
		}
		signal.Default = true;
	}

	public Signal AddSignal(string name) {
		Signal signal = new Signal(this, name);
		ArrayExtensions.Add(ref Signals, signal);
		return signal;
	}

	public Signal GetSignal(string name) {
		return System.Array.Find(Signals, x=>x.Name == name);
	}

	public string[] GetSignalNames() {
		string[] names = new string[Signals.Length];
		for(int i=0; i<names.Length; i++) {
			names[i] = Signals[i].Name;
		}
		return names;
	}

	public bool QueryAnyKey() {
		return Input.anyKey;
	}

	public bool QueryKey(KeyCode k) {
		return Input.GetKey(k);
	}

	public bool QueryAnySignal() {
		for(int i=0; i<Signals.Length; i++) {
			if(Signals[i].Query()) {
				return true;
			}
		}
		return false;
	}

	public bool QuerySignal(string name) {
		Signal signal = GetSignal(name);
		return signal == null ? false : signal.Query();
	}

	public Vector3 QueryMove(KeyCode forward, KeyCode back, KeyCode left, KeyCode right) {
		Vector3 move = Vector3.zero;
		if(QueryKey(forward)) {
			move.z += 1f;
		}
		if(QueryKey(back)) {
			move.z -= 1f;
		}
		if(QueryKey(left)) {
			move.x -= 1f;
		}
		if(QueryKey(right)) {
			move.x += 1f;
		}
		return move.normalized;
	}

	public Vector3 QueryMove(KeyCode forward, KeyCode back, KeyCode left, KeyCode right, float[] weights) {
		Vector3 move = Vector3.zero;
		if(QueryKey(forward)) {
			move.z += 1f;
		}
		if(QueryKey(back)) {
			move.z -= 1f;
		}
		if(QueryKey(left)) {
			move.x -= 1f;
		}
		if(QueryKey(right)) {
			move.x += 1f;
		}
		float bias = 0f;
		for(int i=0; i<weights.Length; i++) {
			bias += weights[i] * Signals[i].Velocity;
		}
		return bias * move.normalized;
	}

	public float QueryTurn(KeyCode left, KeyCode right, float weight) {
		float turn = 0f;
		if(QueryKey(left)) {
			turn -= 1f;
		}
		if(QueryKey(right)) {
			turn += 1f;
		}
		return weight * turn;
	}

	public float[] PoolSignals() {
		float[] values = new float[Signals.Length];
		for(int i=0; i<Signals.Length; i++) {
			values[i] = Signals[i].Query() ? 1f : 0f;
		}
		return values;
	}

	public float PoolUserControl(float[] weights) {
		float blending = 0f;
		for(int i=0; i<Signals.Length; i++) {
			blending += weights[i] * Signals[i].UserControl;
		}
		return blending;
	}

	public float PoolNetworkControl(float[] weights) {
		float blending = 0f;
		for(int i=0; i<Signals.Length; i++) {
			blending += weights[i] * Signals[i].NetworkControl;
		}
		return blending;
	}

	public class Signal {
		public bool Default = false;
		public string Name = string.Empty;
		public float Velocity = 1f;
		public float UserControl = 0.5f;
		public float NetworkControl = 0.5f;
		public KeyCode[] Keys = new KeyCode[0];
		public bool[] Negations = new bool[0];

		private Controller Controller;

		public Signal(Controller controller, string name) {
			Controller = controller;
			Name = name;
		}

		public bool Query() {
			if(Default) {
				bool any = false;
				foreach(Signal signal in Controller.Signals) {
					if(!signal.Default && signal.Query()) {
						any = true;
						break;
					}
				}
				if(!any) {
					return true;
				}
			}

			if(Keys.Length == 0) {
				return false;
			}

			bool active = false;

			for(int i=0; i<Keys.Length; i++) {
				if(!Negations[i]) {
					if(Keys[i] == KeyCode.None) {
						if(!Controller.QueryAnyKey()) {
							active = true;
						}
					} else {
						if(Controller.QueryKey(Keys[i])) {
							active = true;
						}
					}
				}
			}

			for(int i=0; i<Keys.Length; i++) {
				if(Negations[i]) {
					if(Keys[i] == KeyCode.None) {
						if(!Controller.QueryAnyKey()) {
							active = false;
						}
					} else {
						if(Controller.QueryKey(Keys[i])) {
							active = false;
						}
					}
				}
			}

			return active;
		}

		public void AddKey(KeyCode key, bool positiveOrNegative) {
			ArrayExtensions.Add(ref Keys, key);
			ArrayExtensions.Add(ref Negations, !positiveOrNegative);
		}
	}

}
