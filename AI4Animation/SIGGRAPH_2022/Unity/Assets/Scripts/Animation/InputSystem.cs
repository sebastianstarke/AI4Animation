using System;
using UnityEngine;

public class InputSystem {

	public enum ControlType { Controller, Keyboard };
	public ControlType Control = ControlType.Controller;

	public enum Button {A, B, X, Y, LB, RB};

	public bool Visualise = true;

	public int ID = 0;

	public Logic[] Logics = new Logic[0];
	public Value[] Values = new Value[0];
	public Function[] Functions = new Function[0];

	//Listeners
	private Vector3 ControllerDPad = Vector3.zero;
	private Vector3 LeftControllerVector = Vector3.zero;
	private Vector3 LeftControllerMomentum = Vector3.zero;
	private Vector3 RightControllerVector = Vector3.zero;
	private Vector3 RightControllerMomentum = Vector3.zero;
	private float RadialController = 0f;
	private float TriggerController = 0f;
	private float ButtonController = 0f;
	private bool ButtonA, ButtonB, ButtonX, ButtonY, ButtonLB, ButtonRB;
	private bool ButtonDownA, ButtonDownB, ButtonDownX, ButtonDownY, ButtonDownLB, ButtonDownRB;
	//

	private Vector3 LastMousePosition = new Vector3(0f, 0f, 0f);

	public class Logic {
		public string Name = string.Empty;
		public Func<bool> Query = () => false;
		public Logic(string name, Func<bool> func) {
			Name = name;
			Query = func;
		}
	}

	public class Value {
		public string Name = string.Empty;
		public Func<float> Query = () => 0f;
		public Value(string name, Func<float> func) {
			Name = name;
			Query = func;
		}
	}

	public class Function {
		public string Name = string.Empty;
		public Func<float, float> Query = (x) => 0f;
		public Function(string name, Func<float, float> func) {
			Name = name;
			Query = func;
		}
	}

	public InputSystem(int id) {
		ID = id;
	}

	public void Update() {
		//DPad
		{
			switch(Control) {
				 case ControlType.Controller:
					ControllerDPad = new Vector3(Input.GetAxis(ID + "LeftRight"), 0f, Input.GetAxis(ID + "UpDown"));
					break;
				case ControlType.Keyboard:
					// TODO: change this!!
					ControllerDPad = new Vector3(0f, 0f, 0f);
					break;
			}
		}

		//Left Joystick
		{
			Vector3 previous = LeftControllerVector;
			switch(Control) {
				 case ControlType.Controller:
					LeftControllerVector = new Vector3(Input.GetAxis(ID+"X"), Input.GetButton(ID+"XY") ? 1f : 0f, Input.GetAxis(ID+"Y")).ClampMagnitudeXZ(1f);
					break;
				case ControlType.Keyboard:
					LeftControllerVector = new Vector3(0f, 0f, 0f);
					if(Input.GetKey(KeyCode.W)) {
						LeftControllerVector.z += 1.0f;
					}
					if(Input.GetKey(KeyCode.S)) {
						LeftControllerVector.z -= 1.0f;
					}
					if(Input.GetKey(KeyCode.A)) {
						LeftControllerVector.x -= 1.0f;
					}
					if(Input.GetKey(KeyCode.D)) {
						LeftControllerVector.x += 1.0f;
					}
					if(Input.GetKey(KeyCode.LeftShift)) {
						LeftControllerVector.y = 1.0f;
					}
					LeftControllerVector = LeftControllerVector.ClampMagnitudeXZ(1f);
					break;
			}
			LeftControllerMomentum = LeftControllerVector - previous;
		}

		//Right Joystick
		{
			Vector3 previous = RightControllerVector;
			switch(Control) {
				case ControlType.Controller:
					RightControllerVector = new Vector3(Input.GetAxis(ID+"H"), Input.GetButton(ID+"HV") ? 1f : 0f, -Input.GetAxis(ID+"V")).ClampMagnitudeXZ(1f);
					break;
				case ControlType.Keyboard:
					Vector3 mouseUpdate = Input.mousePosition - LastMousePosition;
					if(mouseUpdate.magnitude == 0.0f){
						RightControllerVector = Vector3.Lerp(RightControllerVector, Vector3.zero, 0.1f);
					}
					else{
						RightControllerVector.x += 5f*mouseUpdate.x / Screen.width;
						RightControllerVector.z += 5f*mouseUpdate.y / Screen.width;
						RightControllerVector = RightControllerVector.ClampMagnitudeXZ(1f);
					}
					break;
			}
			RightControllerMomentum = RightControllerVector - previous;
		}

		//Heuristic Values
		switch(Control) {
			case ControlType.Controller:
				float threshold = 0.9f;
				float power = 2f;
				float length = Vector3.ClampMagnitude(LeftControllerVector.ZeroY(), 1f).magnitude.SmoothStep(power, threshold);
				float ratio = Vector3.SignedAngle((LeftControllerVector-LeftControllerMomentum).ZeroY(), LeftControllerVector.ZeroY(), Vector3.up) / 180f;
				RadialController += length * ratio; //Increase
				RadialController *= threshold; //Dampening
				break;
			case ControlType.Keyboard:
				float weight = 0.1f;
				if(Input.GetKey(KeyCode.Q)) {
					RadialController = Mathf.Lerp(RadialController, -1.0f, weight);
				}
				else if(Input.GetKey(KeyCode.E)) {
					RadialController = Mathf.Lerp(RadialController, 1.0f, weight);
				}
				else {
					RadialController = Mathf.Lerp(RadialController, 0.0f, weight);
				}
				break;
		}

		//Trigger Axis
		switch(Control) {
			case ControlType.Controller:
				TriggerController = Input.GetAxis(ID+"LTRT");
				break;
			case ControlType.Keyboard:
				if(Input.GetKey(KeyCode.O)) {
					TriggerController = -1.0f;
				}
				else if (Input.GetKey(KeyCode.P)) {
					TriggerController = 1.0f;
				}
				else {
					TriggerController = 0.0f;
				}
				break;
		}

		//Trigger Button
		switch(Control) {
			case ControlType.Controller:
				ButtonController = (Input.GetButton(ID+"LB") ? -1f : 0f) + (Input.GetButton(ID+"RB") ? 1f : 0f);
				break;
			case ControlType.Keyboard:
				if(Input.GetKey(KeyCode.K)) {
					ButtonController = -1.0f;
				}
				else if (Input.GetKey(KeyCode.L)) {
					ButtonController = 1.0f;
				}
				else {
					ButtonController = 0.0f;
				}
				break;
		}

		//Buttons
		switch(Control) {
			case ControlType.Controller:
				ButtonX = Input.GetButton(ID + "Button" + Button.X.ToString());
				ButtonY = Input.GetButton(ID + "Button" + Button.Y.ToString());
				ButtonA = Input.GetButton(ID + "Button" + Button.A.ToString());
				ButtonB = Input.GetButton(ID + "Button" + Button.B.ToString());
				ButtonLB = Input.GetButton(ID + "Button" + Button.LB.ToString());
				ButtonRB = Input.GetButton(ID + "Button" + Button.RB.ToString());
				ButtonDownX = Input.GetButtonDown(ID + "Button" + Button.X.ToString());
				ButtonDownY = Input.GetButtonDown(ID + "Button" + Button.Y.ToString());
				ButtonDownA = Input.GetButtonDown(ID + "Button" + Button.A.ToString());
				ButtonDownB = Input.GetButtonDown(ID + "Button" + Button.B.ToString());
				ButtonDownLB = Input.GetButtonDown(ID + "Button" + Button.LB.ToString());
				ButtonDownRB = Input.GetButtonDown(ID + "Button" + Button.RB.ToString());
				break;
			case ControlType.Keyboard:
				
				break;
		}

		LastMousePosition = Input.mousePosition;
	}

	public void Draw() {
		UltiDraw.Begin();
		Vector2 center = new Vector2(0.5f, 0.1f);
		Vector2 size = new Vector2(0.5f, 0.1f);
		UltiDraw.GUIFrame(new Vector2(0.5f, 0.1f), new Vector2(0.5f, 0.1f), 0.0025f, UltiDraw.DarkGrey);
		UltiDraw.GUIRectangle(new Vector2(0.5f, 0.1f), new Vector2(0.5f, 0.1f), UltiDraw.White.Opacity(0.5f));
		float start = center.x - 0.5f*size.x + 0.5f*size.x/Logics.Length;
		float width = size.x/Logics.Length;
		for(int i=0; i<Logics.Length; i++) {
			Vector2 c = new Vector2(start + i*width, center.y);
			Vector2 s = new Vector2(0.9f*width, 0.75f*size.y);
			UltiDraw.GUIRectangle(c, s, Logics[i].Query() ? UltiDraw.Mustard.Opacity(0.5f) : UltiDraw.DarkGrey.Opacity(0.5f));
			//UltiDraw.PlotHorizontalBar(new Vector2(c.x, c.y + 0.5f*0.875f*size.y), new Vector2(s.x, 0.125f*s.y), Logics[i].UserControl(), backgroundColor: UltiDraw.DarkGrey, fillColor: UltiDraw.Cyan);
			//UltiDraw.PlotHorizontalBar(new Vector2(c.x, c.y - 0.5f*0.875f*size.y), new Vector2(s.x, 0.125f*s.y), Logics[i].NetworkControl(), backgroundColor: UltiDraw.DarkGrey, fillColor: UltiDraw.Orange);
		}
		UltiDraw.GUIFrame(new Vector2(0.5f, 0.1f), new Vector2(0.5f, 0.1f), 0.0025f, UltiDraw.DarkGrey);
		UltiDraw.End();
	}

	public void GUI() {
		UltiDraw.Begin();
		Vector2 center = new Vector2(0.5f, 0.1f);
		Vector2 size = new Vector2(0.5f, 0.1f);
		float start = center.x - 0.5f*size.x + 0.5f*size.x/Logics.Length;
		float width = size.x/Logics.Length;
		for(int i=0; i<Logics.Length; i++) {
			Vector2 c = new Vector2(start + i*width, center.y);
			Vector2 s = new Vector2(0.9f*width, 0.75f*size.y);
			UltiDraw.OnGUILabel(c, s, 0.01f, Logics[i].Name, UltiDraw.Black, UltiDraw.White.Opacity(0.25f));
		}
		UltiDraw.End();
	}

	public Vector2 GetMouseDeltaCoordinates() {
		return Camera.main.ScreenToViewportPoint(Input.mousePosition) - Camera.main.ScreenToViewportPoint(LastMousePosition);
	}

	public static Vector2 GetMouseCoordinates() {
		return Camera.main.ScreenToViewportPoint(Input.mousePosition);
	}

	public Logic AddLogic(string name, Func<bool> func) {
		Logic item = System.Array.Find(Logics, x=>x.Name == name);
		if(item != null) {
			Debug.Log("Logic with name " + name + " already contained.");
		} else {
			item = new Logic(name, func);
			ArrayExtensions.Append(ref Logics, item);
		}
		return item;
	}

	public bool QueryLogic(string name) {
		Logic item = System.Array.Find(Logics, x=>x.Name == name);
		if(item == null) {
			Debug.Log("Logic with name " + name + " could not be found.");
			return false;
		}
		return item.Query();
	}

	public bool[] QueryLogics(string[] names) {
		bool[] items = new bool[names.Length];
		for(int i=0; i<names.Length; i++) {
			items[i] = QueryLogic(names[i]);
		}
		return items;
	}

	public float[] PoolLogics(string[] names) {
		float[] items = new float[names.Length];
		for(int i=0; i<names.Length; i++) {
			items[i] = QueryLogic(names[i]) ? 1f : 0f;
		}
		return items;
	}

	public Value AddValue(string name, Func<float> func) {
		Value value = System.Array.Find(Values, x=>x.Name == name);
		if(value != null) {
			Debug.Log("Value with name " + name + " already contained.");
		} else {
			value = new Value(name, func);
			ArrayExtensions.Append(ref Values, value);
		}
		return value;
	}

	public float QueryValue(string name) {
		Value value = System.Array.Find(Values, x=>x.Name == name);
		if(value == null) {
			Debug.Log("Value with name " + name + " could not be found.");
			return 0f;
		}
		return value.Query();
	}

	public float[] QueryValues(string[] names) {
		float[] items = new float[names.Length];
		for(int i=0; i<names.Length; i++) {
			items[i] = QueryValue(names[i]);
		}
		return items;
	}

	public float[] PoolValues(string[] names) {
		float[] items = new float[names.Length];
		for(int i=0; i<names.Length; i++) {
			items[i] = QueryValue(names[i]);
		}
		return items;
	}

	public Function AddFunction(string name, Func<float, float> func) {
		Function function = System.Array.Find(Functions, x=>x.Name == name);
		if(function != null) {
			Debug.Log("Function with name " + name + " already contained.");
		} else {
			function = new Function(name, func);
			ArrayExtensions.Append(ref Functions, function);
		}
		return function;
	}

	public float QueryFunction(string name, float arg) {
		Function function = System.Array.Find(Functions, x=>x.Name == name);
		if(function == null) {
			Debug.Log("Function with name " + name + " could not be found.");
			return 0f;
		}
		return function.Query(arg);
	}

	public Vector3 QueryMoveKeyboard() {
		Vector3 move = Vector3.zero;
		if(Input.GetKey(KeyCode.W)) {
			move.z += 1f;
		}
		if(Input.GetKey(KeyCode.S)) {
			move.z -= 1f;
		}
		if(Input.GetKey(KeyCode.A)) {
			move.x -= 1f;
		}
		if(Input.GetKey(KeyCode.D)) {
			move.x += 1f;
		}
		return move.normalized;
	}

	public float QueryTurnKeyboard() {
		float turn = 0f;
		if(Input.GetKey(KeyCode.Q)) {
			turn -= 1f;
		}
		if(Input.GetKey(KeyCode.E)) {
			turn += 1f;
		}
		return turn;
	}

	public Vector3 QueryDPadController() {
		return ControllerDPad;
	}

	public float QueryRadialController() {
		return RadialController;
	}
	
	public float QueryTriggerController() {
		return TriggerController;
	}

	public float QueryButtonController() {
		return ButtonController;
	}

	public Vector3 QueryLeftJoystickVector() {
		return LeftControllerVector;
	}

	public Vector3 QueryLeftJoystickMomentum() {
		return LeftControllerMomentum;
	}

	public Vector3 QueryRightJoystickVector() {
		return RightControllerVector;
	}

	public Vector3 QueryRightJoystickMomentum() {
		return RightControllerMomentum;
	}

	public bool QueryButton(Button button) {
		switch(button) {
			case Button.A:
			return ButtonA;
			case Button.B:
			return ButtonB;
			case Button.X:
			return ButtonX;
			case Button.Y:
			return ButtonY;
			case Button.LB:
			return ButtonLB;
			case Button.RB:
			return ButtonRB;
		}
		Debug.LogError("Undefined control type " + button.ToString() + " for " + Control.ToString());
		return false;
	}

	public bool QueryButtonDown(Button button) {
		switch(button) {
			case Button.A:
			return ButtonDownA;
			case Button.B:
			return ButtonDownB;
			case Button.X:
			return ButtonDownX;
			case Button.Y:
			return ButtonDownY;
			case Button.LB:
			return ButtonDownLB;
			case Button.RB:
			return ButtonDownRB;
		}
		Debug.LogError("Undefined control type " + button.ToString() + " for " + Control.ToString());
		return false;
	}

}
