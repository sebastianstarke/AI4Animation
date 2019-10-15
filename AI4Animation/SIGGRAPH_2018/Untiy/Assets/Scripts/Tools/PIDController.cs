using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PIDController {
	public float PGain;
	public float IGain;
	public float DGain;

	public float Value;

	private float Error;
	private float Integrator;
	private float Differentiator;
	private float LastError;

	public PIDController(float P, float I, float D) {
		SetParameters(P, I, D);
	}

	public void SetParameters(float P, float I, float D) {
		PGain = P;
		IGain = I;
		DGain = D;
	}

	public float Update(float target, float current, float update) {
		Error = target-current;
			
		Integrator += Error*update;

		Differentiator = (Error-LastError)/update;

		LastError = Error;

		Value = Error*PGain + Integrator*IGain + Differentiator*DGain;
		return Value;
	}

	public void Reset(float error = 0f, float integrator = 0f, float differentiator = 0f, float lastError = 0f) {
		Error = error;
		Integrator = integrator;
		Differentiator = differentiator;
		LastError = lastError;
	}
}