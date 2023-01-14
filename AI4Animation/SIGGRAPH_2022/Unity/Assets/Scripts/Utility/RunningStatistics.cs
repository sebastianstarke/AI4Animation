using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RunningStatistics {
	private float m_sum;
	private int m_n;
	private float m_oldM;
	private float m_newM;
	private float m_oldS;
	private float m_newS;
	
	public RunningStatistics() {
		Clear();
	}
	
	public void Clear() {
		m_sum = 0f;
		m_n = 0;
	}
	
	public void Add(float[] samples) {
		for(int i=0; i<samples.Length; i++) {
			Add(samples[i]);
		}
	}

	public void Add(float sample) {
		m_sum += sample;
        m_n++;
        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if (m_n == 1) {
            m_oldM = m_newM = sample;
            m_oldS = 0f;
        } else {
            m_newM = m_oldM + (sample - m_oldM)/m_n;
            m_newS = m_oldS + (sample - m_oldM)*(sample - m_newM);
            // Set up for next iteration
            m_oldM = m_newM; 
            m_oldS = m_newS;
        }
	}
	
	public int Count() {
		return m_n;
	}
	
	public float Sum() {
		return m_sum;
	}

	public float Mean() {
		return m_n > 0 ? m_newM : 0f;
	}

	public float Var() {
		return m_n > 1 ? m_newS/m_n : 0f;
	}
	
	public float Sigma() {
		return Mathf.Sqrt(Var());
	}	
}
