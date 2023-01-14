using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Example_Serialization : MonoBehaviour {

    public IndividualClass Individual;
    public DerivedClass Derived;

    [Serializable]
    public class IndividualClass {
        public string Variable = string.Empty;
    }

    public abstract class BaseClass {
        public string BaseClassVariable = string.Empty;
    }

    [Serializable]
    public class DerivedClass : BaseClass {
        public string DerivedVariable = string.Empty;
    }

}
