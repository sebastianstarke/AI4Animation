using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace SIGGRAPH_2018
{
    [RequireComponent(typeof(Text))]
    public class ColorChanger : MonoBehaviour
    {
        public AnimationAuthoring AnimationAuthoring;
        public AnimationAuthoring.StyleNames Style;
        void Start()
        {
            UpdateColor();
        }

        void Update()
        {
            UpdateColor();
        }

        private void UpdateColor()
        {
            Color col = AnimationAuthoring.StyleColors[(int)Style];
            GetComponent<Text>().color = col;
        }
    }
}
