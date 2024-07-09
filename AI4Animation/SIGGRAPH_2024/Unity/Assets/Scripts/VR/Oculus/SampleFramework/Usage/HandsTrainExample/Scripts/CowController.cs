/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using UnityEngine.Assertions;

namespace OculusSampleFramework
{
	public class CowController : MonoBehaviour
	{
		[SerializeField] private Animation _cowAnimation = null;
		[SerializeField] private AudioSource _mooCowAudioSource = null;

		private void Start()
		{
			Assert.IsNotNull(_cowAnimation);
			Assert.IsNotNull(_mooCowAudioSource);
		}

		public void PlayMooSound()
		{
			_mooCowAudioSource.timeSamples = 0;
			_mooCowAudioSource.Play();
		}

		public void GoMooCowGo()
		{
			_cowAnimation.Rewind();
			_cowAnimation.Play();
		}
	}
}
