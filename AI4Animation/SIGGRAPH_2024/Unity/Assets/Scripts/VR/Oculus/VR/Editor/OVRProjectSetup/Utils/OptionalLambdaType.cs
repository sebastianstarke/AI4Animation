/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Generic;

internal abstract class OptionalLambdaType<TLambdaArgumentType, TValueType>
{
	public static OptionalLambdaType<TLambdaArgumentType, TValueType> Create(TValueType value,
		Func<TLambdaArgumentType, TValueType> lambda, bool allowCache)
    {
        OptionalLambdaType<TLambdaArgumentType, TValueType> optionalLambdaType = null;
        if (lambda != null)
        {
	        if (allowCache)
	        {
		        optionalLambdaType = new OptionalLambdaTypeWithCachedLambda<TLambdaArgumentType, TValueType>(lambda);
	        }
	        else
	        {
		        optionalLambdaType = new OptionalLambdaTypeWithLambda<TLambdaArgumentType, TValueType>(lambda);
	        }
        }
        else
        {
	        optionalLambdaType = new OptionalLambdaTypeWithoutLambda<TLambdaArgumentType, TValueType>(value);
        }

        return optionalLambdaType;
    }

    public static implicit operator OptionalLambdaType<TLambdaArgumentType, TValueType>(TValueType value)
    {
        var optionalLambdaType = new OptionalLambdaTypeWithoutLambda<TLambdaArgumentType, TValueType>(value);
        return optionalLambdaType;
    }

    public static implicit operator OptionalLambdaType<TLambdaArgumentType, TValueType>(Func<TLambdaArgumentType, TValueType> lambda)
    {
        var optionalLambdaType = new OptionalLambdaTypeWithLambda<TLambdaArgumentType, TValueType>(lambda);
        return optionalLambdaType;
    }

    public abstract bool Valid { get; }
    public abstract TValueType GetValue(TLambdaArgumentType arg);
    public abstract TValueType Default { get; }
    public virtual void InvalidateCache(TLambdaArgumentType arg) {}
}

internal class OptionalLambdaTypeWithoutLambda<TLambdaArgumentType, TValueType> : OptionalLambdaType<TLambdaArgumentType, TValueType>
{
    private readonly TValueType _value;

    public OptionalLambdaTypeWithoutLambda(TValueType value)
    {
        _value = value;
    }

    public override TValueType GetValue(TLambdaArgumentType arg) => _value;
    public override TValueType Default => _value;
    public override bool Valid => _value != null;
}

internal class OptionalLambdaTypeWithLambda<TLambdaArgumentType, TValueType> : OptionalLambdaType<TLambdaArgumentType, TValueType>
{
    protected readonly Func<TLambdaArgumentType, TValueType> Lambda;

    public OptionalLambdaTypeWithLambda(Func<TLambdaArgumentType, TValueType> lambda)
    {
	    Lambda = lambda;
    }

    public override TValueType GetValue(TLambdaArgumentType arg) => Lambda.Invoke(arg);

    public override TValueType Default => Lambda.Invoke(default(TLambdaArgumentType));
    public override bool Valid => Lambda != null && Default != null;
}

internal class OptionalLambdaTypeWithCachedLambda<TLambdaArgumentType, TValueType> :
	OptionalLambdaTypeWithLambda<TLambdaArgumentType, TValueType>
{
	private readonly Dictionary<TLambdaArgumentType, TValueType> _cachedValues =
		new Dictionary<TLambdaArgumentType, TValueType>();

	public OptionalLambdaTypeWithCachedLambda(Func<TLambdaArgumentType, TValueType> lambda) : base(lambda)
	{
	}

	public override TValueType GetValue(TLambdaArgumentType arg)
	{
		if (!_cachedValues.TryGetValue(arg, out var value))
		{
			value = base.GetValue(arg);
			_cachedValues.Add(arg, value);
		}

		return value;
	}

	public override void InvalidateCache(TLambdaArgumentType arg)
	{
		_cachedValues.Remove(arg);
	}
}
