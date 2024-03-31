#pragma once

#include "Hardware.hpp"

#pragma pack(1)
template<typename T>
class Interval {
public:
	__TARGET_ALL__
	static bool Contains(Interval i, T value) {
		return i.Min <= value && value <= i.Max;
	}

	__TARGET_ALL__
	static bool StrictlyInside(Interval i, T value) {
		return i.Min < value && value < i.Max;
	}

	__TARGET_ALL__
	static bool StrictlyOutside(Interval i, T value) {
		return value < i.Min || i.Max < value;
	}

public:
	const T Min, Max;
};