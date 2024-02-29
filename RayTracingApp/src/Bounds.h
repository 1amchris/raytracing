#pragma once

#include "Ray.h"

class Bounds
{
public:
	virtual bool Intersect(const Ray& ray) const = 0;
};