#pragma once

#include "Bounds.h"

class AxisAlignedBounds : public Bounds 
{
public:
	AxisAlignedBounds(glm::vec3& a, glm::vec3& b);

	virtual bool Intersect(const Ray& ray) const override;

private:
	struct Range {
		float Min, Max;
	};

	template<typename Func>
	const Range ComputeRange(const Ray& ray, Func member) const;

	template<typename Func>
	const bool IsParallelAndOutside(const Ray& ray, Func member) const;

	template<typename Func>
	const bool FitRangeToInclude(const Ray& ray, Range* range, Func member) const;

private:
	glm::vec3 m_Min, m_Max;
};