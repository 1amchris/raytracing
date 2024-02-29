#include "AxisAlignedBounds.h"
#include "Utils/Maths.hpp"

AxisAlignedBounds::AxisAlignedBounds(glm::vec3& a, glm::vec3& b):
    m_Min(glm::vec3{ glm::min(a.x, b.x), glm::min(a.y, b.y), glm::min(a.z, b.z), }),
    m_Max(glm::vec3{ glm::max(a.x, b.x), glm::max(a.y, b.y), glm::max(a.z, b.z), })
{
}

bool AxisAlignedBounds::Intersect(const Ray& ray) const
{
    if (IsParallelAndOutside(ray, [](glm::vec3 v) { return v.x; })
        || IsParallelAndOutside(ray, [](glm::vec3 v) { return v.y; })
        || IsParallelAndOutside(ray, [](glm::vec3 v) { return v.z; }))
        return false;

    Range result{};
    result.Min = std::numeric_limits<float>::min();
    result.Max = std::numeric_limits<float>::max();

    if (!(FitRangeToInclude(ray, &result, [](glm::vec3 v) { return v.x; })
        && FitRangeToInclude(ray, &result, [](glm::vec3 v) { return v.y; })
        && FitRangeToInclude(ray, &result, [](glm::vec3 v) { return v.z; })))
        return false;

    return result.Max > 1e-4f && result.Min <= result.Max;
}

template<typename Func>
const AxisAlignedBounds::Range AxisAlignedBounds::ComputeRange(const Ray& ray, Func member) const
{
    Range result{};
    result.Min = (member(m_Min) - member(ray.Origin)) / member(ray.Direction);
    result.Max = (member(m_Max) - member(ray.Origin)) / member(ray.Direction);

    if (result.Min > result.Max) 
        Utils::SwapMemory(&result.Min, &result.Max, sizeof(float));

    return result;
}

template<typename Func>
const bool AxisAlignedBounds::IsParallelAndOutside(const Ray& ray, Func member) const
{
    return member(ray.Direction) == 0 
        && (member(ray.Origin) < member(m_Min) || member(m_Max) < member(ray.Origin));
}

template<typename Func>
const bool AxisAlignedBounds::FitRangeToInclude(const Ray& ray, Range* range, Func member) const
{
    if (member(ray.Direction) != 0)
    {
        Range t = ComputeRange(ray, member);
        if (t.Min < 0 && t.Max < 0)
            return false;

        range->Min = glm::max(range->Min, t.Min);
        range->Max = glm::min(range->Max, t.Max);
    }

    return true;
}
