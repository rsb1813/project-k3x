// 런타임 프로파일 이벤트의 결정적 집계를 검증합니다.
#include "k3x/profile.hpp"

#include <cassert>

int main() {
    k3x::Profiler profiler;
    profiler.record({k3x::ProfilePhase::prefill,
                     k3x::ProfileOperation::dense_matvec,
                     k3x::NumericPrecision::fp32,
                     0,
                     100,
                     80,
                     64,
                     0,
                     true});
    profiler.record({k3x::ProfilePhase::decode,
                     k3x::ProfileOperation::host_to_device,
                     k3x::NumericPrecision::none,
                     k3x::profile_global_layer,
                     30,
                     20,
                     0,
                     256,
                     true});
    profiler.record({k3x::ProfilePhase::decode,
                     k3x::ProfileOperation::mxfp4_matvec,
                     k3x::NumericPrecision::mxfp4_e2m1_e8m0,
                     1,
                     999,
                     888,
                     128,
                     0,
                     false});

    const auto& events = profiler.events();
    assert(events.size() == 3);
    assert(events.front().precision == k3x::NumericPrecision::fp32);
    assert(events.back().precision == k3x::NumericPrecision::mxfp4_e2m1_e8m0);

    const auto summary = profiler.summary();
    assert(summary.wall_nanoseconds == 130);
    assert(summary.device_nanoseconds == 100);
    assert(summary.logical_bytes == 64);
    assert(summary.host_to_device_bytes == 256);
    assert(summary.device_to_host_bytes == 0);
    assert(summary.failed_operations == 1);
}
