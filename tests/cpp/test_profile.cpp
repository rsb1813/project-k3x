// 런타임 프로파일 이벤트의 결정적 집계를 검증합니다.
#include "k3x/profile.hpp"

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
    if (events.size() != 3) return 1;
    if (events.front().precision != k3x::NumericPrecision::fp32) return 2;
    if (events.back().precision != k3x::NumericPrecision::mxfp4_e2m1_e8m0) return 3;

    const auto summary = profiler.summary();
    if (summary.wall_nanoseconds != 130) return 4;
    if (summary.device_nanoseconds != 100) return 5;
    if (summary.logical_bytes != 64) return 6;
    if (summary.host_to_device_bytes != 256) return 7;
    if (summary.device_to_host_bytes != 0) return 8;
    if (summary.failed_operations != 1) return 9;
    return 0;
}
