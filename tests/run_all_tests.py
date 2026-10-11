"""
Master Test Runner for Edge-Based Visual Odometry Suite.
Executes all unit and integration test scripts sequentially and reports results.
"""

import os
import subprocess
import sys
import time

TEST_SCRIPTS = [
    "test_distance_transforms.py",
    "test_rgb_edge_detection.py",
    "test_depth_edge_detection.py",
    "test_scene_study.py",
    "test_dynamic_objects.py",
    "test_pipeline.py"
]


def run_test_suite():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 70)
    print("       EDGE-BASED VISUAL ODOMETRY - MASTER TEST RUNNER")
    print("=" * 70)
    print(f"Discovered {len(TEST_SCRIPTS)} test suites in {tests_dir}\n")

    results = {}
    start_all = time.time()

    for script in TEST_SCRIPTS:
        script_path = os.path.join(tests_dir, script)
        print(f"\n>>> Running suite: {script} ...")
        t0 = time.time()
        ret = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        elapsed = time.time() - t0

        if ret.returncode == 0:
            print(ret.stdout.strip())
            print(f"[{script}] RESULT: PASSED ({elapsed:.2f}s)")
            results[script] = ("PASS", elapsed)
        else:
            print(ret.stdout.strip())
            print(ret.stderr.strip())
            print(f"[{script}] RESULT: FAILED ({elapsed:.2f}s)")
            results[script] = ("FAIL", elapsed)

    total_time = time.time() - start_all
    print("\n" + "=" * 70)
    print("                      TEST EXECUTION SUMMARY")
    print("=" * 70)
    all_passed = True
    for script, (status, duration) in results.items():
        status_str = "[ PASS ]" if status == "PASS" else "[ FAIL ]"
        print(f"  {status_str} {script:<35} {duration:>6.2f}s")
        if status != "PASS":
            all_passed = False

    print("-" * 70)
    if all_passed:
        print(f"Status: ALL {len(TEST_SCRIPTS)} TEST SUITES PASSED! (Total time: {total_time:.2f}s)")
    else:
        print(f"Status: SOME TESTS FAILED. (Total time: {total_time:.2f}s)")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_test_suite())
