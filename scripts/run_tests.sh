#!/usr/bin/env bash
set -e
echo "1️⃣  Helpers";              pytest -q tests/test_utils.py tests/test_helpers.py
echo "2️⃣  Dynamics";             pytest -q tests/test_cartpole.py
echo "3️⃣  Closed-loop";          pytest -q tests/test_closedloop.py
echo "4️⃣  Base ctrl";            pytest -q tests/test_base_controller.py
echo "5️⃣  Linear ctrl";          pytest -q tests/test_linear_controller.py
echo "6️⃣  Advanced training";    pytest -q tests/test_advanced_training.py
echo "7️⃣  Energy & memory";      pytest -q tests/test_energy.py
echo "8️⃣  Integration";          pytest -q tests/test_integration.py
echo "9️⃣  Visualiser";           pytest -q tests/test_visualizer.py
echo "🔟  Conftest/project";      pytest -q tests/conftest.py
echo "1️⃣1️⃣  LQR ctrl";           pytest -q tests/test_lqr_controller.py
echo "1️⃣2️⃣  NN ctrl";            pytest -q tests/test_nn_controller.py
echo "1️⃣3️⃣  Trainer smoke";      pytest -q tests/test_trainer.py
echo "✅  All layers green!"