#!/usr/bin/expect
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

# Set timeout for responses
set timeout -1

# Spawn the shell script that includes datalad get
spawn /Users/mjaskir/cnt/data/borel/sauce/littlab/users/mjaskir/structural_tractometry/code/hcpya/move_hcpya_qsirecon_fib.sh

expect {
    "jaskirm@cubic-login's password:" {
        send "Letmeinplzkthx@\$25\r"
        exp_continue
    }
    eof {
        # End of file (script finished)
        puts "Script execution completed."
    }
}
