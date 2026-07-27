#!/usr/bin/expect
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
while [ ! -f "$CODE_ROOT/lib/paths.py" ] && [ "$CODE_ROOT" != "/" ]; do
  CODE_ROOT="$(dirname "$CODE_ROOT")"
done
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$CODE_ROOT\"); from lib.paths import project_root; print(project_root())")}"

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
