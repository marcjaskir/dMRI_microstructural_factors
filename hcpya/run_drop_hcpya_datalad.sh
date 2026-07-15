#!/usr/bin/expect
# Auto-resolved paths via lib/paths.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
BASE="${DMRI_MICRO_ROOT:-$(python3 -c "import sys; sys.path.insert(0, \"$REPO_ROOT\"); from lib.paths import project_root; print(project_root())")}"

# Set timeout for responses
set timeout -1

# Spawn the shell script that includes datalad get
spawn drop_hcpya_datalad.sh

expect {
    "key_id:" {
        send "AKIAXO65CT57CXVQ46NU\r"
        exp_continue
    }
    "secret_id:" {
        send "vTMK9qNOt01wJwH2FaLkB9wV7aU3dhWCK8qY9MpF\r"
        exp_continue
    }
    eof {
        # End of file (script finished)
        puts "Script execution completed."
    }
}