# Source this to load your BadgerBrain gateway key into the current shell.
#
#   source env.sh
#
# Contains a 1Password *reference*, not a secret -- safe to keep on disk.
# The key itself never leaves 1Password.

export OPENAI_API_KEY=$(op read 'op://Employee/MLM26-Bookworm_jsun424/credential')

if [ -n "$OPENAI_API_KEY" ]; then
  echo "badgerchat: key loaded (${OPENAI_API_KEY:0:6}...)"
else
  echo "badgerchat: key NOT loaded -- is 1Password unlocked?" >&2
fi
